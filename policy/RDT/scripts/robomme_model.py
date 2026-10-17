import os
import hashlib

import numpy as np
import torch
from PIL import Image

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.checkpoint_utils import load_rdt_state_dict
from models.lora import load_lora_adapter
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner


ROBOMME_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [
    STATE_VEC_IDX_MAPPING["right_gripper_open"]
]


class RobommeRDTModel:
    def __init__(
        self,
        config,
        pretrained_model_path,
        pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path,
        device="cuda",
        dtype=torch.bfloat16,
        control_frequency=10,
        load_text_encoder=True,
        lora_adapter_path=None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.control_frequency = control_frequency

        self.text_tokenizer, self.text_model = (
            self._get_text_encoder(pretrained_text_encoder_name_or_path)
            if load_text_encoder
            else (None, None)
        )
        self.image_processor, self.vision_model = self._get_vision_encoder(
            pretrained_vision_encoder_name_or_path
        )
        self.policy = self._build_policy()
        load_rdt_state_dict(self.policy, pretrained_model_path, strict=False)
        if lora_adapter_path is not None:
            load_lora_adapter(self.policy, lora_adapter_path)
        self.reset()

    def _get_text_encoder(self, pretrained_text_encoder_name_or_path):
        embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.config["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        return embedder.tokenizer, embedder.model

    def _get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(
            vision_tower=pretrained_vision_encoder_name_or_path,
            args=None,
        )
        return vision_encoder.image_processor, vision_encoder

    def _build_policy(self):
        img_cond_len = (
            self.config["common"]["img_history_size"]
            * self.config["common"]["num_cameras"]
            * self.vision_model.num_patches
        )
        return RDTRunner(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.config["common"]["action_chunk_size"],
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        self.config["common"]["img_history_size"],
                        self.config["common"]["num_cameras"],
                        -self.vision_model.num_patches,
                    ),
                ),
            ],
            lang_pos_embed_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )

    def reset(self):
        self.policy.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(self.device, dtype=self.dtype)
        if self.text_model is not None:
            self.text_model.eval()
            self.text_model = self.text_model.to(self.device, dtype=self.dtype)
        self.vision_model = self.vision_model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_instruction(self, instruction):
        if self.text_model is None or self.text_tokenizer is None:
            raise RuntimeError("Text encoder is disabled; provide precomputed language embeddings.")
        tokens = self.text_tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(self.device)
        return self.text_model(tokens).last_hidden_state.detach()

    @staticmethod
    def _gripper_width_to_open_ratio(gripper_state):
        gripper_state = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
        if gripper_state.size == 0:
            return 0.0
        return float(np.clip(np.mean(gripper_state) / 0.04, 0.0, 1.0))

    def _format_state(self, joint_state, gripper_state):
        joints = np.asarray(joint_state, dtype=np.float32).reshape(-1)[:7]
        gripper_open = self._gripper_width_to_open_ratio(gripper_state)
        raw_state = np.concatenate([joints, np.array([gripper_open], dtype=np.float32)])
        state = torch.zeros((1, 1, self.config["model"]["state_token_dim"]), dtype=self.dtype, device=self.device)
        state[:, :, ROBOMME_STATE_INDICES] = torch.as_tensor(raw_state, dtype=self.dtype, device=self.device)
        state_elem_mask = torch.zeros((1, self.config["model"]["state_token_dim"]), dtype=self.dtype, device=self.device)
        state_elem_mask[:, ROBOMME_STATE_INDICES] = 1
        return state, state_elem_mask

    @staticmethod
    def _to_pil(image):
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image
        image = np.asarray(image)
        return Image.fromarray(image.astype(np.uint8))

    def _preprocess_images(self, images):
        background_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean],
            dtype=np.uint8,
        ).reshape(1, 1, 3)
        background_image = (
            np.ones(
                (
                    self.image_processor.size["height"],
                    self.image_processor.size["width"],
                    3,
                ),
                dtype=np.uint8,
            )
            * background_color
        )

        tensors = []
        for image in images:
            image = self._to_pil(image)
            if image is None:
                image = Image.fromarray(background_image)

            if self.config["dataset"].get("image_aspect_ratio", "pad") == "pad":
                width, height = image.size
                if width != height:
                    side = max(width, height)
                    result = Image.new(
                        image.mode,
                        (side, side),
                        tuple(int(x * 255) for x in self.image_processor.image_mean),
                    )
                    result.paste(image, ((side - width) // 2, (side - height) // 2))
                    image = result

            tensors.append(
                self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            )
        return torch.stack(tensors, dim=0).to(self.device, dtype=self.dtype)

    @staticmethod
    def _unformat_actions(action_tensor):
        raw_actions = action_tensor[:, :, ROBOMME_STATE_INDICES].to(torch.float32).cpu().numpy()
        raw_actions[:, :, 7] = np.clip(raw_actions[:, :, 7], 0.0, 1.0) * 2.0 - 1.0
        raw_actions[:, :, :7] = np.nan_to_num(raw_actions[:, :, :7], nan=0.0)
        raw_actions[:, :, 7] = np.where(raw_actions[:, :, 7] >= 0.0, 1.0, -1.0)
        return raw_actions

    @torch.no_grad()
    def step(self, joint_state, gripper_state, images, text_embeds):
        image_tensor = self._preprocess_images(images)
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        states, state_elem_mask = self._format_state(joint_state, gripper_state)
        ctrl_freqs = torch.tensor([self.control_frequency], device=self.device)
        text_embeds = text_embeds.to(self.device, dtype=self.dtype)
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(0)

        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device
            ),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        return self._unformat_actions(trajectory)[0]


def maybe_load_text_embed(cache_dir, task, instruction, policy):
    os.makedirs(cache_dir, exist_ok=True)
    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task)
    instr_hash = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]
    shared_path = os.path.join(cache_dir, f"{instr_hash}.pt")
    path = os.path.join(cache_dir, f"{safe_task}_{instr_hash}.pt")
    if os.path.exists(shared_path):
        return torch.load(shared_path, map_location=policy.device)
    if os.path.exists(path):
        return torch.load(path, map_location=policy.device)
    text_embed = policy.encode_instruction(instruction)
    torch.save(text_embed.cpu(), path)
    return text_embed
