"""GITBench input/output adapter around the unmodified RDT components."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.checkpoint_utils import load_rdt_state_dict
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.rdt_runner import RDTRunner


GITBENCH_STATE_INDICES = [
    *(STATE_VEC_IDX_MAPPING[f"left_arm_joint_{index}_pos"] for index in range(7)),
    *(STATE_VEC_IDX_MAPPING[f"right_arm_joint_{index}_pos"] for index in range(7)),
]


class GiTBenchRDTModel:
    """RDT model loaded for inference with precomputed T5 language embeddings."""

    def __init__(self, config, pretrained_model_path, vision_encoder_path, device="cuda", dtype=torch.bfloat16):
        self.config = config
        self.device = device
        self.dtype = dtype
        vision = SiglipVisionTower(vision_tower=vision_encoder_path, args=None)
        self.image_processor = vision.image_processor
        self.vision_model = vision
        image_tokens = (
            config["common"]["img_history_size"]
            * config["common"]["num_cameras"]
            * vision.num_patches
        )
        self.policy = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=image_tokens,
            img_pos_embed_config=[
                ("image", (config["common"]["img_history_size"], config["common"]["num_cameras"], -vision.num_patches)),
            ],
            lang_pos_embed_config=[("lang", -config["dataset"]["tokenizer_max_length"])],
            dtype=dtype,
        )
        load_rdt_state_dict(self.policy, pretrained_model_path, strict=False)
        self.policy.eval().to(device, dtype=dtype)
        self.vision_model.eval().to(device, dtype=dtype)

    def _images(self, images):
        background = np.array([int(value * 255) for value in self.image_processor.image_mean], dtype=np.uint8)
        background = np.broadcast_to(
            background,
            (self.image_processor.size["height"], self.image_processor.size["width"], 3),
        ).copy()
        processed = []
        for image in images:
            array = background if image is None else np.asarray(image, dtype=np.uint8)
            pil_image = Image.fromarray(array)
            if self.config["dataset"].get("image_aspect_ratio", "pad") == "pad" and pil_image.width != pil_image.height:
                side = max(pil_image.width, pil_image.height)
                padded = Image.new(pil_image.mode, (side, side), tuple(background[0, 0]))
                padded.paste(pil_image, ((side - pil_image.width) // 2, (side - pil_image.height) // 2))
                pil_image = padded
            processed.append(self.image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0])
        return torch.stack(processed).to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, state, images, text_embeds, control_frequency=30):
        raw_state = np.asarray(state, dtype=np.float32).reshape(14)
        unified = torch.zeros((1, 1, self.config["model"]["state_token_dim"]), device=self.device, dtype=self.dtype)
        unified[:, :, GITBENCH_STATE_INDICES] = torch.as_tensor(raw_state, device=self.device, dtype=self.dtype)
        mask = torch.zeros((1, self.config["model"]["state_token_dim"]), device=self.device, dtype=self.dtype)
        mask[:, GITBENCH_STATE_INDICES] = 1
        image_embeds = self.vision_model(self._images(images)).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)
        language = text_embeds.to(self.device, dtype=self.dtype)
        if language.ndim == 2:
            language = language.unsqueeze(0)
        actions = self.policy.predict_action(
            lang_tokens=language,
            lang_attn_mask=torch.ones(language.shape[:2], dtype=torch.bool, device=self.device),
            img_tokens=image_embeds,
            state_tokens=unified,
            action_mask=mask.unsqueeze(1),
            ctrl_freqs=torch.tensor([control_frequency], device=self.device),
        )
        return actions[0, :, GITBENCH_STATE_INDICES].to(torch.float32).cpu().numpy()
