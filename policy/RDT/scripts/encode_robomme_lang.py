import argparse
import hashlib
import os

import h5py
import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


def instruction_hash(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]


def decode_h5_str(value) -> str:
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def collect_instructions(hdf5_dir: str) -> list[str]:
    instructions = {}
    for filename in sorted(os.listdir(hdf5_dir)):
        if not filename.startswith("record_dataset_") or not filename.endswith(".h5"):
            continue
        file_path = os.path.join(hdf5_dir, filename)
        with h5py.File(file_path, "r") as f:
            episode_keys = sorted(
                (key for key in f.keys() if key.startswith("episode_")),
                key=lambda key: int(key.split("_")[1]),
            )
            for episode_key in episode_keys:
                raw = f[episode_key]["setup"]["task_goal"][0]
                instruction = decode_h5_str(raw)
                instructions[instruction_hash(instruction)] = instruction
    return list(instructions.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", type=str, default="/data/wangyi/robomme_data_h5")
    parser.add_argument("--save_dir", type=str, default="outs/robomme_lang_embeds")
    parser.add_argument("--config_path", type=str, default="configs/robomme.yaml")
    parser.add_argument("--model_path", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--offload_dir", type=str, default="/tmp/rdt_t5_offload")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.offload_dir:
        os.makedirs(args.offload_dir, exist_ok=True)

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    instructions = collect_instructions(args.hdf5_dir)
    print(f"Found {len(instructions)} unique RoboMME instructions.")

    pending = [
        instr for instr in instructions
        if not os.path.exists(os.path.join(args.save_dir, f"{instruction_hash(instr)}.pt"))
    ]
    print(f"{len(pending)} instructions need encoding; {len(instructions) - len(pending)} already exist.")
    if not pending:
        print(f"All RoboMME language embeddings already exist in {args.save_dir}")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    text_embedder = T5Embedder(
        from_pretrained=args.model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    for start in tqdm(range(0, len(pending), args.batch_size)):
        batch = pending[start : start + args.batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attn_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            embeds = text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
            )["last_hidden_state"].detach().cpu()

        attn_mask_cpu = attn_mask.cpu().bool()
        for i, instruction in enumerate(batch):
            text_embed = embeds[i][attn_mask_cpu[i]]
            save_path = os.path.join(args.save_dir, f"{instruction_hash(instruction)}.pt")
            torch.save(text_embed, save_path)

    print(f"Saved RoboMME language embeddings to {args.save_dir}")


if __name__ == "__main__":
    main()
