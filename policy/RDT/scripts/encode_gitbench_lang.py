"""Precompute T5 embeddings for unique GITBench LeRobot instructions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


def instruction_hash(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot_root", default="/data/wangyi/xgq/GITBenchv4/data/GiTBench_train")
    parser.add_argument("--save_dir", default="outs/gitbench_lang_embeds")
    parser.add_argument("--config_path", default="configs/gitbench.yaml")
    parser.add_argument("--model_path", default="google/t5-v1_1-xxl")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--offload_dir", default="outs/t5_offload_gitbench")
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.offload_dir:
        Path(args.offload_dir).mkdir(parents=True, exist_ok=True)
    table = pq.read_table(Path(args.lerobot_root) / "meta/tasks.parquet")
    instructions = {
        instruction_hash(str(value)): str(value)
        for value in table.column("task").to_pylist()
    }
    pending = [value for key, value in instructions.items() if not (save_dir / f"{key}.pt").exists()]
    print(f"Found {len(instructions)} unique GITBench instructions; {len(pending)} need encoding.")
    if not pending:
        return
    with open(args.config_path, "r") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    embedder = T5Embedder(
        from_pretrained=args.model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir or None,
    )
    tokenizer, model = embedder.tokenizer, embedder.model
    for start in tqdm(range(0, len(pending), args.batch_size)):
        batch = pending[start : start + args.batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        with torch.no_grad():
            embeds = model(input_ids=input_ids, attention_mask=attention)["last_hidden_state"].detach().cpu()
        for item, embed, mask in zip(batch, embeds, attention.cpu().bool(), strict=True):
            torch.save(embed[mask], save_dir / f"{instruction_hash(item)}.pt")
    print(f"Saved GITBench language embeddings to {save_dir}")


if __name__ == "__main__":
    main()
