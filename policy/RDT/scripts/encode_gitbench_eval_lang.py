"""Precompute RDT language embeddings for public GITBench manifest instructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder
from scripts.encode_lerobot_lang import instruction_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="../../task_config/test_manifest.json")
    parser.add_argument("--save_dir", default="outs/gitbench_lang_embeds")
    parser.add_argument("--config_path", default="configs/gitbench.yaml")
    parser.add_argument("--model_path", default="google/t5-v1_1-xxl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--offload_dir", default="outs/t5_offload_gitbench")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    instructions: dict[str, str] = {}
    wanted_tasks = set(args.tasks) if args.tasks else None
    for task, rows in manifest.get("tasks", {}).items():
        if wanted_tasks is not None and task not in wanted_tasks:
            continue
        for row in rows:
            row_split = str(row.get("split", ""))
            if not (row_split == args.split or (args.split == "test" and row_split.startswith("test-"))):
                continue
            instruction = str(row.get("public", {}).get("instruction", ""))
            if instruction:
                instructions[instruction_hash(instruction)] = instruction
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pending = [instruction for digest, instruction in instructions.items() if not (save_dir / f"{digest}.pt").exists()]
    print(f"Found {len(instructions)} public benchmark instructions; {len(pending)} need encoding.")
    if not pending:
        return
    Path(args.offload_dir).mkdir(parents=True, exist_ok=True)
    with open(args.config_path, "r") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    embedder = T5Embedder(
        from_pretrained=args.model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
    )
    tokenizer, model = embedder.tokenizer, embedder.model
    for start in tqdm(range(0, len(pending), args.batch_size)):
        batch = pending[start : start + args.batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True)
        attention = encoded["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = model(input_ids=encoded["input_ids"].to(device), attention_mask=attention)["last_hidden_state"].detach().cpu()
        for instruction, embedding, mask in zip(batch, embeddings, attention.cpu().bool(), strict=True):
            torch.save(embedding[mask], save_dir / f"{instruction_hash(instruction)}.pt")
    print(f"Saved benchmark language embeddings to {save_dir}")


if __name__ == "__main__":
    main()
