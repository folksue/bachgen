import argparse
import json
from pathlib import Path
from typing import List

import torch

from .midi_utils import score_from_timesteps, write_outputs
from .model import ChoraleTransformerLM, ModelConfig
from .representation import ChoraleTokenizer


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        cutoff = values[-1]
        logits = torch.where(logits < cutoff, torch.tensor(float("-inf"), device=logits.device), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_tokens(
    model: ChoraleTransformerLM,
    tokenizer: ChoraleTokenizer,
    max_steps: int,
    temperature: float,
    top_k: int,
    device: str,
) -> List[int]:
    model.eval()
    tokens = [tokenizer.bos_id]
    with torch.no_grad():
        for _ in range(max_steps):
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1]
            next_id = sample_next(logits, temperature, top_k)
            tokens.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--step-duration", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    config_path = checkpoint_path.parent / "model_config.json"
    if tokenizer_path.exists():
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tokenizer = ChoraleTokenizer(
            pad_id=tok_cfg["pad_id"],
            bos_id=tok_cfg["bos_id"],
            eos_id=tok_cfg["eos_id"],
            time_id=tok_cfg["time_id"],
            rest_value=tok_cfg["rest_value"],
        )
    else:
        tokenizer = ChoraleTokenizer()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        config = ModelConfig(**cfg)
    else:
        config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = ChoraleTransformerLM(config).to(args.device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    max_steps = min(args.max_steps, max(1, config.max_len - 1))
    tokens = generate_tokens(
        model,
        tokenizer,
        max_steps=max_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    timesteps = tokenizer.decode_timesteps(tokens)
    score = score_from_timesteps(timesteps, step_duration=args.step_duration)
    write_outputs(score, out_dir, "sample_000")


if __name__ == "__main__":
    main()
