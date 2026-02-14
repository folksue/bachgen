import argparse
from pathlib import Path

import torch

from .generation_utils import generate_tokens, load_model_from_checkpoint
from .midi_utils import score_from_timesteps, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, default="runs/from_scratch/model_epoch_20.pt")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--step-duration", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    model, tokenizer, config = load_model_from_checkpoint(checkpoint_path, args.device)

    max_steps = min(int(args.max_steps), max(1, int(config.max_len) - 1))
    tokens = generate_tokens(
        model,
        tokenizer,
        max_tokens=max_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    timesteps = tokenizer.decode_timesteps(tokens)
    score = score_from_timesteps(timesteps, step_duration=args.step_duration)
    write_outputs(score, out_dir, "sample_000")


if __name__ == "__main__":
    main()
