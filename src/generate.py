import argparse
from pathlib import Path

import torch

from .generation_utils import generate_tokens, load_model_from_checkpoint
from .midi_utils import score_from_timesteps, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, default="runs/from_scratch/model_epoch_49_best.pt")
    parser.add_argument("--out-dir", type=str, required=True)
    # 生成长度控制：
    # - --max-steps: 以 token 为单位（默认）
    # - --max-timesteps: 以 TIME step 为单位（更直观，覆盖 --max-steps）
    parser.add_argument("--max-steps", type=int, default=512, help="Max number of tokens to generate.")
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Max number of TIME steps to generate (overrides --max-steps).",
    )
    parser.add_argument(
        "--min-tokens-before-eos",
        type=int,
        default=0,
        help="Prevent EOS from being sampled before this many tokens have been generated (excluding the initial BOS).",
    )
    parser.add_argument(
        "--no-enforce-structure",
        dest="enforce_structure",
        action="store_false",
        help="Disable structure-constrained sampling (TIME + 4 voice tokens).",
    )
    parser.set_defaults(enforce_structure=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--step-duration", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    model, tokenizer, config = load_model_from_checkpoint(checkpoint_path, args.device)

    # 关键点：虽然模型的 learned positional embedding 只支持长度 config.max_len，
    # 但我们可以用“滑动窗口”继续生成：每一步只把最近的 max_len 个 token 喂给模型。
    # generation_utils.generate_tokens() 已实现该滑动窗口逻辑，因此这里不再把生成长度截断到 max_len。
    max_timesteps = int(args.max_timesteps) if args.max_timesteps is not None else None
    if max_timesteps is not None:
        # 仅用于兜底上限：真实停止由 generation_utils 通过 max_timesteps 控制。
        max_steps = max(int(args.max_steps), max_timesteps * 5 + 8)
    else:
        max_steps = int(args.max_steps)
    tokens = generate_tokens(
        model,
        tokenizer,
        max_tokens=max_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        min_tokens_before_eos=args.min_tokens_before_eos,
        max_timesteps=max_timesteps,
        enforce_structure=bool(args.enforce_structure),
    )
    timesteps = tokenizer.decode_timesteps(tokens)
    score = score_from_timesteps(timesteps, step_duration=args.step_duration)
    write_outputs(score, out_dir, "sample_000")


if __name__ == "__main__":
    main()
