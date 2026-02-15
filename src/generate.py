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

    max_context_len = int(config.max_len)
    max_timesteps = int(args.max_timesteps) if args.max_timesteps is not None else None
    
    # 分段生成逻辑：每段最多生成 (max_context_len - 1) 个新 token，
    # 末尾保留 10 个 token 作为下一段的 context overlap（2个 block）
    overlap_tokens = 10
    tokens_per_segment = max_context_len - 1 - overlap_tokens
    
    if max_timesteps is not None:
        target_timesteps = int(max_timesteps)
        # 每个 timestep = 1 TIME + 4 NOTE = 5 token
        # 所以需要约 target_timesteps * 5 个 token（加上 BOS 和可能的 EOS）
        target_tokens = target_timesteps * 5 + 2
    else:
        target_tokens = int(args.max_steps)
    
    # 第一段：从 BOS 开始生成
    all_tokens = generate_tokens(
        model,
        tokenizer,
        max_tokens=tokens_per_segment,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        min_tokens_before_eos=args.min_tokens_before_eos,
        max_timesteps=None,
        enforce_structure=bool(args.enforce_structure),
    )
    
    # 分段生成后续段落
    while len(all_tokens) < target_tokens:
        # 检查是否已经以 EOS 结尾（生成完毕）
        if all_tokens and all_tokens[-1] == tokenizer.eos_id:
            break
        
        # 提取末尾 overlap 作为下一段的 initial_tokens
        # 保留最后 overlap_tokens 个 token + BOS
        if len(all_tokens) > overlap_tokens:
            context_for_next = [tokenizer.bos_id] + all_tokens[-(overlap_tokens):]
        else:
            context_for_next = all_tokens
        
        # 计算这一段还要生成多少 token
        remaining_tokens = target_tokens - len(all_tokens)
        max_new_tokens = min(tokens_per_segment, remaining_tokens)
        
        # 检查目标 timesteps 是否已满足
        if max_timesteps is not None:
            current_timesteps = sum(1 for token in all_tokens if token == tokenizer.time_id)
            remaining_timesteps = target_timesteps - current_timesteps
            if remaining_timesteps <= 0:
                all_tokens.append(tokenizer.eos_id)
                break
        
        # 生成下一段
        segment = generate_tokens(
            model,
            tokenizer,
            max_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            min_tokens_before_eos=args.min_tokens_before_eos,
            max_timesteps=None,
            enforce_structure=bool(args.enforce_structure),
            initial_tokens=context_for_next,
        )
        
        # 拼接：去掉 initial_tokens 部分，只保留新生成的部分
        # segment 从 context_for_next 开始，所以新内容从索引 len(context_for_next) 开始
        new_content = segment[len(context_for_next):]
        all_tokens.extend(new_content)
        
        # 如果这一段以 EOS 结尾，停止生成
        if new_content and new_content[-1] == tokenizer.eos_id:
            break
    
    tokens = all_tokens
    timesteps = tokenizer.decode_timesteps(tokens)
    score = score_from_timesteps(timesteps, step_duration=args.step_duration)
    write_outputs(score, out_dir, "sample_000")


if __name__ == "__main__":
    main()
