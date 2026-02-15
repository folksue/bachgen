import json
from pathlib import Path
from typing import List, Tuple

import torch

from .model import ChoraleTransformerLM, ModelConfig
from .representation import ChoraleTokenizer


def _repo_root() -> Path:
    # src/.. => repo root
    return Path(__file__).resolve().parent.parent


def _artifact_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    # Put metadata outside `runs/**` so it won't be ignored by the default .gitignore.
    # Example: runs/from_scratch/model_epoch_49_best.pt -> artifacts/from_scratch/
    return _repo_root() / "artifacts" / checkpoint_path.parent.name


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Sample one token id from a logits vector with temperature + top-k filtering."""
    logits = logits / max(float(temperature), 1e-6)
    if int(top_k) > 0:
        values, _ = torch.topk(logits, int(top_k))
        cutoff = values[-1]
        logits = torch.where(
            logits < cutoff,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
            logits,
        )
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_tokens(
    model: ChoraleTransformerLM,
    tokenizer: ChoraleTokenizer,
    max_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    min_tokens_before_eos: int = 0,
    max_timesteps: int | None = None,
    enforce_structure: bool = True,
) -> List[int]:
    """Non-streaming token-by-token generation (returns the full token list)."""
    model.eval()
    tokens: List[int] = [int(tokenizer.bos_id)]

    # Learned positional embeddings are limited to config.max_len.
    max_context_len = int(getattr(model, "config", ModelConfig(vocab_size=tokenizer.vocab_size)).max_len)

    # Structure constraints for this representation:
    # BOS, then repeated blocks of: TIME, NOTE(S), NOTE(A), NOTE(T), NOTE(B), then EOS.
    note_min = int(getattr(tokenizer, "note_offset", 4))
    note_max = int(note_min + int(getattr(tokenizer, "rest_value", 128)))
    vocab_size = int(getattr(tokenizer, "vocab_size", note_max + 1))

    def _mask_to_allowed(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed_ids] = logits[allowed_ids]
        return masked

    expecting_time = True
    note_idx = 0
    steps_done = 0

    with torch.no_grad():
        for _ in range(int(max_tokens)):
            if len(tokens) > max_context_len:
                context = tokens[-max_context_len:]
            else:
                context = tokens
            input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1]

            if enforce_structure:
                if expecting_time:
                    allowed = [int(tokenizer.time_id)]
                    # 只有在满足最小长度且已生成至少一个 time step 时才允许 EOS
                    allow_eos = steps_done > 0 and not (
                        int(min_tokens_before_eos) > 0 and len(tokens) < int(min_tokens_before_eos)
                    )
                    if allow_eos:
                        allowed.append(int(tokenizer.eos_id))
                    allowed_ids = torch.tensor(allowed, device=logits.device, dtype=torch.long)
                    logits = _mask_to_allowed(logits, allowed_ids)
                else:
                    # expecting NOTE token
                    allowed_ids = torch.arange(note_min, note_max + 1, device=logits.device, dtype=torch.long)
                    # Guard against any mismatch between tokenizer/model vocab
                    allowed_ids = allowed_ids[allowed_ids < vocab_size]
                    logits = _mask_to_allowed(logits, allowed_ids)
            else:
                # 防止过早结束：在达到最小 token 数之前禁止采样 EOS
                if int(min_tokens_before_eos) > 0 and len(tokens) < int(min_tokens_before_eos):
                    logits = logits.clone()
                    logits[int(tokenizer.eos_id)] = float("-inf")
            next_id = sample_next(logits, temperature, top_k)
            tokens.append(int(next_id))

            if enforce_structure:
                if expecting_time:
                    if next_id == tokenizer.eos_id:
                        break
                    # must be TIME
                    expecting_time = False
                    note_idx = 0
                else:
                    # NOTE
                    note_idx += 1
                    if note_idx >= 4:
                        steps_done += 1
                        expecting_time = True
                        note_idx = 0
                        if max_timesteps is not None and steps_done >= int(max_timesteps):
                            tokens.append(int(tokenizer.eos_id))
                            break
            else:
                if next_id == tokenizer.eos_id:
                    break

    return tokens


def load_tokenizer(checkpoint_path: Path) -> ChoraleTokenizer:
    candidates = [
        checkpoint_path.parent / "tokenizer.json",
        _artifact_dir_for_checkpoint(checkpoint_path) / "tokenizer.json",
    ]
    tokenizer_path = next((p for p in candidates if p.exists()), None)
    if tokenizer_path is not None:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        return ChoraleTokenizer(
            pad_id=tok_cfg["pad_id"],
            bos_id=tok_cfg["bos_id"],
            eos_id=tok_cfg["eos_id"],
            time_id=tok_cfg["time_id"],
            rest_value=tok_cfg["rest_value"],
        )
    return ChoraleTokenizer()


def load_model_config(checkpoint_path: Path, tokenizer: ChoraleTokenizer) -> ModelConfig:
    candidates = [
        checkpoint_path.parent / "model_config.json",
        _artifact_dir_for_checkpoint(checkpoint_path) / "model_config.json",
    ]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        return ModelConfig(vocab_size=tokenizer.vocab_size)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    allowed_keys = set(ModelConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
    if "vocab_size" not in filtered:
        filtered["vocab_size"] = tokenizer.vocab_size
    return ModelConfig(**filtered)


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> Tuple[ChoraleTransformerLM, ChoraleTokenizer, ModelConfig]:
    tokenizer = load_tokenizer(checkpoint_path)
    config = load_model_config(checkpoint_path, tokenizer)
    model = ChoraleTransformerLM(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, tokenizer, config
