import json
from pathlib import Path
from typing import List, Tuple

import torch

from .model import ChoraleTransformerLM, ModelConfig
from .representation import ChoraleTokenizer


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
) -> List[int]:
    """Non-streaming token-by-token generation (returns the full token list)."""
    model.eval()
    tokens: List[int] = [int(tokenizer.bos_id)]

    # Learned positional embeddings are limited to config.max_len.
    max_context_len = int(getattr(model, "config", ModelConfig(vocab_size=tokenizer.vocab_size)).max_len)

    with torch.no_grad():
        for _ in range(int(max_tokens)):
            if len(tokens) > max_context_len:
                context = tokens[-max_context_len:]
            else:
                context = tokens
            input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1]
            next_id = sample_next(logits, temperature, top_k)
            tokens.append(int(next_id))
            if next_id == tokenizer.eos_id:
                break

    return tokens


def load_tokenizer(checkpoint_path: Path) -> ChoraleTokenizer:
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.exists():
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
    config_path = checkpoint_path.parent / "model_config.json"
    if not config_path.exists():
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
