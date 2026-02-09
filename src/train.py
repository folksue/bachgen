import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_token_sequences, load_chorale_folder, CausalDataset
from .model import ChoraleTransformerLM, ModelConfig
from .representation import ChoraleTokenizer


def _compute_entropy(counts: torch.Tensor) -> float:
    total = counts.sum().item()
    if total == 0:
        return 0.0
    probs = counts[counts > 0].float() / total
    entropy = -(probs * torch.log(probs)).sum().item()
    return entropy / torch.log(torch.tensor(2.0)).item()


def _ngram_repeat_rate(tokens: torch.Tensor, n: int) -> float:
    if tokens.numel() < n:
        return 0.0
    values = tokens.tolist()
    total = 0
    seen = {}
    for i in range(len(values) - n + 1):
        ng = tuple(values[i : i + n])
        seen[ng] = seen.get(ng, 0) + 1
        total += 1
    repeats = sum(c - 1 for c in seen.values() if c > 1)
    return repeats / max(1, total)


def train_epoch(model, loader, optimizer, device, pad_id, vocab_size, max_metric_tokens=100000):
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    metric_tokens = []
    kept = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if kept < max_metric_tokens:
            flat = y.view(-1)
            flat = flat[flat != pad_id].detach().cpu()
            if flat.numel() > 0:
                remaining = max_metric_tokens - kept
                metric_tokens.append(flat[:remaining])
                kept += min(remaining, flat.numel())
    avg_loss = total_loss / max(1, len(loader))
    if not metric_tokens:
        return avg_loss, 0.0, 0.0
    tokens = torch.cat(metric_tokens)
    counts = torch.bincount(tokens, minlength=vocab_size)
    entropy_bits = _compute_entropy(counts)
    repeat_4gram = _ngram_repeat_rate(tokens, 4)
    return avg_loss, entropy_bits, repeat_4gram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ChoraleTokenizer()
    chorales = load_chorale_folder(data_dir)
    sequences = build_token_sequences(chorales, tokenizer)

    train_ds = CausalDataset(sequences, args.seq_len, args.stride, tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = ChoraleTransformerLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, entropy_bits, repeat_4gram = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            tokenizer.pad_id,
            tokenizer.vocab_size,
        )
        torch.save(model.state_dict(), out_dir / "model.pt")
        print(
            f"epoch {epoch} train {train_loss:.4f} "
            f"entropy_bits {entropy_bits:.3f} repeat4 {repeat_4gram:.3f}"
        )

    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({
            "pad_id": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "time_id": tokenizer.time_id,
            "rest_value": tokenizer.rest_value,
            "vocab_size": tokenizer.vocab_size,
        }, f, indent=2)

    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
            "max_len": config.max_len,
        }, f, indent=2)


if __name__ == "__main__":
    main()
