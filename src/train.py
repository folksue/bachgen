import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_token_sequences, load_chorale_folder, CausalDataset
from .model import ChoraleTransformerLM, ModelConfig
from .representation import ChoraleTokenizer


def split_sequences(sequences, train_ratio: float) -> Tuple[list, list]:
    cut = int(len(sequences) * train_ratio)
    return sequences[:cut], sequences[cut:]


def train_epoch(model, loader, optimizer, device, pad_id):
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
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
    return total_loss / max(1, len(loader))


def eval_epoch(model, loader, device, pad_id):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


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
    train_seqs, val_seqs = split_sequences(sequences, 0.9)

    train_ds = CausalDataset(train_seqs, args.seq_len, args.stride, tokenizer.pad_id)
    val_ds = CausalDataset(val_seqs, args.seq_len, args.stride, tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = ChoraleTransformerLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, tokenizer.pad_id)
        val_loss = eval_epoch(model, val_loader, args.device, tokenizer.pad_id)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}")

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
