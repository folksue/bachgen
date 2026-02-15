import argparse
import json
from pathlib import Path
import sys
import re
import torch
from torch import nn
from torch.utils.data import DataLoader

# Allow running as a script (python src/train.py) by falling back to absolute imports
try:
    from .dataset import build_token_sequences, load_chorale_folder, CausalDataset
    from .model import ChoraleTransformerLM, ModelConfig
    from .representation import ChoraleTokenizer
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset import build_token_sequences, load_chorale_folder, CausalDataset
    from src.model import ChoraleTransformerLM, ModelConfig
    from src.representation import ChoraleTokenizer


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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume from. Supports either a plain model state_dict (.pt) "
            "or a full checkpoint dict saved by this script (checkpoint_last.pt)."
        ),
    )
    parser.add_argument(
        "--no-resume-optimizer",
        dest="resume_optimizer",
        action="store_false",
        help="When resuming from a full checkpoint, do NOT restore optimizer state.",
    )
    parser.set_defaults(resume_optimizer=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    # 默认把 test 并入训练集（保留 valid 做验证）。如需关闭，用 --no-include-test。
    parser.add_argument(
        "--include-test",
        dest="include_test",
        action="store_true",
        help="Include data/test in the training set while still using data/valid for validation. (default: enabled)",
    )
    parser.add_argument(
        "--no-include-test",
        dest="include_test",
        action="store_false",
        help="Do NOT include data/test in training.",
    )
    parser.set_defaults(include_test=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ChoraleTokenizer()
    # 加载训练集和验证集
    train_chorales = load_chorale_folder(data_dir / "train")
    if args.include_test and (data_dir / "test").exists():
        test_chorales = load_chorale_folder(data_dir / "test")
        train_chorales = list(train_chorales) + list(test_chorales)
    valid_chorales = load_chorale_folder(data_dir / "valid")
    train_sequences = build_token_sequences(train_chorales, tokenizer)
    valid_sequences = build_token_sequences(valid_chorales, tokenizer)

    train_ds = CausalDataset(train_sequences, args.seq_len, args.stride, tokenizer.pad_id)
    valid_ds = CausalDataset(valid_sequences, args.seq_len, args.stride, tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = ChoraleTransformerLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_epoch = 0
    last_ckpt_path = out_dir / "checkpoint_last.pt"
    best_path_current: Path | None = None

    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=args.device)
        # Full checkpoint dict (preferred for true resume)
        if isinstance(payload, dict) and ("model" in payload or "model_state" in payload):
            state = payload.get("model", payload.get("model_state"))
            if state is None:
                raise ValueError(f"Invalid checkpoint (missing model state): {resume_path}")
            model.load_state_dict(state)
            if args.resume_optimizer and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "best_val_loss" in payload:
                best_val_loss = float(payload["best_val_loss"])
            if "best_epoch" in payload:
                best_epoch = int(payload["best_epoch"])
            if "best_path" in payload and payload["best_path"]:
                try:
                    best_path_current = Path(str(payload["best_path"]))
                except Exception:
                    best_path_current = None
            if "epoch" in payload:
                start_epoch = int(payload["epoch"]) + 1
            print(
                f"resumed from {resume_path} (start_epoch={start_epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
            )
        else:
            # Plain model state_dict
            model.load_state_dict(payload)
            # 尝试从文件名推断已训练到的 epoch：model_epoch_{N}.pt / model_epoch_{N}_best.pt
            m = re.search(r"model_epoch_(\d+)", resume_path.name)
            if m is not None:
                inferred = int(m.group(1))
                start_epoch = inferred + 1
            # 如果你是从 *_best.pt 恢复，记录当前 best 文件路径（后续若出现更优会替换并删除旧 best）
            if resume_path.name.endswith("_best.pt"):
                best_path_current = resume_path
            print(
                f"loaded model weights from {resume_path} (optimizer not restored, start_epoch={start_epoch})"
            )

    def evaluate(model, loader, device, pad_id, vocab_size):
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
        avg_loss = total_loss / max(1, len(loader))
        return avg_loss

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, entropy_bits, repeat_4gram = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            tokenizer.pad_id,
            tokenizer.vocab_size,
        )
        val_loss = evaluate(
            model,
            valid_loader,
            args.device,
            tokenizer.pad_id,
            tokenizer.vocab_size,
        )

        # 每个 epoch 都保存一次普通权重（便于回溯/手动挑选）。
        torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            new_best_path = out_dir / f"model_epoch_{epoch}_best.pt"
            torch.save(model.state_dict(), new_best_path)
            # 保证目录里始终只有一个 best：有新的 best 时删除旧 best。
            if best_path_current is not None and best_path_current != new_best_path and best_path_current.exists():
                try:
                    best_path_current.unlink()
                except OSError:
                    pass
            best_path_current = new_best_path

        # 保存可恢复训练的 last checkpoint（不带 best 后缀）
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_path": str(best_path_current) if best_path_current is not None else "",
                "config": {
                    "vocab_size": config.vocab_size,
                    "d_model": config.d_model,
                    "n_head": config.n_head,
                    "n_layer": config.n_layer,
                    "dropout": config.dropout,
                    "max_len": config.max_len,
                },
            },
            last_ckpt_path,
        )
        print(
            f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
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

    # 同时写一份到仓库根目录 artifacts/<run_name>/，避免被 runs/** 的 .gitignore 忽略
    repo_root = Path(__file__).resolve().parent.parent
    artifact_dir = repo_root / "artifacts" / out_dir.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "tokenizer.json", "w", encoding="utf-8") as f:
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
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, f, indent=2)

    with open(artifact_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
            "max_len": config.max_len,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, f, indent=2)


if __name__ == "__main__":
    main()
