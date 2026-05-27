from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextTaskDataset, build_vocab, load_local_archive_continual_tasks, toy_continual_tasks
from model import TransformerCMSClassifier, TransformerClassifier


@dataclass
class TaskResult:
    task_name: str
    train_loss: float
    test_acc: float
    seen_task_accs: dict[str, float]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train_one_task(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def build_task_loaders(tasks, vocab, max_len: int, batch_size: int):
    loaders = []
    for task in tasks:
        train_ds = TextTaskDataset(task.train_texts, task.train_labels, vocab, max_len=max_len)
        test_ds = TextTaskDataset(task.test_texts, task.test_labels, vocab, max_len=max_len)
        loaders.append(
            (
                task.task_name,
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(test_ds, batch_size=batch_size),
            )
        )
    return loaders


def save_results(save_dir: Path, results: List[TaskResult], model_name: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_dir / f"{model_name}_results.json"
    csv_path = save_dir / f"{model_name}_results.csv"

    payload = [
        {
            "task_name": r.task_name,
            "train_loss": r.train_loss,
            "test_acc": r.test_acc,
            "seen_task_accs": r.seen_task_accs,
        }
        for r in results
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        row = {"task_name": r.task_name, "train_loss": r.train_loss, "test_acc": r.test_acc}
        for k, v in r.seen_task_accs.items():
            row[f"acc_{k}"] = v
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(save_dir: Path, results: List[TaskResult], model_name: str) -> None:
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    tasks = [r.task_name for r in results]
    accs = [r.test_acc for r in results]
    losses = [r.train_loss for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(tasks, accs, marker="o", label="test_acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis="x", rotation=30)

    ax2 = ax1.twinx()
    ax2.plot(tasks, losses, marker="s", color="orange", label="train_loss")
    ax2.set_ylabel("Loss")

    fig.suptitle(f"{model_name} continual learning results")
    fig.tight_layout()
    fig.savefig(save_dir / f"{model_name}_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cms", action="store_true", help="Use Transformer + CMS instead of baseline Transformer")
    parser.add_argument("--toy", action="store_true", help="Use the toy dataset instead of the local archive")
    parser.add_argument("--archive-dir", type=str, default=r"..\archive", help="Relative path to the local dataset archive")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    tasks = toy_continual_tasks() if args.toy else load_local_archive_continual_tasks(archive_dir=args.archive_dir, num_tasks=args.num_tasks)
    vocab = build_vocab(tasks)
    num_classes = max(max(task.train_labels + task.test_labels) for task in tasks) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCMSClassifier(len(vocab), num_classes, max_len=args.max_len) if args.cms else TransformerClassifier(len(vocab), num_classes, max_len=args.max_len)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    task_loaders = build_task_loaders(tasks, vocab, args.max_len, args.batch_size)
    seen_task_loaders: List[tuple[str, DataLoader]] = []
    results: List[TaskResult] = []

    for task_name, train_loader, test_loader in task_loaders:
        train_loss = 0.0
        for _ in range(args.epochs):
            train_loss = train_one_task(model, train_loader, optimizer, criterion, device)

        seen_task_loaders.append((task_name, test_loader))
        seen_task_accs = {name: evaluate(model, loader, device) for name, loader in seen_task_loaders}
        task_result = TaskResult(
            task_name=task_name,
            train_loss=train_loss,
            test_acc=seen_task_accs[task_name],
            seen_task_accs=seen_task_accs,
        )
        results.append(task_result)

        print(f"Finished {task_name}")
        print(f"  train_loss={train_loss:.4f}")
        print(f"  test_acc={task_result.test_acc:.4f}")
        for prev_name, acc in seen_task_accs.items():
            print(f"  eval[{prev_name}]={acc:.4f}")

    all_accs = [r.test_acc for r in results]
    avg_acc = sum(all_accs) / max(len(all_accs), 1)
    final_seen_accs = results[-1].seen_task_accs if results else {}
    forgetting = 0.0
    if results:
        for r in results[:-1]:
            best_before = max(res.seen_task_accs.get(r.task_name, 0.0) for res in results if r.task_name in res.seen_task_accs)
            final_acc = final_seen_accs.get(r.task_name, 0.0)
            forgetting += best_before - final_acc
        forgetting /= max(len(results) - 1, 1)

    print("\nSummary")
    print(f"  avg_task_acc={avg_acc:.4f}")
    print(f"  avg_forgetting={forgetting:.4f}")

    save_dir = Path(args.save_dir) / ("cms" if args.cms else "baseline")
    save_results(save_dir, results, "cms" if args.cms else "baseline")
    try:
        plot_results(save_dir, results, "cms" if args.cms else "baseline")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
