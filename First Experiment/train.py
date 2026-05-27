from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextTaskDataset, build_vocab, load_20newsgroups_continual_tasks, toy_continual_tasks
from model import TransformerCMSClassifier, TransformerClassifier


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cms", action="store_true", help="Use Transformer + CMS instead of baseline Transformer")
    parser.add_argument("--toy", action="store_true", help="Use the toy dataset instead of 20 Newsgroups")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--num-tasks", type=int, default=4)
    args = parser.parse_args()

    tasks = toy_continual_tasks() if args.toy else load_20newsgroups_continual_tasks(num_tasks=args.num_tasks)
    vocab = build_vocab(tasks)
    num_classes = max(len(task.class_names) for task in tasks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCMSClassifier(len(vocab), num_classes, max_len=args.max_len) if args.cms else TransformerClassifier(len(vocab), num_classes, max_len=args.max_len)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    seen_test_sets: List[tuple[str, DataLoader]] = []
    for task in tasks:
        train_ds = TextTaskDataset(task.train_texts, task.train_labels, vocab, max_len=args.max_len)
        test_ds = TextTaskDataset(task.test_texts, task.test_labels, vocab, max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        for _ in range(args.epochs):
            train_loss = train_one_task(model, train_loader, optimizer, criterion, device)

        seen_test_sets.append((task.task_name, test_loader))
        print(f"Finished {task.task_name}")
        print(f"  train_loss={train_loss:.4f}")

        for prev_name, prev_loader in seen_test_sets:
            acc = evaluate(model, prev_loader, device)
            print(f"  eval[{prev_name}]={acc:.4f}")


if __name__ == "__main__":
    main()
