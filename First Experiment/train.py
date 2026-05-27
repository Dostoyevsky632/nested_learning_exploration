from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextTaskDataset, build_vocab, toy_continual_tasks
from model import TransformerCMSClassifier, TransformerClassifier


@dataclass
class ResultRow:
    task_name: str
    accuracy: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return accuracy_score(targets, preds)


def train_one_task(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, criterion):
    model.train()
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def run_experiment(use_cms: bool = False, epochs: int = 10, batch_size: int = 8, lr: float = 3e-4):
    tasks = toy_continual_tasks()
    vocab = build_vocab(tasks)
    num_classes = max(max(t.train_labels) for t in tasks) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        TransformerCMSClassifier(vocab_size=len(vocab), num_classes=num_classes)
        if use_cms
        else TransformerClassifier(vocab_size=len(vocab), num_classes=num_classes)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: List[ResultRow] = []

    for task in tasks:
        train_ds = TextTaskDataset(task.train_texts, task.train_labels, vocab)
        test_ds = TextTaskDataset(task.test_texts, task.test_labels, vocab)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        for _ in range(epochs):
            train_one_task(model, train_loader, device, optimizer, criterion)

        acc = evaluate(model, test_loader, device)
        history.append(ResultRow(task.task_name, acc))
        print(f"[{task.task_name}] acc={acc:.4f}")

    avg_acc = sum(r.accuracy for r in history) / len(history)
    print("\nFinal results")
    print(f"Model: {'Transformer + CMS' if use_cms else 'Transformer'}")
    for row in history:
        print(f"- {row.task_name}: {row.accuracy:.4f}")
    print(f"- Average: {avg_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cms", action="store_true", help="Use Transformer + CMS")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    run_experiment(use_cms=args.cms, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
