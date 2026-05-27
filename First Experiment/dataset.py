from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import csv
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TaskData:
    train_texts: Sequence[str]
    train_labels: Sequence[int]
    test_texts: Sequence[str]
    test_labels: Sequence[int]
    task_name: str
    class_names: Sequence[str]


class TextTaskDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: dict[str, int], max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.lower().split()
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens[: self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        return self._encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def build_vocab(task_sets: Sequence[TaskData]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for task in task_sets:
        for text in list(task.train_texts) + list(task.test_texts):
            for tok in text.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def _read_archive_files(archive_dir: Path) -> tuple[list[str], list[int], list[str]]:
    list_csv = archive_dir / "list.csv"
    if not list_csv.exists():
        raise FileNotFoundError(f"Could not find {list_csv}")

    class_files = sorted(archive_dir.glob("*.txt"))
    class_files = [p for p in class_files if p.name != "list.csv"]
    class_names = [p.stem for p in class_files]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    texts: list[str] = []
    labels: list[int] = []

    for file_path in class_files:
        class_name = file_path.stem
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Newsgroup:") or line.startswith("document_id:"):
                    continue
                texts.append(line)
                labels.append(class_to_idx[class_name])

    return texts, labels, class_names


def load_local_archive_continual_tasks(
    archive_dir: str | Path = Path("..") / "archive",
    num_tasks: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> List[TaskData]:
    archive_dir = Path(archive_dir)
    texts, labels, class_names = _read_archive_files(archive_dir)

    unique_labels = list(range(len(class_names)))
    labels_per_task = max(len(unique_labels) // num_tasks, 1)
    tasks: List[TaskData] = []

    for task_idx in range(num_tasks):
        start = task_idx * labels_per_task
        end = len(unique_labels) if task_idx == num_tasks - 1 else min((task_idx + 1) * labels_per_task, len(unique_labels))
        task_labels = set(unique_labels[start:end])
        if not task_labels:
            continue

        task_pairs = [(t, y) for t, y in zip(texts, labels) if y in task_labels]
        if len(task_pairs) < 4:
            continue

        task_texts = [t for t, _ in task_pairs]
        task_y = [y for _, y in task_pairs]

        stratify = task_y if len(set(task_y)) > 1 else None
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            task_texts,
            task_y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        tasks.append(
            TaskData(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                task_name=f"archive_task_{task_idx + 1}",
                class_names=[class_names[i] for i in sorted(task_labels)],
            )
        )

    return tasks


def toy_continual_tasks() -> List[TaskData]:
    return [
        TaskData(
            task_name="task_1_banking",
            train_texts=[
                "reset my card pin",
                "how do i check my balance",
                "transfer money to savings",
                "lost my debit card",
            ],
            train_labels=[0, 1, 2, 3],
            test_texts=[
                "i forgot my pin",
                "show me account balance",
                "move cash to savings",
                "card was stolen",
            ],
            test_labels=[0, 1, 2, 3],
            class_names=["reset_pin", "balance", "transfer", "lost_card"],
        ),
        TaskData(
            task_name="task_2_travel",
            train_texts=[
                "book a flight to seattle",
                "change my hotel reservation",
                "cancel my trip",
                "find cheap tickets",
            ],
            train_labels=[0, 1, 2, 3],
            test_texts=[
                "reserve flight to boston",
                "modify hotel booking",
                "i want to cancel travel",
                "search for low cost flights",
            ],
            test_labels=[0, 1, 2, 3],
            class_names=["book_flight", "change_hotel", "cancel_trip", "cheap_tickets"],
        ),
    ]
