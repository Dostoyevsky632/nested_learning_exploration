from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from sklearn.datasets import fetch_20newsgroups
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


def load_20newsgroups_continual_tasks(num_tasks: int = 4, test_size: float = 0.2, random_state: int = 42) -> List[TaskData]:
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = data.data
    labels = data.target
    class_names = list(data.target_names)

    unique_labels = list(range(len(class_names)))
    labels_per_task = len(unique_labels) // num_tasks
    tasks: List[TaskData] = []

    for task_idx in range(num_tasks):
        start = task_idx * labels_per_task
        end = len(unique_labels) if task_idx == num_tasks - 1 else (task_idx + 1) * labels_per_task
        task_labels = set(unique_labels[start:end])

        task_texts = [t for t, y in zip(texts, labels) if y in task_labels]
        task_y = [y for y in labels if y in task_labels]

        remapped = {old: new for new, old in enumerate(sorted(task_labels))}
        task_y = [remapped[y] for y in task_y]
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            task_texts,
            task_y,
            test_size=test_size,
            random_state=random_state,
            stratify=task_y,
        )

        tasks.append(
            TaskData(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                task_name=f"20ng_task_{task_idx + 1}",
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
