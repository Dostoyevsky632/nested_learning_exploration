from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TaskData:
    train_texts: Sequence[str]
    train_labels: Sequence[int]
    test_texts: Sequence[str]
    test_labels: Sequence[int]
    task_name: str


class TextTaskDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: dict[str, int], max_len: int = 64):
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
        ),
    ]
