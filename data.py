

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .facts import infer_entity_relation_count
except ImportError:                           
    from facts import infer_entity_relation_count


def load_facts(path: str, data_format: str = "auto", assume_unique: bool = False):
    if data_format == "auto":
        data_format = "npy" if str(path).endswith(".npy") else "txt"
    if data_format == "npy":
        facts = np.load(path, mmap_mode="r")
        if facts.ndim != 2 or facts.shape[1] < 3:
            raise ValueError(f"expected an Nx3 fact array in {path}, got shape={facts.shape}")
        return facts[:, :3]
    if data_format != "txt":
        raise ValueError(f"unknown data format: {data_format}")
    facts = []
    with Path(path).open() as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 3:
                facts.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return facts if assume_unique else sorted(set(facts))


def load_dataset(cfg: SimpleNamespace) -> tuple[np.ndarray, int, int]:
    npy_path = Path(f"data/{cfg.dataset}/all_id.npy")
    if cfg.dataset == "freebase" and not npy_path.exists():
        npy_path = Path("data/freebase/train.npy")
    text_path = Path(f"data/{cfg.dataset}/all_id.txt")
    path = npy_path if npy_path.exists() else text_path
    facts = np.asarray(
        load_facts(str(path), data_format="auto", assume_unique=False),
        dtype=np.int64,
    )
    entity_count, relation_count = infer_entity_relation_count(facts)
    return facts, entity_count, relation_count


def parse_relation_list(text: str, relation_count: int) -> np.ndarray:
    inverse = np.zeros(relation_count, dtype=np.bool_)
    if not text.strip():
        return inverse
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rel = int(item)
        if rel < 0 or rel >= relation_count:
            raise ValueError(f"relation id out of range: {rel}")
        inverse[rel] = True
    return inverse


def orientation_inverse(facts: np.ndarray, relation_count: int, entity_count: int, mode: str) -> np.ndarray:
    if mode == "forward":
        return np.zeros(relation_count, dtype=np.bool_)
    if mode == "reverse_all":
        return np.ones(relation_count, dtype=np.bool_)
    if mode == "custom":
        raise ValueError("custom orientation should be supplied through --inverse-relations")
    if mode != "auto_head_tail":
        raise ValueError(f"unknown orientation mode: {mode}")
    rel = np.asarray(facts[:, 1], dtype=np.int64)
    head_key = rel * int(entity_count) + np.asarray(facts[:, 0], dtype=np.int64)
    tail_key = rel * int(entity_count) + np.asarray(facts[:, 2], dtype=np.int64)
    unique_rel_head = np.unique(head_key)
    unique_rel_tail = np.unique(tail_key)
    head_counts = np.bincount(unique_rel_head // int(entity_count), minlength=relation_count)
    tail_counts = np.bincount(unique_rel_tail // int(entity_count), minlength=relation_count)
    return tail_counts < head_counts


def orientation_list(inverse: np.ndarray) -> list[int]:
    return [idx for idx, flag in enumerate(inverse) if bool(flag)]


                                                                          
                                                                                   
                                                                                       
                                                                                  
                                                                               


class _TailView:
    __slots__ = ("_tails",)

    def __init__(self, tails: np.ndarray):
        self._tails = tails                                                

    def __contains__(self, value) -> bool:
        v = int(value)
        pos = int(np.searchsorted(self._tails, v, side="left"))
        return pos < self._tails.shape[0] and int(self._tails[pos]) == v

    def __len__(self) -> int:
        return int(self._tails.shape[0])

    def __iter__(self):
        return iter(self._tails.tolist())


class TruthLookup:
    __slots__ = ("_hr", "_t", "_rel_base", "_empty")

    def __init__(self, hr_sorted: np.ndarray, t_sorted: np.ndarray, rel_base: int):
        self._hr = hr_sorted
        self._t = t_sorted
        self._rel_base = int(rel_base)
        self._empty = _TailView(t_sorted[:0])

    def get(self, key: tuple[int, int], default=None):
        h, r = key
        k = int(h) * self._rel_base + int(r)
        lo = int(np.searchsorted(self._hr, k, side="left"))
        hi = int(np.searchsorted(self._hr, k, side="right"))
        if lo == hi:
            return self._empty if default is None else default
        return _TailView(self._t[lo:hi])


def build_truth_lookup(facts: np.ndarray, inverse: np.ndarray | None = None) -> TruthLookup:
    h = np.asarray(facts[:, 0], dtype=np.int64)
    r = np.asarray(facts[:, 1], dtype=np.int64)
    t = np.asarray(facts[:, 2], dtype=np.int64)
    if inverse is not None:
        flip = np.asarray(inverse, dtype=bool)[r]
        h, t = np.where(flip, t, h), np.where(flip, h, t)
    rel_base = int(r.max()) + 1 if r.size else 1
    hr = h * rel_base + r
    order = np.lexsort((t, hr))
    return TruthLookup(np.ascontiguousarray(hr[order]), np.ascontiguousarray(t[order]), rel_base)
