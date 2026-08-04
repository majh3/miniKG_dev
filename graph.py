

from __future__ import annotations

import torch


class Graph:
    def __init__(self, facts_tensor: torch.Tensor, *, huge_graph: bool | None = None):
                                                                               
                                                                                
        head = facts_tensor[:, 0]
        rel = facts_tensor[:, 1]
        tail = facts_tensor[:, 2]
        if head.dtype not in (torch.int32, torch.int64, torch.long):
            head = head.int()
            rel = rel.int()
            tail = tail.int()
        self.head = head.contiguous() if not head.is_contiguous() else head
        self.rel = rel.contiguous() if not rel.is_contiguous() else rel
        self.tail = tail.contiguous() if not tail.is_contiguous() else tail
        n = int(self.head.shape[0])
        if huge_graph is None:
            huge_graph = n >= 50_000_000
        self.mask = torch.ones(n, dtype=torch.bool, device=self.head.device)
                                                                                  
                                                                                
        if huge_graph:
            self.mask_float = None
        else:
            self.mask_float = torch.ones(n, dtype=torch.float32, device=self.head.device)
        self.e2triple = (self.head, None, self.mask)
        self.triple2e = (None, self.tail, self.mask)
        self.r2triple = (self.rel, None, self.mask)
