"""
Distributed Data Parallel: individual parameter all-reduce and bucketed all-reduce.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.utils import _flatten_dense_tensors, _unflatten_dense_tensors
except ImportError:
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPIndividualParameters(nn.Module):
    """
    DDP that all-reduces each parameter's gradient as soon as it is ready (overlaps with backward).
    Broadcasts initial parameters from rank 0; after backward, gradients are averaged via all-reduce.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        # Broadcast parameters from rank 0 to all others so all ranks start with same weights
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Register hook on each parameter with grad: when its grad is ready, all-reduce it (async)
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self):
        use_avg = hasattr(dist.ReduceOp, "AVG")

        def hook(_param: torch.Tensor) -> None:
            if _param.grad is not None:
                op = dist.ReduceOp.AVG if use_avg else dist.ReduceOp.SUM
                h = dist.all_reduce(_param.grad, op=op, async_op=True)
                self._handles.append((h, _param, use_avg))

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for item in self._handles:
            if isinstance(item, tuple):
                h, param, use_avg = item
                h.wait()
                if not use_avg and param.grad is not None:
                    param.grad.data.div_(self._world_size)
            else:
                item.wait()
        self._handles.clear()


class DDPBucketed(nn.Module):
    """
    DDP with gradient bucketing: parameters are grouped into buckets (by reverse order);
    when all grads in a bucket are ready, all-reduce the flattened bucket (async).
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self._handles: list[dist.Work] = []
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._buckets: list[list[torch.nn.Parameter]] = []
        self._bucket_ready_count: list[int] = []
        self._bucket_flattened: list[torch.Tensor | None] = []

        # Broadcast initial parameters from rank 0
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Build buckets in reverse order of model.parameters() (grads become ready in ~that order)
        params_reversed = list(reversed(list(self.module.parameters())))
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        current_bucket: list[torch.nn.Parameter] = []
        current_size = 0

        for p in params_reversed:
            if not p.requires_grad:
                continue
            p_numel = p.numel()
            p_bytes = p_numel * (p.element_size())
            if current_bucket and current_size + p_bytes > bucket_size_bytes:
                self._buckets.append(current_bucket)
                self._bucket_ready_count.append(0)
                self._bucket_flattened.append(None)
                current_bucket = []
                current_size = 0
            current_bucket.append(p)
            current_size += p_bytes

        if current_bucket:
            self._buckets.append(current_bucket)
            self._bucket_ready_count.append(0)
            self._bucket_flattened.append(None)

        # Map param -> (bucket_idx, idx_in_bucket) for hook
        self._param_to_bucket: dict[int, tuple[int, int]] = {}
        for bi, bucket in enumerate(self._buckets):
            for ii, p in enumerate(bucket):
                self._param_to_bucket[id(p)] = (bi, ii)

        self._hook_handles = []
        for bi, bucket in enumerate(self._buckets):
            for p in bucket:
                self._hook_handles.append(p.register_post_accumulate_grad_hook(self._make_bucket_hook(bi)))

    def _make_bucket_hook(self, bucket_idx: int):
        def hook(_param: torch.Tensor) -> None:
            if _param.grad is None:
                return
            self._bucket_ready_count[bucket_idx] += 1
            bucket = self._buckets[bucket_idx]
            if self._bucket_ready_count[bucket_idx] != len(bucket):
                return
            # All grads in this bucket are ready: flatten, all-reduce, unflatten after wait
            grads = [p.grad for p in bucket]
            flat = _flatten_dense_tensors(grads)
            use_avg = hasattr(dist.ReduceOp, "AVG")
            op = dist.ReduceOp.AVG if use_avg else dist.ReduceOp.SUM
            h = dist.all_reduce(flat, op=op, async_op=True)
            self._bucket_flattened[bucket_idx] = flat
            self._handles.append((h, bucket_idx, use_avg))

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for h, bucket_idx, use_avg in self._handles:
            h.wait()
            flat = self._bucket_flattened[bucket_idx]
            bucket = self._buckets[bucket_idx]
            if flat is not None:
                if not use_avg:
                    flat.div_(self._world_size)
                unflat = _unflatten_dense_tensors(flat, bucket)
                for p, g in zip(bucket, unflat):
                    p.grad.copy_(g)
            self._bucket_flattened[bucket_idx] = None
        self._handles.clear()
        for i in range(len(self._bucket_ready_count)):
            self._bucket_ready_count[i] = 0

    def on_train_batch_start(self) -> None:
        """Reset bucket ready counts at start of each training step (grads will be recomputed)."""
        for i in range(len(self._bucket_ready_count)):
            self._bucket_ready_count[i] = 0
        self._handles.clear()
        for i in range(len(self._bucket_flattened)):
            self._bucket_flattened[i] = None
