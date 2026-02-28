"""
Optimizer state sharding: each rank only holds and updates a subset of parameters,
then broadcasts updated params so all ranks stay in sync.
"""
from __future__ import annotations

from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer(optim.Optimizer):
    """
    Wraps an optimizer so that each rank only holds optimizer state for a shard of parameters.
    After each step(), each rank broadcasts its updated parameters to all others.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: Type[optim.Optimizer],
        **kwargs: Any,
    ):
        self._optimizer_cls = optimizer_cls
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        # Flatten to list of all params and build local param list for this rank
        if isinstance(params, (dict, list)) and params and isinstance(next(iter(params), None), dict):
            # Param groups: [{'params': [...], 'lr': ...}, ...]
            self._param_groups_raw = list(params)
            self._all_params: list[torch.nn.Parameter] = []
            for g in self._param_groups_raw:
                self._all_params.extend(g["params"])
        else:
            self._param_groups_raw = [{"params": list(params), **kwargs}]
            self._all_params = list(self._param_groups_raw[0]["params"])

        # This rank owns params at indices: rank, rank + world_size, rank + 2*world_size, ...
        self._local_indices = [i for i in range(len(self._all_params)) if i % self._world_size == self._rank]
        local_params = [self._all_params[i] for i in self._local_indices]

        # Build param_groups for the base Optimizer. Create _local_optimizer first so add_param_group
        # (called by super().__init__) can use it when adding later groups; for the initial group we skip.
        default_kwargs = {k: v for k, v in kwargs.items() if k != "params"}
        local_param_groups = [{"params": local_params, **default_kwargs}]
        self._local_optimizer = optimizer_cls(local_params, **kwargs)
        self._skip_initial_add_param_group = True
        super().__init__(local_param_groups, default_kwargs)

    def __getstate__(self):
        return self._local_optimizer.__getstate__()

    def __setstate__(self, state):
        self._local_optimizer.__setstate__(state)

    def state_dict(self):
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._local_optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        if set_to_none:
            for p in self._all_params:
                if p.grad is not None:
                    p.grad = None
        else:
            for p in self._all_params:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self, closure=None, **kwargs):
        self._local_optimizer.step(closure=closure, **kwargs)
        # Broadcast updated parameters: each rank broadcasts the params it owns
        for i in range(self._world_size):
            for j in range(i, len(self._all_params), self._world_size):
                dist.broadcast(self._all_params[j].data, src=i)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Base Optimizer.__init__ calls add_param_group for each initial group; we already created
        # _local_optimizer with that group, so just delegate to base and return.
        if getattr(self, "_skip_initial_add_param_group", False):
            self._skip_initial_add_param_group = False
            super().add_param_group(param_group)
            return
        new_params = param_group["params"]
        if not isinstance(new_params, list):
            new_params = list(new_params)
        start_idx = len(self._all_params)
        self._all_params.extend(new_params)
        local_new = [new_params[i] for i in range(len(new_params)) if (start_idx + i) % self._world_size == self._rank]
        rest = {k: v for k, v in param_group.items() if k != "params"}
        new_group = {"params": local_new, **rest}
        self._local_optimizer.add_param_group(new_group)
        self._local_indices = [i for i in range(len(self._all_params)) if i % self._world_size == self._rank]
        super().add_param_group(new_group)
