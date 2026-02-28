"""
1.2.1 PyTorch attention benchmarking script.
Batch size 8, no multihead (single head). Sweep d_model and seq_len.
Time 100 forward / 100 backward with warmup and torch.cuda.synchronize().
"""
from __future__ import annotations

import itertools
import math
import timeit

import torch

# Reference scaled dot-product attention (no multihead)
def attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Q, K, V: (batch, seq, d). Returns (batch, seq, d)."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def run_attention_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping attention benchmark")
        return

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    n_warmup = 10
    n_forward = 100
    n_backward = 100

    results = []
    for d_model, seq_len in itertools.product(d_models, seq_lengths):
        try:
            Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)

            # Warmup
            for _ in range(n_warmup):
                out = attention_forward(Q, K, V)
                out.backward(out)
                Q.grad = None
                K.grad = None
                V.grad = None
            torch.cuda.synchronize()

            # Time forward
            fwd_times = []
            for _ in range(n_forward):
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                out = attention_forward(Q, K, V)
                torch.cuda.synchronize()
                fwd_times.append(timeit.default_timer() - t0)
            fwd_ms = sum(fwd_times) / len(fwd_times) * 1000

            # Memory before backward (peak after forward)
            mem_before_bwd = torch.cuda.max_memory_allocated() / (1024**2)

            # Time backward
            out = attention_forward(Q, K, V)
            grad_out = torch.randn_like(out, device=device)
            bwd_times = []
            for _ in range(n_backward):
                Q.grad = None
                K.grad = None
                V.grad = None
                out = attention_forward(Q, K, V)
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                out.backward(grad_out)
                torch.cuda.synchronize()
                bwd_times.append(timeit.default_timer() - t0)
            bwd_ms = sum(bwd_times) / len(bwd_times) * 1000

            results.append({
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_ms": round(fwd_ms, 3),
                "backward_ms": round(bwd_ms, 3),
                "mem_before_bwd_MB": round(mem_before_bwd, 2),
            })
            print(f"d_model={d_model} seq_len={seq_len} fwd_ms={fwd_ms:.3f} bwd_ms={bwd_ms:.3f} mem_MB={mem_before_bwd:.2f}")
        except torch.cuda.OutOfMemoryError:
            results.append({"d_model": d_model, "seq_len": seq_len, "error": "OOM"})
            print(f"d_model={d_model} seq_len={seq_len} OOM")

    return results


if __name__ == "__main__":
    run_attention_benchmark()
