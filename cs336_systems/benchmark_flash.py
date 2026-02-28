"""
FlashAttention-2 vs PyTorch attention benchmarking using triton.testing.do_bench.
Batch size 1, causal masking. Sweep seq lengths (powers of 2 from 128 to 65536),
embedding dims (powers of 2 from 16 to 128), and bfloat16/float32.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    from triton.testing import do_bench
except ImportError:
    do_bench = None
    triton = None

from .flash_attention import FlashAttention2Triton


def _pytorch_attention_forward(Q, K, V, is_causal=True):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        n_q, n_k = Q.shape[1], K.shape[1]
        mask = torch.arange(n_q, device=Q.device)[:, None] >= torch.arange(n_k, device=Q.device)[None, :]
        S = torch.where(mask, S, -1e6)
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def run_flash_benchmark():
    if not torch.cuda.is_available() or do_bench is None:
        print("CUDA and triton required for flash benchmark")
        return

    batch_size = 1
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_list = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    flash_fn = FlashAttention2Triton.apply

    results = []
    for seq_len in seq_lengths:
        for d in d_list:
            for dtype in dtypes:
                try:
                    Q = torch.randn(batch_size, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)
                    K = torch.randn(batch_size, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)
                    V = torch.randn(batch_size, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)

                    # Flash forward
                    def flash_fwd():
                        o = flash_fn(Q, K, V, True)
                        return o

                    def flash_bwd():
                        o = flash_fn(Q, K, V, True)
                        o.sum().backward()

                    def flash_fwd_bwd():
                        o = flash_fn(Q, K, V, True)
                        o.sum().backward()

                    # PyTorch
                    def pt_fwd():
                        o = _pytorch_attention_forward(Q, K, V, True)
                        return o

                    def pt_bwd():
                        o = _pytorch_attention_forward(Q, K, V, True)
                        o.sum().backward()

                    def pt_fwd_bwd():
                        o = _pytorch_attention_forward(Q, K, V, True)
                        o.sum().backward()

                    fwd_flash = do_bench(flash_fwd, rep=100, warmup=25)
                    bwd_flash = do_bench(flash_bwd, rep=100, warmup=25)
                    fwd_bwd_flash = do_bench(flash_fwd_bwd, rep=100, warmup=25)
                    fwd_pt = do_bench(pt_fwd, rep=100, warmup=25)
                    bwd_pt = do_bench(pt_bwd, rep=100, warmup=25)
                    fwd_bwd_pt = do_bench(pt_fwd_bwd, rep=100, warmup=25)

                    results.append({
                        "seq_len": seq_len, "d": d, "dtype": str(dtype).split(".")[-1],
                        "flash_fwd_ms": fwd_flash, "flash_bwd_ms": bwd_flash, "flash_fwd_bwd_ms": fwd_bwd_flash,
                        "pytorch_fwd_ms": fwd_pt, "pytorch_bwd_ms": bwd_pt, "pytorch_fwd_bwd_ms": fwd_bwd_pt,
                    })
                except Exception as e:
                    results.append({"seq_len": seq_len, "d": d, "dtype": str(dtype), "error": str(e)})

    return results


if __name__ == "__main__":
    run_flash_benchmark()
