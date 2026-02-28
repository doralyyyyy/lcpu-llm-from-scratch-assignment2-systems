"""
End-to-end benchmarking script for Transformer forward/backward passes.
Supports: model sizing, warmup, timing with timeit + cuda.synchronize(),
mixed precision (BF16), and memory profiling.
"""
from __future__ import annotations

import argparse
import timeit
from contextlib import nullcontext

import torch

from .model_configs import DEFAULT_BATCH_SIZE, MODEL_SIZES, VOCAB_SIZE


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward.")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(MODEL_SIZES),
        help="Model size from Table 1",
    )
    parser.add_argument("--context-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Time forward only or forward+backward",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use BF16 mixed precision via torch.autocast",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Run memory profiler and dump snapshot to memory_snapshot.pickle",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from cs336_basics.model import BasicsTransformerLM

    cfg = MODEL_SIZES[args.model_size]
    context_length = args.context_length
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    # Random batch: input_ids in [0, vocab_size)
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.mixed_precision and device.type == "cuda"
        else nullcontext()
    )

    if args.mode == "forward_backward":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def run_step():
        if device.type == "cuda":
            torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.view(-1, logits.size(-1)).mean()
        if args.mode == "forward_backward":
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(args.warmup_steps):
        run_step()

    # Optional: start memory recording before measurement
    if args.memory_profile and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Timed runs
    timings = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        run_step()
        elapsed = timeit.default_timer() - start
        timings.append(elapsed)

    if args.memory_profile and device.type == "cuda":
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    import numpy as np
    timings_arr = np.array(timings)
    print(f"model_size={args.model_size} context_length={context_length} mode={args.mode}")
    print(f"mixed_precision={args.mixed_precision} warmup={args.warmup_steps} steps={args.measure_steps}")
    print(f"mean_s={timings_arr.mean():.4f} std_s={timings_arr.std():.4f}")


if __name__ == "__main__":
    main()
