"""
Simple benchmark for the LMKAN2DLayer forward pass as shown in the README.

Performs 5 warmup runs and 10 timed runs on CUDA and prints timing stats
along with the benchmark settings.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from lmKAN import LMKAN2DLayer


# Enforce true float32 math (disable TF32 kernels and prefer highest precision)
torch.backends.cuda.matmul.allow_tf32 = False
try:
    torch.set_float32_matmul_precision('highest')  # falls back if unavailable
except Exception:
    pass
torch.backends.cudnn.allow_tf32 = False


# Global settings (mirroring the README example)
NUM_GRIDS: int = 26
BATCH_SIZE: int = 1024
INPUT_DIM: int = 128
OUTPUT_DIM: int = 128
TILE_SIZE_FORWARD: int = 8
TILE_SIZE_BACKWARD: int = 4
BLOCK_SIZE_FORWARD: int = 1024
BLOCK_SIZE_BACKWARD: int = 512  

# Benchmark controls
NUM_WARMUP_RUNS: int = 5
NUM_TIMED_RUNS: int = 10


def _print_settings() -> None:
    """Print the benchmark settings and environment information."""
    print("\nSettings:")
    print(f"  NUM_GRIDS          = {NUM_GRIDS}")
    print(f"  BATCH_SIZE         = {BATCH_SIZE}")
    print(f"  INPUT_DIM          = {INPUT_DIM}")
    print(f"  OUTPUT_DIM         = {OUTPUT_DIM}")
    print(f"  TILE_SIZE_FORWARD  = {TILE_SIZE_FORWARD}")
    print(f"  TILE_SIZE_BACKWARD = {TILE_SIZE_BACKWARD}")
    print(f"  BLOCK_SIZE_FORWARD = {BLOCK_SIZE_FORWARD}")
    print(f"  BLOCK_SIZE_BACKWARD= {BLOCK_SIZE_BACKWARD}")
    print(f"  NUM_WARMUP_RUNS    = {NUM_WARMUP_RUNS}")
    print(f"  NUM_TIMED_RUNS     = {NUM_TIMED_RUNS}")
    if torch.cuda.is_available():
        current_device_index = torch.cuda.current_device()
        print(f"  DEVICE             = cuda:{current_device_index} ({torch.cuda.get_device_name(current_device_index)})")
    else:
        print("  DEVICE             = cpu (CUDA not available)")


def benchmark_forward() -> float | None:
    """Run the forward pass multiple times and return total latency in milliseconds for all timed runs."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark. Please install CUDA-enabled PyTorch and run on a machine with a CUDA GPU.")
        return None

    device = torch.device("cuda")

    # Instantiate the layer as in the README and move to CUDA
    layer = LMKAN2DLayer(
        num_grids=NUM_GRIDS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        tile_size_forward=TILE_SIZE_FORWARD,
        tile_size_backward=TILE_SIZE_BACKWARD,
        block_size_forward=BLOCK_SIZE_FORWARD,
        block_size_backward=BLOCK_SIZE_BACKWARD,
    ).cuda()
    layer.eval()

    # lmKANs use batch-last data layout: input shape [INPUT_DIM, BATCH_SIZE]
    x = torch.randn(INPUT_DIM, BATCH_SIZE, device=device, dtype=torch.float32)

    # Warmup runs (not timed)
    with torch.inference_mode():
        for _ in range(NUM_WARMUP_RUNS):
            _ = layer(x)
    torch.cuda.synchronize()

    # Timed runs: measure total time with a single synchronization after the loop
    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(NUM_TIMED_RUNS):
            _ = layer(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    return total_ms


def benchmark_linear_forward() -> float | None:
    """Run nn.Linear forward multiple times and return total latency in ms for all timed runs."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark. Please install CUDA-enabled PyTorch and run on a machine with a CUDA GPU.")
        return None

    device = torch.device("cuda")

    linear = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).to(device=device, dtype=torch.float32)
    linear.eval()

    # nn.Linear expects [batch_size, INPUT_DIM]
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, dtype=torch.float32)

    # Warmup runs (not timed)
    with torch.inference_mode():
        for _ in range(NUM_WARMUP_RUNS):
            _ = linear(x)
    torch.cuda.synchronize()

    # Timed runs: measure total time with a single synchronization after the loop
    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(NUM_TIMED_RUNS):
            _ = linear(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    return total_ms




def main() -> int:
    lmkan_total_ms = benchmark_forward()
    linear_total_ms = benchmark_linear_forward()
    if (lmkan_total_ms is None or linear_total_ms is None):
        _print_settings()
        return 1

    lmkan_avg_ms = lmkan_total_ms / float(NUM_TIMED_RUNS)
    linear_avg_ms = linear_total_ms / float(NUM_TIMED_RUNS)

    _print_settings()
    
    print(f"\nResults ({NUM_TIMED_RUNS} runs after {NUM_WARMUP_RUNS} warmups):")
    print("  lmKAN:")
    print(f"    Average time: {lmkan_avg_ms:.3f} ms")
    print(f"    Total time:      {lmkan_total_ms:.3f} ms")
    print("  Linear (nn.Linear):")
    print(f"    Average time: {linear_avg_ms:.3f} ms")
    print(f"    Total time:      {linear_total_ms:.3f} ms")

    # Per-parameter statistics (seconds per parameter) using average per-run times
    lmkan_params = (NUM_GRIDS + 1) ** 2 * OUTPUT_DIM * (INPUT_DIM // 2)
    linear_params = INPUT_DIM * OUTPUT_DIM

    lmkan_fwd_spp = (lmkan_avg_ms / 1000.0) / float(lmkan_params * BATCH_SIZE)
    linear_fwd_spp = (linear_avg_ms / 1000.0) / float(linear_params * BATCH_SIZE)

    print("\nPer-parameter timing (seconds per parameter; average per run):")
    print(f"  Parameters: lmKAN={lmkan_params:,}, Linear={linear_params:,}")

    print(f"    lmKAN:  {lmkan_fwd_spp:.3e} s/param")
    print(f"    Linear: {linear_fwd_spp:.3e} s/param")
    print(f"    LMKAN vs Linear time per param speed up: {linear_fwd_spp / lmkan_fwd_spp:.3f}x")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


