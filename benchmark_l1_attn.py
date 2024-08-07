import torch
import l1attn_cuda
import time

def benchmark_gpu(batch_size, seq_length, num_heads, head_dim, num_runs=100):
    device = torch.device("cuda")
    
    # Initialize inputs
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    
    # Warm-up
    for _ in range(10):
        _ = l1attn_cuda.L1AttnFn.apply(q, k)
    
    # GPU benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        _ = l1attn_cuda.L1AttnFn.apply(q, k)
    end_event.record()
    
    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    
    return gpu_time

def run_gpu_benchmarks():
    configs = [
        # batch_size, seq_length, num_heads, head_dim
        (1, 128, 8, 64),
        (1, 512, 8, 64),
        (1, 1024, 8, 64),
        (8, 128, 8, 64),
        (8, 512, 8, 64),
        (8, 1024, 8, 64),
        (32, 1024, 8, 64),
    ]
    
    print("Configuration | GPU Time (s) | Time per Run (ms)")
    print("-" * 50)
    
    for batch_size, seq_length, num_heads, head_dim in configs:
        gpu_time = benchmark_gpu(batch_size, seq_length, num_heads, head_dim)
        time_per_run = (gpu_time * 1000) / 100  # Convert to ms
        print(f"{batch_size}x{seq_length}x{num_heads}x{head_dim} | {gpu_time:.4f} | {time_per_run:.2f}")

if __name__ == "__main__":
    run_gpu_benchmarks()