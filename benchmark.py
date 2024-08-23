import torch
import time
import l1attn_cuda
import csv


def benchmark_l1attn(batch_size, seq_lengths, widths, num_runs=10):
	device = torch.device("cuda")
	results = []
	n_heads=8
	for seq_length in seq_lengths:
		forward_times = []
		backward_times = []
		for width in widths: 
			for _ in range(num_runs):
				q = torch.randn(batch_size, seq_length, n_heads, width, device=device, requires_grad=True)
				k = torch.randn(batch_size, seq_length, n_heads, width, device=device, requires_grad=True)

				# Forward pass
				torch.cuda.synchronize()
				start_time = time.perf_counter()
				attn = l1attn_cuda.L1AttnFn.apply(q, k)
				torch.cuda.synchronize()
				forward_time = time.perf_counter() - start_time
				forward_times.append(forward_time)

				# Backward pass
				grad_output = torch.randn_like(attn)
				torch.cuda.synchronize()
				start_time = time.perf_counter()
				attn.backward(grad_output)
				torch.cuda.synchronize()
				backward_time = time.perf_counter() - start_time
				backward_times.append(backward_time)

			avg_forward_time = sum(forward_times) / num_runs
			avg_backward_time = sum(backward_times) / num_runs
			
			results.append({
					'batch_size': batch_size,
					'seq_length': seq_length,
					'width': width, 
					'avg_forward_time': avg_forward_time,
					'avg_backward_time': avg_backward_time
			})

	return results

def save_results(results, filename):
	with open(filename, 'w', newline='') as csvfile:
		fieldnames = ['batch_size', 'seq_length', 'width', 'avg_forward_time', 'avg_backward_time']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in results:
			writer.writerow(row)

if __name__ == "__main__":
	batch_size = 512
	seq_lengths = [16, 32, 64, 128, 256]
	widths = [16, 32, 33, 64, 65, 96, 97, 128]

	results = benchmark_l1attn(batch_size, seq_lengths, widths)
	filename = f"results_sai.csv"

	save_results(results, filename)
	print(f"Benchmark results saved to {filename}")
