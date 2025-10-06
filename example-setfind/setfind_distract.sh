#!/bin/bash

# Array of training data sizes
# data_sizes=(32 48 64 80 96 128 160 192 224 256 320 384 448 512 640 768)
# data_sizes=(32 64 128 256 512 1024 2048 4096)
data_sizes=(4096 2048 1024 512 256 128 64 32)
# data_sizes=(4096)

# Function to run the script with specified parameters
run_script() {
	local data_size=$1
	local gpu_id=$2
	local heads=$3
	local layers=$4
	local npos=$5
	local distract=$6
	echo "Run $run_id for data size $data_size, heads $heads, layers $layers npos $npos distract $distract on GPU $gpu_id"
	python setfind2.py -m -b $data_size -d $gpu_id --heads $heads --layers $layers --npos $npos --distract $distract
}

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

for data_size in "${data_sizes[@]}"; do
	echo "Running with data size: $data_size"

	if [ "$NUM_GPUS" -ge 2 ]; then
# 		run_script $data_size 1 2 1 16 0 & # gpu, heads, layers, npos, distract
# 		run_script $data_size 0 2 1 16 8 &
#
# 		run_script $data_size 1 2 1 16 16 & # run these all in parallel
# 		run_script $data_size 0 2 1 16 24 & # saturate the GPUs
# 		wait
		run_script $data_size 1 2 1 16 32 &
		run_script $data_size 0 2 1 16 48 &

		run_script $data_size 1 2 1 16 96 &
		run_script $data_size 0 2 1 16 128 &
		wait
		run_script $data_size 1 2 1 16 256 &
		run_script $data_size 0 2 1 16 256 &
		wait
	else
# 		run_script $data_size 0 2 1 16 32 &
# 		run_script $data_size 0 2 1 16 48 &
# 		wait
# 		run_script $data_size 0 2 1 16 64 & # gpu, heads, layers, npos, distract
# 		run_script $data_size 0 2 1 16 64 &
# 		wait
# 		run_script $data_size 0 2 1 16 16 & # run these all in parallel
# 		run_script $data_size 0 2 1 16 24 & # saturate the GPUs
# 		wait
# 		run_script $data_size 0 2 1 16 32 &
# 		run_script $data_size 0 2 1 16 48 &
# 		wait
		run_script $data_size 0 2 1 16 96
		run_script $data_size 0 2 1 16 96
	fi
	echo "Finished train data size: $data_size"
	echo "------------------------"
done



echo "All runs completed."
