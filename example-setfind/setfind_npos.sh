#!/bin/bash

# Array of training data sizes
# data_sizes=(32 48 64 80 96 128 160 192 224 256 320 384 448 512 640 768)
data_sizes=(32 64 128 256 512 1024 2048 4096)
# data_sizes=(4096)

# Function to run the script with specified parameters
run_script() {
	local data_size=$1
	local gpu_id=$2
	local heads=$3
	local layers=$4
	local npos=$5
	echo "Run $run_id for data size $data_size, heads $heads, layers $layers npos $npos on GPU $gpu_id"
	python setfind2.py -x -m -b $data_size -d $gpu_id --heads $heads --layers $layers --npos $npos
}

for data_size in "${data_sizes[@]}"; do
	echo "Running with data size: $data_size"
	
	run_script $data_size 1 2 1 8 & # gpu, heads, layers, npos
	run_script $data_size 0 2 1 16 &
	wait
	run_script $data_size 1 2 1 32 & # run these all in parallel
	run_script $data_size 0 2 1 64 & # saturate the GPUs
	wait
	run_script $data_size 1 2 1 128 &
	run_script $data_size 0 2 1 256 &
	wait
	
	echo "Finished train data size: $data_size"
	echo "------------------------"
done



echo "All runs completed."
