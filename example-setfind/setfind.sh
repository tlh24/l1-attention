#!/bin/bash

# Array of training data sizes
data_sizes=(32 48 64 80 96 128 160 192 224 256 320 384 448 512 640 768)
# data_sizes=(256 320 384 448 512 640 768)

# Function to run the script with specified parameters
run_script() {
	local data_size=$1
	local gpu_id=$2
	local heads=$3
	local layers=$4
	echo "Run $run_id for data size $data_size, heads $heads, layers $layers on GPU $gpu_id"
	python setfind.py -x -m -b $data_size -d $gpu_id --heads $heads --layers $layers
}

for data_size in "${data_sizes[@]}"; do
	echo "Running with data size: $data_size"
	
	run_script $data_size 1 1 1 & # gpu, heads, layers
	run_script $data_size 0 2 1 &
	wait
	run_script $data_size 1 2 2 & # gpu, heads, layers
	run_script $data_size 0 4 4 &
	wait
	run_script $data_size 0 6 6 & 
	run_script $data_size 0 8 8 & 
	wait
	
	echo "Finished train data size: $data_size"
	echo "------------------------"
done



echo "All runs completed."
