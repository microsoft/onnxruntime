echo "Benchmark Scaled Dot Product Attention (SDPA) performance on GPU:"

set CUDA_VISIBLE_DEVICES=0
python benchmark_mha.py --use_gpu
python benchmark_mha.py --use_gpu --use_cuda_graph
python benchmark_mha.py --use_gpu --torch

type benchmark_mha_gpu_*.csv > mha_gpu_benchmark_results.csv

echo "Benchmark performance on CPU with number of threads:"
set MKL_DYNAMIC=FALSE
set OMP_NUM_THREADS=1
python benchmark_mha.py --torch

set OMP_NUM_THREADS=2
python benchmark_mha.py --torch

set OMP_NUM_THREADS=4
python benchmark_mha.py --torch

set OMP_NUM_THREADS=8
python benchmark_mha.py --torch

set MKL_DYNAMIC=
set OMP_NUM_THREADS=

set ORT_DISABLE_FLASH_ATTENTION=0
python benchmark_mha.py --intra_op_num_threads 1
python benchmark_mha.py --intra_op_num_threads 2
python benchmark_mha.py --intra_op_num_threads 4
python benchmark_mha.py --intra_op_num_threads 8

echo "Benchmark performance on CPU with default threads settings:"
python benchmark_mha.py

python benchmark_mha.py --torch

python benchmark_mha.py --causal
python benchmark_mha.py --torch --causal

python benchmark_mha.py --causal --has_past

set ORT_DISABLE_FLASH_ATTENTION=1
python benchmark_mha.py
set ORT_DISABLE_FLASH_ATTENTION=

type benchmark_mha_cpu_*.csv > mha_cpu_benchmark_results.csv
