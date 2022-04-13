#!/bin/bash

# rocblas trace
if [[ "$*" == *"rocblas"* ]]; then
    echo "ROCBLAS Trace"
    export ROCBLAS_LAYER=3
    export ROCBLAS_LOG_TRACE_PATH=../../rocblas_log_trace.txt
    export ROCBLAS_LOG_BENCH_PATH=../../rocblas_log_bench.txt
fi

cd build/RelWithDebInfo
./onnxruntime_training_bert --model_name /data/onnx/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm --num_train_steps 100 --train_batch_size 6 --mode perf

cd ..

if [[ "$*" == *"rocblas"* ]]; then
    echo "ROCBLAS Trace"
    export ROCBLAS_LAYER=3
    export ROCBLAS_LOG_TRACE_PATH=../../rocblas_log_trace.txt
    export ROCBLAS_LOG_BENCH_PATH=../../rocblas_log_bench.txt
fi