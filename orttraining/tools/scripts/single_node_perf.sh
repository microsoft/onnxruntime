#!/bin/bash

model_name=/bert_ort/bert_models/nv/bert-large/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm
optimizer=lamb
commit=$(git rev-parse HEAD | cut -c1-8)
time_now=$(date +%m%d%H%M)
log_file="single_gpu_perf_${commit}_${time_now}.log"

echo ${log_file}
echo "commit: ${commit}" >> ${log_file}
echo "model: ${model_name}" >> ${log_file}

run_perf(){
    local use_mixed_precision=$1
    local bs_min=$2
    local bs_max=$3
    local max_seq_length=$4
    local max_predictions_per_seq=$5

    for ((bs=$bs_min; bs<=$bs_max; bs++)); do
        config="use_mixed_precision=${use_mixed_precision} max_seq_length=${max_seq_length} max_predictions_per_seq=${max_predictions_per_seq} optimizer=${optimizer} bs=${bs}"
        value=$(./onnxruntime_training_bert --model_name ${model_name} --num_train_steps 100 --train_batch_size ${bs} --optimizer ${optimizer} --max_seq_length ${max_seq_length} --max_predictions_per_seq ${max_predictions_per_seq} --use_mixed_precision=${use_mixed_precision} --mode perf | grep Throughput)
        echo ${config} ${value} >> ${log_file}
    done
}

#fp32
run_perf false 2 5 512 80
run_perf false 28 32 128 20

#fp16
run_perf true 4 10 512 80
run_perf true 56 60 128 20

# View result
more ${log_file}