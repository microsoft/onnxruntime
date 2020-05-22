#!/bin/bash

ngpu=1
seq_len=512
max_predictions_per_seq=80
batch_size=8
grad_acc=1
num_train_steps=400
optimizer=adam
lr=5e-4
warmup_ratio=0.1
warmup_mode=Linear
effective_batch_size=$((ngpu * batch_size * grad_acc))
commit=$(git rev-parse HEAD | cut -c1-8)
time_now=$(date +%m%d%H%M)
run_name=ort_${commit}_nvbertbase_bookwiki128_fp32_${optimizer}_lr${lr}_${warmup_mode}${warmup_ratio}_g${ngpu}_bs${batch_size}_acc${grad_acc}_efbs${effective_batch_size}_steps${num_train_steps}_${time_now}_mi50

work_dir=/data/wezhan
#model_dir=${work_dir}/bert/model/bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1_optimized_layer_norm
#model_dir=${work_dir}/bert/model/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm
model_dir=${work_dir}/bert/model/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm
data_dir=${work_dir}/bert/${seq_len}/train
mpi_dir=${OMPI_DIR}
training_bert_dir=${work_dir}/onnxruntime/build/RelWithDebInfo

#export LD_LIBRARY_PATH=${mpi_dir}/lib

nohup ${mpi_dir}/bin/mpirun --allow-run-as-root -n ${ngpu} ${training_bert_dir}/onnxruntime_training_bert --model_name ${model_dir} --train_data_dir ${data_dir} --test_data_dir ${data_dir} --train_batch_size ${batch_size} --mode train --num_train_steps ${num_train_steps} --optimizer ${optimizer} --learning_rate ${lr} --warmup_ratio ${warmup_ratio} --warmup_mode ${warmup_mode} --gradient_accumulation_steps ${grad_acc} --max_seq_length ${seq_len} --max_predictions_per_seq=${max_predictions_per_seq} --data_parallel_size ${ngpu} --use_nccl --lambda 0 --display_loss_steps 1  > ${run_name}.log 2>&1 &

tail -f ${run_name}.log

