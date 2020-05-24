if [ "$#" -ne 6 ]; then
  echo "Usage: $0 ngpu min_batch_size max_batch_size seq_len max_predictions_per_seq num_train_steps"
  exit 1
fi

ngpu=$1
min_batch_size=$2
max_batch_size=$3
seq_len=$4
max_predictions_per_seq=$5
num_train_steps=$6

grad_acc=1
optimizer=adam
lr=5e-4
warmup_ratio=0.1
warmup_mode=Poly
effective_batch_size=$((ngpu * batch_size * grad_acc))
time_now=$(date +%m%d%H%M)

work_dir=/data/wezhan
#model_dir=${work_dir}/bert/model/bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1_optimized_layer_norm
#model_dir=${work_dir}/bert/model/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm
model_dir=${work_dir}/bert/model/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm
data_dir=${work_dir}/bert/${seq_len}/train
mpi_dir=${OMPI_DIR}
training_bert_dir=${work_dir}/onnxruntime/build/RelWithDebInfo

commit=$(git -C ${work_dir}/onnxruntime rev-parse HEAD | cut -c1-8)
run_name=ort_${commit}_nvbertlarge_fp32_${optimizer}_g${ngpu}_bs${min_batch_size}_acc${grad_acc}_efbs${effective_batch_size}_steps${num_train_steps}_${time_now}_mi50

batch_size=${min_batch_size}
#while [ $batch_size -le ${max_batch_size} ]; do
  nohup ${mpi_dir}/bin/mpirun --allow-run-as-root -n ${ngpu} ${training_bert_dir}/onnxruntime_training_bert --model_name ${model_dir} --train_data_dir ${data_dir} --test_data_dir ${data_dir} --train_batch_size ${batch_size} --mode train --num_train_steps ${num_train_steps} --optimizer ${optimizer} --learning_rate ${lr} --warmup_ratio ${warmup_ratio} --warmup_mode ${warmup_mode} --gradient_accumulation_steps ${grad_acc} --max_seq_length ${seq_len} --max_predictions_per_seq=${max_predictions_per_seq} --data_parallel_size ${ngpu} --use_nccl --lambda 0 --display_loss_steps 1 >> ${run_name}.log 2>&1 &

  tail -f ${run_name}.log
  #batch_size=$((${batch_size}+2))
#done
