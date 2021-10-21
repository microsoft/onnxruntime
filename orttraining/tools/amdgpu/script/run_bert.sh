if [ "$#" -ne 12 ]; then
  echo "Usage: $0 ngpu batch_size seq_len num_train_steps optimizer model_size training_mode[fp32|fp16] display_loss_steps gradient_accumulation_steps loss_scale gpu_name profile"
  exit 1
fi

ngpu=${1:-1}
batch_size=${2:-64}
seq_len=${3:-128}

if [ ${seq_len} == 128 ]; then
  max_predictions_per_seq=20
elif [ ${seq_len} == 512 ]; then
  max_predictions_per_seq=80
else
  echo "seq_len is not 128 or 512"
  exit 1
fi

num_train_steps=${4:-400}
optimizer=${5:-"adam"}
model_size=${6:-"large"}
training_mode=${7:-"fp32"}
display_loss_steps=${8:-1}
grad_acc=${9:-1}
loss_scale=${10:-1024}
gpu_name=${11:-"mi100"}
profile=${12:-0}

lr=5e-5
warmup_ratio=0.2843
warmup_mode=Poly
effective_batch_size=$((ngpu * batch_size * grad_acc))
time_now=$(date +%m%d%H%M)

HOME_DIR=/workspace
ORT_DIR=${HOME_DIR}/github/onnxruntime
commit=$(git -C ${ORT_DIR} rev-parse HEAD | cut -c1-8)

if [ ${model_size} == "large" ]; then
  model_dir=${HOME_DIR}/model/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12
elif [ ${model_size} == "base" ]; then
  model_dir=${HOME_DIR}/model/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12
elif [ ${model_size} == "tiny" ]; then
  model_dir=${HOME_DIR}/model/bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12
else
  echo "model_size is not large, base or tiny"
  exit 1
fi

data_dir=/data/wezhan/bert/${seq_len}/train
training_bert_dir=${ORT_DIR}/build/RelWithDebInfo

log_dir=${HOME_DIR}/logs/bert_${model_size}/$(date +%m%d)
if [ ! -d ${log_dir} ]; then
  mkdir -p ${log_dir}
fi

run_name=bert_${model_size}_${commit}_g${ngpu}_bs${batch_size}_sl${seq_len}_steps${num_train_steps}_${optimizer}_${training_mode}_acc${grad_acc}_efbs${effective_batch_size}_${time_now}_${gpu_name}

if [ ! -d ${log_dir}/${run_name} ]; then
  mkdir -p ${log_dir}/${run_name}
fi

if [ ${ngpu} != 1 ]; then
  mpi_cmd="${OPENMPI_DIR}/bin/mpirun --allow-run-as-root -n ${ngpu} -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,COLL -x NCCL_MIN_NCHANNELS=4"
fi

if [ ${training_mode} == "fp16" ]; then
  fp16_commands="--use_mixed_precision --allreduce_in_fp16 --loss_scale ${loss_scale}"
fi

if [ ${profile} == 1 ]; then
  if [ ${gpu_name} == "mi100" ]; then
    profile_commands="/opt/rocm/bin/rocprof --obj-tracking on --stats"
  elif [ ${gpu_name} == "v100" ]; then
    profile_commands="nvprof --print-gpu-summary --log-file ${log_dir}/${run_name}-trace.log"
  fi
fi

nohup ${profile_commands} ${mpi_cmd} ${training_bert_dir}/onnxruntime_training_bert --model_name ${model_dir} --train_data_dir ${data_dir} --test_data_dir ${data_dir} --train_batch_size ${batch_size} --mode train --num_train_steps ${num_train_steps} --optimizer ${optimizer} --learning_rate ${lr} --warmup_ratio ${warmup_ratio} --warmup_mode ${warmup_mode} --gradient_accumulation_steps ${grad_acc} --max_seq_length ${seq_len} --max_predictions_per_seq=${max_predictions_per_seq} --use_nccl --lambda 0 ${fp16_commands} --display_loss_steps ${display_loss_steps} --log_dir ${log_dir}/${run_name} > ${log_dir}/${run_name}.log 2>&1 &

tail -f ${log_dir}/${run_name}.log

exit 0
