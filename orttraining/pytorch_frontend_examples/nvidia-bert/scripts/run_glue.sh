#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MRPC_DIR=/workspace/bert/data/glue/MRPC
OUT_DIR=/results/MRPC

mkdir -p $OUT_DIR

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"/workspace/bert/checkpoints/ckpt_8601.pt"}
mode=${2:-"train eval"}
max_steps=${3:-"-1.0"} # if < 0, has no effect
batch_size=${4:-"8"}
learning_rate=${5:-"2e-5"}
precision=${6:-"fp16"}
num_gpu=${7:-8}
epochs=${8:-"3"}
warmup_proportion=${9:-"0.01"}
seed=${10:-2}
vocab_file=${11:-"$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"}
CONFIG_FILE=${12:-"/workspace/bert/bert_config.json"}

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi


CMD="python $mpi_command run_glue.py "
CMD+="--task_name MRPC "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [ "$mode" == "eval" ] ; then
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
if [ "$mode" == "train eval" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--do_lower_case "
CMD+="--data_dir $MRPC_DIR "
CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$CONFIG_FILE "
CMD+="--output_dir $OUT_DIR "
CMD+="$use_fp16"

LOGFILE=$OUT_DIR/logfile

echo $CMD
$CMD |& tee $LOGFILE
