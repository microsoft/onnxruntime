#!/usr/bin/env bash

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

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"/workspace/bert/checkpoints/bert_uncased.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
precision=${5:-"fp16"}
num_gpu=${6:-"8"}
seed=${7:-"1"}
squad_dir=${8:-"$BERT_PREP_WORKING_DIR/download/squad/v1.1"}
vocab_file=${9:-"$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${10:-"/workspace/bert/results/SQuAD"}
mode=${11:-"train eval"}
CONFIG_FILE=${12:-"/workspace/bert/bert_config.json"}
max_steps=${13:-"-1"}

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python  $mpi_command run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
