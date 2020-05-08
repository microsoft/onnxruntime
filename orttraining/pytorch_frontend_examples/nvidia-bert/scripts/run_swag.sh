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

SWAG_DIR=/workspace/bert/data/swag
OUT_DIR=/results/SWAG

mkdir -p $OUT_DIR

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1}
mode=${2:-"train"}
max_steps=${3:-"-1.0"} # if < 0, has no effect
batch_size=${4:-"12"}
learning_rate=${5:-"5e-6"}
precision=${6:-"fp32"}
num_gpu=${7:-"8"}
epochs=${8:-"2"}

if [ "$mode" != "train" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  mpi_command=""
else
  mpi_command="torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python -m $mpi_command run_swag.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
else
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
CMD+="--do_lower_case "
CMD+="--data_dir $SWAG_DIR/data/ "
CMD+="--bert_model bert-large-uncased "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--output_dir $OUT_DIR "
CMD+="$use_fp16"

LOGFILE=$OUT_DIR/logfile
$CMD |& tee $LOGFILE

sed -r 's/
|(\[A)/\n/g' $LOGFILE > $LOGFILE.edit

throughput=`cat $LOGFILE.edit | grep -E 'Iteration.*[0-9.]+(s/it|it/s)' | tail -1 | egrep -o '[0-9.]+(s/it|it/s)'`

echo "throughput: $throughput"

