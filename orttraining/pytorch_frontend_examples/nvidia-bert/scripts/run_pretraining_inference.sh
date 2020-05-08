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

echo "Container nvidia build = " $NVIDIA_BUILD_ID

DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets
DATA_DIR=${22:-$BERT_PREP_WORKING_DIR/${DATASET}/}

BERT_CONFIG=bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=/results/checkpoints

if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Inference cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be loaded from $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

eval_batch_size=${1:-14}
precision=${2:-"fp16"}
num_gpus=${3:-8}
inference_mode=${4:-"eval"}
model_checkpoint=${5:-"-1"}
inference_steps=${6:-"-1"}
create_logfile=${7:-"true"}
seed=${8:-42}

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi


MODE=""
if [ "$inference_mode" = "eval" ] ; then
   MODE="--eval"
elif [ "$inference_mode" = "prediction" ] ; then
   MODE="--prediction"
else
   echo "Unknown <inference_mode> argument"
   exit -2
fi

echo $DATA_DIR
CMD=" /workspace/bert/run_pretraining_inference.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --ckpt_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$inference_steps"
CMD+=" --ckpt_step=$model_checkpoint"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $MODE"

if [ "$num_gpus" -gt 1 ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
else
   CMD="python3  $CMD"
fi

if [ "$create_logfile" = "true" ] ; then
  export GBS=$((eval_batch_size * num_gpus))
  printf -v TAG "pyt_bert_pretraining_inference_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x