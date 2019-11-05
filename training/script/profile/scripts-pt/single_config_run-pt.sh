#!/bin/bash
#$b fp16 $gpu_num 512 $max_predictions_per_seq

# Remove last time run results, since we are only profiling now.
rm /workspace/bert/results -rf #https://github.com/simpeng/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh#L43

VCNAME="$1"
fp_precision=$2  #"fp16" or "fp32"
gpu_num=$3
phase1_batch_size=$4
phase2_batch_size=$5

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  exit 0
fi

# the following is copied from PT NV repo
train_batch_size=4
learning_rate="6e-3"
precision="fp16"
num_gpus=8
warmup_proportion="0.2843"
train_steps=7038
save_checkpoint_steps=200
resume_training="false"
create_logfile="true"
accumulate_gradients="false"
gradient_accumulation_steps=128
seed=$RANDOM
job_name="bert_lamb_pretraining"
allreduce_post_accumulation="true"
allreduce_post_accumulation_fp16="true"
accumulate_into_fp16="true"

train_batch_size_phase2=4096
learning_rate_phase2="4e-3"
warmup_proportion_phase2="0.128"
train_steps_phase2=1563
gradient_accumulation_steps_phase2=512
# copy ends

data_dir=$PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/py_data/PT_Data/bert_data/seq128
data_dir_phase2=$PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/py_data/PT_Data/bert_data/seq512

precision=$fp_precision
if [ "$precision" == "fp32" ]; then
  accumulate_into_fp16="false"
  allreduce_post_accumulation_fp16="false"
  allreduce_post_accumulation="false"
fi

train_batch_size=$phase1_batch_size
train_batch_size_phase2=$phase2_batch_size
gradient_accumulation_steps=1
gradient_accumulation_steps_phase2=1
train_steps=100
train_steps_phase2=100
num_gpus=$gpu_num

bash scripts/run_pretraining.sh $train_batch_size $learning_rate $precision $num_gpus $warmup_proportion \
    $train_steps $save_checkpoint_steps $resume_training $create_logfile \
    $accumulate_gradients $gradient_accumulation_steps $seed $job_name \
    $allreduce_post_accumulation $allreduce_post_accumulation_fp16 $accumulate_into_fp16 \
    $train_batch_size_phase2 $learning_rate_phase2 $warmup_proportion_phase2 $train_steps_phase2 \
    $gradient_accumulation_steps_phase2 $data_dir $data_dir_phase2 2>&1 | tee $RESULTDIR"/"$fp_precision"_"$gpu_num"_"$phase1_batch_size"_"$phase2_batch_size


exit 0
