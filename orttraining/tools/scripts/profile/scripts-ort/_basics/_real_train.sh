#!/bin/bash

# Phase 1 + Phase 2 Training

gpu_num=$1
phase1_batch=$2
phase2_batch=$3
phase1_step=$4
phase2_step=$5
phase1_effective_batch=$6
phase2_effective_batch=$7

train_batch_size_phase1=$2
accumulation_steps_phase1=$(( phase1_effective_batch / gpu_num / train_batch_size_phase1 ))
num_train_steps_phase1=$(( phase1_step * accumulation_steps_phase1 ))

train_batch_size_phase2=$3
accumulation_steps_phase2=$(( phase2_effective_batch / gpu_num / train_batch_size_phase2 ))
num_train_steps_phase2=$(( phase2_step * accumulation_steps_phase2 ))

PHASE1_DATADIR="/bert_data/128/books_wiki_en_corpus"
PHASE2_DATADIR="/bert_data/512/books_wiki_en_corpus"

echo "gpu_num: "$gpu_num


declare -a params=( \
      --model_name "/code/bert" \
      --train_data_dir=$PHASE1_DATADIR"/train" \
      --test_data_dir=$PHASE1_DATADIR"/test" \
      --train_batch_size=$train_batch_size_phase1 \
      --gradient_accumulation_steps=$accumulation_steps_phase1 \
      --optimizer=Lamb \
      --learning_rate=3e-3 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=$num_train_steps_phase1 \
      --warmup_ratio=0.2843 \
      --warmup_mode=Poly \
      --train_data_dir_phase2=$PHASE2_DATADIR"/train" \
      --test_data_dir_phase2=$PHASE2_DATADIR"/test" \
      --train_batch_size_phase2=$train_batch_size_phase2 \
      --gradient_accumulation_steps_phase2=$accumulation_steps_phase2 \
      --learning_rate_phase2=2e-3 \
      --num_train_steps_phase2=$num_train_steps_phase2 \
      --warmup_ratio_phase2=0.128 \
      --use_mixed_precision=True \
      --display_loss_steps=10 \
      --allreduce_in_fp16 \
      --log_dir=$PHILLY_JOB_DIRECTORY \
      --mode=train \
 )

full_params=( "${params[@]}" "${CUSTOM_PARAM_STRING}")

echo "running command" $PHILLY_RUNTIME_UTILS/philly-mpirun -- /code/binary/onnxruntime_training_bert ${full_params[@]}
# Be noted: max_seq_length and max_predictions_per_seq are inferred automatically from dataset
# -map-by core will break on bj1 cluster, but works on rr3.
$PHILLY_RUNTIME_UTILS/philly-mpirun -- /code/binary/onnxruntime_training_bert ${full_params[@]}

exit 0