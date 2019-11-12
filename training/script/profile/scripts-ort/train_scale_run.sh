#!/bin/bash

VCNAME="$1"
gpu_num=$2
cp $PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx /code/binary/bert.onnx 

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

train_batch_size_phase1=64
accumulation_steps_phase1=$(( 65536 / gpu_num / train_batch_size_phase1 ))
num_train_steps_phase1=$(( 7038 * accumulation_steps_phase1 ))
#num_train_steps_phase1=100 #$(( 10 * accumulation_steps_phase1 ))

train_batch_size_phase2=8
accumulation_steps_phase2=$(( 32768 / gpu_num / train_batch_size_phase2 ))
num_train_steps_phase2=$(( 1563 * accumulation_steps_phase2 ))
#num_train_steps_phase2=100 
#PHASE1_DATADIR=$PHILLY_DATA_DIRECTORY"/"$VCNAME"/pengwa/bert_data/128/books_wiki_en_corpus"
#PHASE2_DATADIR=$PHILLY_DATA_DIRECTORY"/"$VCNAME"/pengwa/bert_data/512/books_wiki_en_corpus"

PHASE1_DATADIR="/bert_data/128/books_wiki_en_corpus"
PHASE2_DATADIR="/bert_data/512/books_wiki_en_corpus"

echo "gpu_num: "$gpu_num

# Be noted: max_seq_length and max_predictions_per_seq are inferred automatically from dataset 
$PHILLY_RUNTIME_UTILS/philly-mpirun --  \
  -np $gpu_num -bind-to core -map-by core -display-map -display-allocation -report-bindings -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    /code/binary/onnxruntime_training_bert \
      --model_name "/code/binary/bert" \
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
      --display_loss_steps=3 \
      --allreduce_in_fp16 \
      --log_dir=$PHILLY_JOB_DIRECTORY

  exit 0