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

echo "enter container of index "$PHILLY_CONTAINER_INDEX
echo `hostname`

train_batch_size=64
accumulation_steps=$(( 65536 / gpu_num / train_batch_size ))
num_train_steps=$(( 7038 * accumulation_steps ))
DATADIR=$PHILLY_DATA_DIRECTORY"/"$VCNAME"/pengwa/bert_data/128/books_wiki_en_corpus"

echo "compute complete"$accumulation_steps
# training phase 1 only


#/usr/local/mpi/bin/mpirun --hostfile $PHILLY_SCRATCH_DIRECTORY/mpi-hosts --tag-output \
$PHILLY_RUNTIME_UTILS/philly-mpirun --  \
  -np $gpu_num -bind-to core -map-by core -display-map -display-allocation -report-bindings -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    /code/binary/onnxruntime_training_bert \
      --model_name "/code/binary/bert" \
      --train_data_dir=$DATADIR"/train" \
      --test_data_dir=$DATADIR"/test" \
      --train_batch_size=$train_batch_size \
      --gradient_accumulation_steps=$accumulation_steps \
      --optimizer=Lamb \
      --learning_rate=3e-3 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=$num_train_steps \
      --warmup_ratio=0.2843 \
      --warmup_mode=Poly \
      --use_mixed_precision=True \
      --display_loss_steps=3 \
      --allreduce_in_fp16 \
      --log_dir=$PHILLY_JOB_DIRECTORY   #$PHILLY_LOG_DIRECTORY