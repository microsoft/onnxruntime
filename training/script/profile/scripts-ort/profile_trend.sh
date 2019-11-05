#!/bin/bash
VCNAME="$1"
accumu_steps_type=$2
mpikind=$3  # "philly" or "openmpi"

cp $PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx /code/binary/bert.onnx 

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  exit 0
fi

declare -a phase1_fp16_batch_sizes=(2 4 8 10 12 14 16 18 20 22 24 32 40 48 52 60 64 72 80 88 96 104 112 120 128 136 140 144)

declare -a phase1_fp32_batch_sizes=(2 4 8 16 32 64 80)

declare -a phase1_gpu_nums=(1)

export SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/profile/scripts-ort/
timestamp=$(date +%s)
export SHARED_RES_PATH=$PHILLY_LOG_DIRECTORY/$timestamp
mkdir $SHARED_RES_PATH

# Phase 1 - sequence length 128
max_predictions_per_seq=20
for gpu_num in "${phase1_gpu_nums[@]}"
do
  for b in "${phase1_fp16_batch_sizes[@]}"
  do 
     if [ $accumu_steps_type != "fixed" ]; then
       updated_accu_step=$((65536 / gpu_num / b ))
     else
       updated_accu_step=1
     fi
     $SCRIPT_PATH"mpirun_bert.sh" $b fp16 $gpu_num $updated_accu_step 128 $max_predictions_per_seq $mpikind
  done

#  for b in "${phase1_fp32_batch_sizes[@]}"
#   do  
#      if [ $accumu_steps_type != "fixed" ]; then
#        updated_accu_step=$((65536 / gpu_num / b ))
#      else
#        updated_accu_step=1
#      fi
#      $SCRIPT_PATH"mpirun_bert.sh" $b fp32 $gpu_num $updated_accu_step 128 $max_predictions_per_seq $mpikind
#   done

done


declare -a phase2_fp16_batch_sizes=(2 4 8 10 12 14 16 18 20 22 24)

declare -a phase2_fp32_batch_sizes=(2 4 8 12)

declare -a phase2_gpu_nums=(1)

# Phase 2 - sequence length 512
max_predictions_per_seq=80
for gpu_num in "${phase2_gpu_nums[@]}"
do
  for b in "${phase2_fp16_batch_sizes[@]}"
  do  
     if [ $accumu_steps_type != "fixed" ]; then
       updated_accu_step=$((32768 / gpu_num / b ))
     else
       updated_accu_step=1
     fi
     $SCRIPT_PATH"mpirun_bert.sh" $b fp16 $gpu_num $updated_accu_step 512 $max_predictions_per_seq $mpikind
  done

#  for b in "${phase2_fp32_batch_sizes[@]}"
#   do
#      if [ $accumu_steps_type != "fixed" ]; then
#        updated_accu_step=$((32768 / gpu_num / b ))
#      else
#        updated_accu_step=1
#      fi
#      $SCRIPT_PATH"mpirun_bert.sh" $b fp32 $gpu_num $updated_accu_step 512 $max_predictions_per_seq $mpikind
#   done

done

echo "Aggreate throughput on different workers \n"
AGGREGATE_DIR=/tmp/tmp_results
rm -rf $AGGREGATE_DIR  # remove results for last runs
mkdir $AGGREGATE_DIR
cp $SHARED_RES_PATH/* $AGGREGATE_DIR -r
cd $AGGREGATE_DIR
grep "Throughput" * > throughput.txt
python $SCRIPT_PATH"collect.py" --path=throughput.txt

exit 0
