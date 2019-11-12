#!/bin/bash
accumu_steps_type=$1
mpikind=$2  # "philly" or "openmpi"

cp $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx /code/binary/bert.onnx 

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi



if [ -z "${TEST_MODE}" ]
then
  eval_steps=200

  declare -a phase1_fp16_batch_sizes=(64 144)
  declare -a phase1_fp32_batch_sizes=(32 80)
  declare -a phase2_fp16_batch_sizes=(8 24)
  declare -a phase2_fp32_batch_sizes=(4 12)
else
  eval_steps=15

  declare -a phase1_fp16_batch_sizes=(64)
  declare -a phase1_fp32_batch_sizes=(32)
  declare -a phase2_fp16_batch_sizes=(8)
  declare -a phase2_fp32_batch_sizes=(4)
fi

export SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/
timestamp="ort_"$(date +%s)"_profile_nv_compare_github"
export SHARED_RES_PATH=$PHILLY_LOG_DIRECTORY/$timestamp
mkdir $SHARED_RES_PATH

#########################################
################ Phase 1 ################

if [ -z "${TEST_MODE}" ]; then
  declare -a phase1_gpu_nums=(1 4 8 16)
else
  declare -a phase1_gpu_nums=(4)
fi

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
     bash $SCRIPT_PATH"mpirun_bert.sh" $b fp16 $gpu_num $updated_accu_step 128 $max_predictions_per_seq $eval_steps $mpikind
  done

 for b in "${phase1_fp32_batch_sizes[@]}"
  do  
     if [ $accumu_steps_type != "fixed" ]; then
       updated_accu_step=$((65536 / gpu_num / b ))
     else
       updated_accu_step=1
     fi
     bash $SCRIPT_PATH"mpirun_bert.sh" $b fp32 $gpu_num $updated_accu_step 128 $max_predictions_per_seq $eval_steps $mpikind
  done

done

#########################################
################ Phase 2 ################

if [ -z "${TEST_MODE}" ]; then
  declare -a phase2_gpu_nums=(1 4 8 16)
else
  declare -a phase2_gpu_nums=(4)
fi

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
     bash $SCRIPT_PATH"mpirun_bert.sh" $b fp16 $gpu_num $updated_accu_step 512 $max_predictions_per_seq $eval_steps $mpikind
  done

 for b in "${phase2_fp32_batch_sizes[@]}"
  do
     if [ $accumu_steps_type != "fixed" ]; then
       updated_accu_step=$((32768 / gpu_num / b ))
     else
       updated_accu_step=1
     fi
     bash $SCRIPT_PATH"mpirun_bert.sh" $b fp32 $gpu_num $updated_accu_step 512 $max_predictions_per_seq $eval_steps $mpikind
  done

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
