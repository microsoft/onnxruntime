#!/bin/bash
accumulation_step=$1
phase1_gpus=$2
phase2_gpus=$3
phase1_run_steps=$4
phase2_run_steps=$5
fp16_phase1_batches=$6
fp16_phase2_batches=$7
fp32_phase1_batches=$8
fp32_phase2_batches=$9
job_name="${10}"

IFS=', ' read -r -a phase1_gpu_nums <<< "$phase1_gpus"
IFS=', ' read -r -a phase2_gpu_nums <<< "$phase2_gpus"

IFS=', ' read -r -a phase1_fp16_batch_sizes <<< "$fp16_phase1_batches"
IFS=', ' read -r -a phase2_fp16_batch_sizes <<< "$fp16_phase2_batches"
IFS=', ' read -r -a phase1_fp32_batch_sizes <<< "$fp32_phase1_batches"
IFS=', ' read -r -a phase2_fp32_batch_sizes <<< "$fp32_phase2_batches"

timestamp="ort_"$(date +%s)"_profile_"$job_name
result_path=$PHILLY_LOG_DIRECTORY/$timestamp
mkdir $result_path

################ Phase 1 ################
max_predictions_per_seq=20
for gpu_num in "${phase1_gpu_nums[@]}"
do
  for b in "${phase1_fp16_batch_sizes[@]}"
  do 
     if [ $accumulation_step -eq 0 ]; then
       updated_accu_step=$((65536 / gpu_num / b ))
     else
       updated_accu_step=$accumulation_step
     fi
     bash $ORT_SCRIPT_PATH"_basics/_mpirun_bert.sh" \
         $b fp16 $gpu_num $updated_accu_step 128 $max_predictions_per_seq \
         $phase1_run_steps $result_path
  done

 for b in "${phase1_fp32_batch_sizes[@]}"
  do  
     if [ $accumulation_step -eq 0 ]; then
       updated_accu_step=$((65536 / gpu_num / b ))
     else
       updated_accu_step=$accumulation_step
     fi
     bash $ORT_SCRIPT_PATH"_basics/_mpirun_bert.sh" \
         $b fp32 $gpu_num $updated_accu_step 128 $max_predictions_per_seq \
         $phase1_run_steps $result_path
  done

done

################ Phase 2 ################
max_predictions_per_seq=80
for gpu_num in "${phase2_gpu_nums[@]}"
do
  for b in "${phase2_fp16_batch_sizes[@]}"
  do  
    if [ $accumulation_step -eq 0 ]; then
      updated_accu_step=$((32768 / gpu_num / b ))
    else
      updated_accu_step=$accumulation_step
    fi

    bash $ORT_SCRIPT_PATH"_basics/_mpirun_bert.sh" \
        $b fp16 $gpu_num $updated_accu_step 512 $max_predictions_per_seq \
        $phase2_run_steps $result_path
  done

  for b in "${phase2_fp32_batch_sizes[@]}"
  do
    if [ $accumulation_step -eq 0 ]; then
      updated_accu_step=$((32768 / gpu_num / b ))
    else
      updated_accu_step=$accumulation_step
    fi

    bash $ORT_SCRIPT_PATH"_basics/_mpirun_bert.sh" \
        $b fp32 $gpu_num $updated_accu_step 512 $max_predictions_per_seq \
        $phase2_run_steps $result_path
  done
done

echo "Aggreate throughput on different workers \n"
AGGREGATE_DIR=/tmp/tmp_results
rm -rf $AGGREGATE_DIR  # remove results for last runs
mkdir $AGGREGATE_DIR
cp $result_path/* $AGGREGATE_DIR -r
cd $AGGREGATE_DIR
grep "Throughput" * > throughput.txt
python $ORT_SCRIPT_PATH"_basics/_collect.py" --path=throughput.txt
exit 0