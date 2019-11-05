#!/bin/bash
VCNAME="$1"

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  exit 0
fi

declare -a phase1_fp16_batch_sizes=(64)
declare -a phase2_fp16_batch_sizes=(8)

declare -a phase1_fp32_batch_sizes=(32)
declare -a phase2_fp32_batch_sizes=(4)

declare -a gpu_nums=(1 4)
export SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$VCNAME/pengwa/profile/scripts-pt/
export RESULTDIR=/tmp/perf_results_pt
mkdir $RESULTDIR

SINGLECONFIGRUN_SCRIPT_PATH=$SCRIPT_PATH"single_config_run-pt.sh"

phase1_max_sequence_length=128
phase2_max_sequence_length=512

phase1_max_predictions_per_seq=20
phase2_max_predictions_per_seq=80

for gpu_num in "${gpu_nums[@]}"
do
  for phase1_b in "${phase1_fp16_batch_sizes[@]}"
  do
      for phase2_b in "${phase2_fp16_batch_sizes[@]}"
      do
        $SINGLECONFIGRUN_SCRIPT_PATH $VCNAME "fp16" $gpu_num \
            $phase1_b $phase2_b
      done
  done


  for phase1_b in "${phase1_fp32_batch_sizes[@]}"
  do
      for phase2_b in "${phase2_fp32_batch_sizes[@]}"
      do
        $SINGLECONFIGRUN_SCRIPT_PATH $VCNAME "fp32" $gpu_num \
            $phase1_b $phase2_b
      done
  done
done

cd $RESULTDIR
grep -e "Batch size = " -e "finished pretraining, starting benchmarking" -e "training throughput phase1" -e "finished phase2" -e "training throughput phase2" *

exit 0
