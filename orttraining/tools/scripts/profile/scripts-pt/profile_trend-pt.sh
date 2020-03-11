#!/bin/bash
if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  exit 0
fi


if [ -z "${TEST_MODE}" ]
then
  eval_steps=200

  declare -a phase1_fp16_batch_sizes=(2 4 8 16 32 64 80)
  declare -a phase2_fp16_batch_sizes=(1)

  declare -a phase1_fp16_batch_sizes2=(1)
  declare -a phase2_fp16_batch_sizes2=(2 4 8 10 12 14)
else
  eval_steps=15

  declare -a phase1_fp16_batch_sizes=(2)
  declare -a phase2_fp16_batch_sizes=(1)

  declare -a phase1_fp16_batch_sizes2=(1)
  declare -a phase2_fp16_batch_sizes2=(2)
fi



#declare -a phase1_fp32_batch_sizes=(32)
#declare -a phase2_fp32_batch_sizes=(4)

declare -a gpu_nums=(1)
export SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-pt/
export RESULTDIR=/tmp/perf_results_pt
rm $RESULTDIR -rf
mkdir $RESULTDIR
timestamp="pt_"$(date +%s)"_profile_trend"
export SHARED_RES_PATH=$PHILLY_LOG_DIRECTORY/$timestamp
mkdir $SHARED_RES_PATH
echo "RESULT FODLER :"$SHARED_RES_PATH

SINGLECONFIGRUN_SCRIPT_PATH=$SCRIPT_PATH"single_run-pt.sh"

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
        $SINGLECONFIGRUN_SCRIPT_PATH "fp16" $gpu_num \
            $phase1_b $phase2_b $eval_steps 2 1 1 $SHARED_RES_PATH # 2 means we don't really care phase 2 training.
      done
  done

  for phase1_b in "${phase1_fp16_batch_sizes2[@]}"
  do
      for phase2_b in "${phase2_fp16_batch_sizes2[@]}"
      do
        $SINGLECONFIGRUN_SCRIPT_PATH "fp16" $gpu_num \
            $phase1_b $phase2_b 2 $eval_steps 1 1 $SHARED_RES_PATH
      done
  done

  # for phase1_b in "${phase1_fp32_batch_sizes[@]}"
  # do
  #     for phase2_b in "${phase2_fp32_batch_sizes[@]}"
  #     do
  #       $SINGLECONFIGRUN_SCRIPT_PATH "fp32" $gpu_num \
  #           $phase1_b $phase2_b
  #           #$phase1_max_sequence_length $phase2_max_sequence_length \
  #           #$phase1_max_predictions_per_seq $phase2_max_predictions_per_seq
  #     done
  # done
done

cd $RESULTDIR
grep -e "Batch size = " -e "finished pretraining, starting benchmarking" -e "training throughput phase1" -e "finished phase2" -e "training throughput phase2" * | grep "training throughput phase" > throughput
python $SCRIPT_PATH"collect.py" --path throughput

exit 0