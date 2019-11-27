#!/bin/bash

batch=$1
fpkind=$2
gpu_num=$3
accu_steps=$4
max_seq_length=$5
max_predictions_per_seq=$6
eval_steps=$7
mpikind=$8

echo "############ mpirun_bert.sh seperator starts ###################"
echo "Use "$mpikind" mpi for multiple gpu runs"
mkdir -p "/tmp/results"

#-display-map -display-allocation -report-bindings
if [ $mpikind == "philly" ]
then
    $PHILLY_RUNTIME_UTILS/philly-mpirun -- -np $gpu_num --allow-run-as-root -bind-to core -map-by core \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x SHARED_RES_PATH -x CUSTOM_PARAMS_STRING -x OMP_NUM_THREADS \
        bash $SCRIPT_PATH"single_run.sh" ${batch} ${fpkind} ${gpu_num} ${accu_steps} ${max_seq_length} ${max_predictions_per_seq} $eval_steps
else
    mpirun -np $gpu_num --allow-run-as-root -bind-to core -map-by core \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x SHARED_RES_PATH -x CUSTOM_PARAMS_STRING -x OMP_NUM_THREADS \
        bash $SCRIPT_PATH"single_run.sh" ${batch} ${fpkind} ${gpu_num} ${accu_steps} ${max_seq_length} ${max_predictions_per_seq} $eval_steps
fi

echo "############ mpirun_bert.sh seperator ends ###################"
exit 0
