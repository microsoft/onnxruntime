#!/bin/bash

batch=$1
fpkind=$2
gpu_num=$3
accu_steps=$4
max_seq_length=$5
max_predictions_per_seq=$6
eval_steps=$7
result_path=$8

echo "############ _basics/_mpirun_bert.sh seperator starts ###################"
echo "Use "$MPI_TYPE" mpi for multiple gpu runs"
mkdir -p "/tmp/results"
export RESULT_PATH=$result_path
#-display-map -display-allocation -report-bindings
if [ $MPI_TYPE == "philly" ]
then
     # -map-by core MUST be removed in bj1 cluster, but okay with rr3
    $PHILLY_RUNTIME_UTILS/philly-mpirun -- -np $gpu_num --allow-run-as-root -bind-to core \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH  -x OMP_NUM_THREADS -x CUSTOM_PARAM_STRING -x RESULT_PATH \
        bash $ORT_SCRIPT_PATH"_basics/_single_run.sh" ${batch} ${fpkind} ${gpu_num} ${accu_steps} ${max_seq_length} ${max_predictions_per_seq} $eval_steps
else
    # -map-by core MUST be removed in bj1 cluster, but okay with rr3
    # mpirun only works for single machine.
    mpirun -np $gpu_num --allow-run-as-root -bind-to core \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x OMP_NUM_THREADS -x CUSTOM_PARAM_STRING -x RESULT_PATH \
        bash $ORT_SCRIPT_PATH"_basics/_single_run.sh" ${batch} ${fpkind} ${gpu_num} ${accu_steps} ${max_seq_length} ${max_predictions_per_seq} $eval_steps
fi
export RESULT_PATH=""
echo "############ _basics/_mpirun_bert.sh seperator ends ###################"

exit 0
