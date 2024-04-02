
MPI="mpirun --allow-run-as-root
    -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0
    --tag-output --npernode 2 --bind-to numa
    -x MIOPEN_FIND_MODE=1"

CMD="$MPI python mixtral_parity.py"

set -x
$CMD
