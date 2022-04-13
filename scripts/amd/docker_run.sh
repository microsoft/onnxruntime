alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun_nodevice='sudo docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data -v $HOME/.cache:/root/.cache"

# WORK_DIR='/root/onnxruntime'
WORK_DIR=/workspace/GPT2/transformers

IMAGE_NAME=rocm4.1.pytorch

# stop all containers
docker kill $(docker ps -q)
docker ps

# start new container
CONTAINER_ID=$(drun -d -w $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp scripts/run_lm_gpt2_new.sh $CONTAINER_ID:$WORK_DIR/scripts/run_lm_gpt2_new.sh
docker exec $CONTAINER_ID bash -c "bash scripts/run_lm_gpt2_new.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
