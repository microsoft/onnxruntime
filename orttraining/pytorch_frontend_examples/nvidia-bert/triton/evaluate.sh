#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export TRITON_MODEL_OVERWRITE=True
NV_VISIBLE_DEVICES=0

bert_model=${1:-"large"}
precision=${2:-"fp32"}
init_checkpoint=${3:-"/workspace/bert/checkpoints/bert_qa.pt"}
EXPORT_FORMAT=${4:-"ts-script"}

MODEL_NAME="bert_${bert_model}_${precision}"
BERT_DIR="/workspace/bert"
VOCAB_FILE="/workspace/bert/vocab/vocab"
PREDICT_FILE="/workspace/bert/data/squad/v1.1/dev-v1.1.json"
SQUAD_DIR="/workspace/bert/data/squad/v1.1"
OUT_DIR="/results"
BATCH_SIZE="8"
# Create common bridge for client and server
BRIDGE_NAME="tritonnet"
docker network create ${BRIDGE_NAME}

EXPORT_MODEL_ARGS="${BATCH_SIZE} ${BERT_DIR} ${EXPORT_FORMAT} ${precision} 1 ${MODEL_NAME} 0 1"

# Clean up
cleanup() {
    docker kill trt_server_cont
    docker network rm ${BRIDGE_NAME}
}
trap cleanup EXIT
trap cleanup SIGTERM

./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${EXPORT_MODEL_ARGS} ${TRITON_MODEL_OVERWRITE}

# Start Server
echo Starting server...
SERVER_ID=$( ./triton/launch_triton_server.sh ${BRIDGE_NAME} --NV_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES )
SERVER_IP=$( docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${SERVER_ID} )

./triton/wait_for_triton_server.sh

CMD="python triton/run_squad_client.py \
    --model_name ${MODEL_NAME} \
    --do_lower_case \
    --vocab_file ${VOCAB_FILE} \
    --output_dir ${OUT_DIR} \
    --predict_file ${PREDICT_FILE} \
    --batch_size ${BATCH_SIZE}"

bash scripts/docker/launch.sh "${CMD}"

bash scripts/docker/launch.sh "python ${SQUAD_DIR}/evaluate-v1.1.py ${PREDICT_FILE} ${OUT_DIR}/predictions.json"
