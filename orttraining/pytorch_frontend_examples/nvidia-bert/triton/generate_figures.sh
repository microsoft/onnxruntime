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

PROFILING_DATA="triton/profiling_data_int64"

MODEL_NAME="bert_${bert_model}_${precision}_${EXPORT_FORMAT}"
BERT_DIR="/workspace/bert"
# Create common bridge for client and server
BRIDGE_NAME="tritonnet"
docker network create ${BRIDGE_NAME}

# Start Server
echo Starting server...
#bash triton/launch_triton_server.sh
SERVER_ID=$( ./triton/launch_triton_server.sh ${BRIDGE_NAME} --NV_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES )
SERVER_IP=$( docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${SERVER_ID} )


EXPORT_MODEL_ARGS="${BERT_DIR} ${EXPORT_FORMAT} ${precision} 1 ${MODEL_NAME}"
PERF_CLIENT_ARGS="50000 10 20"

# Restart Server
restart_server() {
docker kill trt_server_cont
#bash triton/launch_triton_server.sh
SERVER_ID=$( ./triton/launch_triton_server.sh ${BRIDGE_NAME} --NV_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES )
SERVER_IP=$( docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${SERVER_ID} )
}

# Clean up
cleanup() {
    docker kill trt_server_cont
    docker network rm ${BRIDGE_NAME}
}
trap cleanup EXIT

############## Dynamic Batching Comparison ##############
SERVER_BATCH_SIZE=8
CLIENT_BATCH_SIZE=1
TRITON_ENGINE_COUNT=1
TEST_NAME="DYN_BATCH_"

# Dynamic batching 10 ms

TRITON_DYN_BATCHING_DELAY=10
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_DYN_BATCHING_DELAY} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# Dynamic batching 5 ms
TRITON_DYN_BATCHING_DELAY=5
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_DYN_BATCHING_DELAY} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# Dynamic batching 2 ms
TRITON_DYN_BATCHING_DELAY=2
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_DYN_BATCHING_DELAY} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}


# Static Batching (i.e. Dynamic batching 0 ms)
TRITON_DYN_BATCHING_DELAY=0
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_DYN_BATCHING_DELAY} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

############## Engine Count Comparison ##############
SERVER_BATCH_SIZE=1
CLIENT_BATCH_SIZE=1
TRITON_DYN_BATCHING_DELAY=0
TEST_NAME="ENGINE_C_"

# Engine Count = 4
TRITON_ENGINE_COUNT=4
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_ENGINE_COUNT} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# Engine Count = 2
TRITON_ENGINE_COUNT=2
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_ENGINE_COUNT} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# Engine Count = 1
TRITON_ENGINE_COUNT=1
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS} ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${TRITON_ENGINE_COUNT} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}


############## Batch Size Comparison ##############
# BATCH=1 Generate model and perf
SERVER_BATCH_SIZE=1
CLIENT_BATCH_SIZE=1
TRITON_ENGINE_COUNT=1
TRITON_DYN_BATCHING_DELAY=0
TEST_NAME="BATCH_SIZE_"

./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 64 ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${SERVER_BATCH_SIZE} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# BATCH=2 Generate model and perf
SERVER_BATCH_SIZE=2
CLIENT_BATCH_SIZE=2
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 32 ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${SERVER_BATCH_SIZE} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# BATCH=4 Generate model and perf
SERVER_BATCH_SIZE=4
CLIENT_BATCH_SIZE=4
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 16 ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${SERVER_BATCH_SIZE} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}

# BATCH=8 Generate model and perf
SERVER_BATCH_SIZE=8
CLIENT_BATCH_SIZE=8
./triton/export_model.sh ${NV_VISIBLE_DEVICES} ${BRIDGE_NAME} ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
restart_server
sleep 15
./triton/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 8 ${SERVER_IP} ${BRIDGE_NAME} ${TEST_NAME}${SERVER_BATCH_SIZE} ${PROFILING_DATA} ${NV_VISIBLE_DEVICES}
