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

NV_VISIBLE_DEVICES=${1:-"0"}
DOCKER_BRIDGE=${2:-"host"}
checkpoint=${3:-"/workspace/bert/checkpoints/bert_qa.pt"}
batch_size=${4:-"8"}
BERT_DIR=${5:-"/workspace/bert"}
EXPORT_FORMAT=${6:-"ts-script"}
precision=${7:-"fp16"}
triton_model_version=${8:-1}
triton_model_name=${9:-"bertQA-ts-script"}
triton_dyn_batching_delay=${10:-0}
triton_engine_count=${11:-1}
triton_model_overwrite=${12:-"False"}

PREDICT_FILE="/workspace/bert/data/squad/v1.1/dev-v1.1.json"

DEPLOYER="deployer.py"

CMD="python triton/${DEPLOYER} \
    --${EXPORT_FORMAT} \
    --save-dir /results/triton_models \
    --triton-model-name ${triton_model_name} \
    --triton-model-version ${triton_model_version} \
    --triton-max-batch-size ${batch_size} \
    --triton-dyn-batching-delay ${triton_dyn_batching_delay} \
    --triton-engine-count ${triton_engine_count} "

CMD+="-- --checkpoint ${checkpoint} \
    --config_file ${BERT_DIR}/bert_config.json \
    --vocab_file /workspace/bert/vocab/vocab \
    --predict_file ${PREDICT_FILE} \
    --do_lower_case \
    --batch_size=${batch_size} "

if [[ $precision == "fp16" ]]; then
    CMD+="--fp16 "
fi

bash scripts/docker/launch.sh "${CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
