#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

echo "Downloading dataset for squad..."

# Download SQuAD

v1="v1.1"
mkdir $v1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $v1/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $v1/dev-v1.1.json
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O $v1/evaluate-v1.1.py

EXP_TRAIN_v1='981b29407e0affa3b1b156f72073b945  -'
EXP_DEV_v1='3e85deb501d4e538b6bc56f786231552  -'
EXP_EVAL_v1='afb04912d18ff20696f7f88eed49bea9  -'
CALC_TRAIN_v1=`cat ${v1}/train-v1.1.json |md5sum`
CALC_DEV_v1=`cat ${v1}/dev-v1.1.json |md5sum`
CALC_EVAL_v1=`cat ${v1}/evaluate-v1.1.py |md5sum`

v2="v2.0"
mkdir $v2
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $v2/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $v2/dev-v2.0.json
wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O $v2/evaluate-v2.0.py

EXP_TRAIN_v2='62108c273c268d70893182d5cf8df740  -'
EXP_DEV_v2='246adae8b7002f8679c027697b0b7cf8  -'
EXP_EVAL_v2='ff23213bed5516ea4a6d9edb6cd7d627  -'

CALC_TRAIN_v2=`cat ${v2}/train-v2.0.json |md5sum`
CALC_DEV_v2=`cat ${v2}/dev-v2.0.json |md5sum`
CALC_EVAL_v2=`cat ${v2}/evaluate-v2.0.py |md5sum`

echo "Squad data download done!"

echo "Verifying Dataset...."

if [ "$EXP_TRAIN_v1" != "$CALC_TRAIN_v1" ]; then
    echo "train-v1.1.json is corrupted! md5sum doesn't match"
fi

if [ "$EXP_DEV_v1" != "$CALC_DEV_v1" ]; then
    echo "dev-v1.1.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_EVAL_v1" != "$CALC_EVAL_v1" ]; then
    echo "evaluate-v1.1.py is corrupted! md5sum doesn't match"
fi


if [ "$EXP_TRAIN_v2" != "$CALC_TRAIN_v2" ]; then
    echo "train-v2.0.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_DEV_v2" != "$CALC_DEV_v2" ]; then
    echo "dev-v2.0.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_EVAL_v2" != "$CALC_EVAL_v2" ]; then
    echo "evaluate-v2.0.py is corrupted! md5sum doesn't match"
fi

echo "Complete!"
