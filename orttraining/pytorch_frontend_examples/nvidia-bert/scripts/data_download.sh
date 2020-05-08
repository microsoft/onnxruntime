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

DATA_DIR=${1:-/workspace/bert/data}

# Download vocab files from pretrained model
cd vocab && python3 download_models.py && rm *.zip && rm ./*/*.ckpt.*

# Download SQUAD
cd $DATA_DIR/squad && . squad_download.sh

# Download SWAG
git clone https://github.com/rowanz/swagaf.git $DATA_DIR/swag

# Download GLUE
cd $DATA_DIR/glue && . download_mrpc.sh

# WIKI Download
cd $DATA_DIR/wikipedia_corpus && . download_wikipedia.sh

# Bookcorpus  Download
cd $DATA_DIR/bookcorpus && . download_bookcorpus.sh

cd $DATA_DIR
# Create HDF5 files for WIKI
bash create_datasets_from_start.sh wikipedia_corpus ./wikipedia_corpus/wikipedia_corpus.txt \
  && rm -r ./wikipedia_corpus/final_* \

# Create HDF5 files for Bookcorpus
bash create_datasets_from_start.sh bookcorpus ./bookcorpus/bookcorpus.txt \
  && rm -r ./bookcorpus/final_* \

# Create HDF5 files for inter sequence-pair mixed Wikipedia and Bookcorpus
bash merge_datasets_after_creation.sh merged_wiki+books wikipedia_corpus/hdf5_shards,bookcorpus/hdf5_shards 1024
