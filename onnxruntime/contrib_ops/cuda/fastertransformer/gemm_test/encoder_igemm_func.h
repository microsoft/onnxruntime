/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdlib.h>
//#include <unistd.h>
#include <map>

namespace fastertransformer{

typedef struct {
    int algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize;
    //only used in cublasLt >= 11.0
    int stages;
    float exec_time;
} cublasLtMatmulAlgo_info;
/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

/* CAUTION : must match cublasLtMatmulTile_t */
const char * const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8"   ,
    "8x32"   ,
    "16x16"  ,
    "32x8"   ,
    "8x64"   ,
    "16x32"  ,
    "32x16"  ,
    "64x8"   ,
    "32x32"  ,
    "32x64"  ,
    "64x32"  ,
    "32x128" ,
    "64x64"  ,
    "128x32" ,
    "64x128" ,
    "128x64" ,
    "64x256" ,
    "128x128",
    "256x64" ,
    "64x512" ,
    "128x256",
    "256x128",
    "512x64" ,
};


int generate_encoder_igemm_config(int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend = true);

size_t calGemmTestBufSizeInByte(int batch_size,
                                int seq_len,
                                int head_num,
                                int size_per_head,
                                int int8_mode,
                                int is_fp16);
}
