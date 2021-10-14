/*
* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "contrib_ops/cuda/fastertransformer/utils/common.h"

#ifdef BUILD_GPT
#include "nccl.h"
#include "mpi.h"
#endif

struct ParallelParam : AbstractParam
{
    int rank{0};
    int world_size{1};
#ifdef BUILD_GPT
    ncclComm_t nccl_comm;
#endif

};

struct TensorParallelParam : public ParallelParam
{
  int local_head_num_{0};
  int local_hidden_units_{0};
};

struct LayerParallelParam : public ParallelParam
{
  int rank{0};
  int world_size{1};

  int layers_per_group{0};

  bool is_valid(int i)
  {
    if(i >= layers_per_group * rank && i < layers_per_group * (rank + 1)) return true;
    else return false;
  }
  int local_batch_size{-1};
};

#define MPICHECK(cmd) do {                                  \
    int e = cmd;                                            \
    if( e != MPI_SUCCESS ) {                                \
        printf("Failed: MPI error %s:%d '%d'\n",            \
                __FILE__,__LINE__, e);                      \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        printf("Failed: Cuda error %s:%d '%s'\n",           \
                __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r!= ncclSuccess) {                                  \
        printf("Failed, NCCL error %s:%d '%s'\n",           \
                __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

#ifdef BUILD_GPT
void get_nccl_uid(int rank, ncclUniqueId *uid);
#endif

template<typename T>
void all2all_reduce_sum(const T* send_buf, T* recv_buf, const int data_size,
                        ParallelParam param, cudaStream_t stream);

template<typename T>
void all2all_gather(const T* send_buf, T* recv_buf, const int data_size,
                    ParallelParam param, cudaStream_t stream);

template<typename T>
void nccl_send(const T* send_buf, const int data_size, const int peer, ParallelParam param, cudaStream_t stream);

template<typename T>
void nccl_recv(T* recv_buf, const int data_size, const int peer, ParallelParam param, cudaStream_t stream);

template<typename T>
void nccl_broadcast(T* buff, const int data_size, const int root, ParallelParam param, cudaStream_t stream);

#ifdef BUILD_GPT

template<typename T>
void nccl_recv(T* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

template<typename T>
void nccl_send(const T* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

#endif