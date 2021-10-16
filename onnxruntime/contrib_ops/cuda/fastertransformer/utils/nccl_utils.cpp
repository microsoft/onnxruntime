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

#include "contrib_ops/cuda/fastertransformer/utils/nccl_utils.h"

#ifdef PARALLEL_GPT

void get_nccl_uid(int rank, ncclUniqueId *uid) 
{
    if (rank == 0) 
    {
        NCCLCHECK( ncclGetUniqueId(uid));
    }
    MPICHECK( MPI_Bcast((void *)uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
}

template<typename T>
void all2all_reduce_sum(const T* send_buf, T* recv_buf, const int data_size,
                        ParallelParam param, cudaStream_t stream)
{
    if(param.world_size <= 1) return;

    ncclDataType_t nccl_data_type;
    if(std::is_same<T, float>::value) nccl_data_type = ncclFloat;
    else if(std::is_same<T, half>::value) nccl_data_type = ncclHalf;
    else if(std::is_same<T, int>::value) nccl_data_type = ncclInt;
    else
    {
        printf("[ERROR] reduce sum only support float, half and int. \n");
        exit(-1);
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce((const void*)send_buf, (void*)recv_buf, data_size, 
                             nccl_data_type, ncclSum, param.nccl_comm, stream));
    NCCLCHECK(ncclGroupEnd());

#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
}

template<typename T>
void all2all_gather(const T* send_buf, T* recv_buf, const int data_size,
                    ParallelParam param, cudaStream_t stream)
{
    if(param.world_size <= 1) return;
    ncclDataType_t nccl_data_type;
    if(std::is_same<T, float>::value) nccl_data_type = ncclFloat;
    else if(std::is_same<T, half>::value) nccl_data_type = ncclHalf;
    else if(std::is_same<T, int>::value) nccl_data_type = ncclInt;
    else
    {
        printf("[ERROR] all2all gather only support float, half and int. \n");
        exit(-1);
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllGather(send_buf + param.rank * data_size, recv_buf, data_size, nccl_data_type, param.nccl_comm, stream));
    NCCLCHECK(ncclGroupEnd());

#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
}

template<typename T>
void nccl_send(const T* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if(std::is_same<T, float>::value) nccl_data_type = ncclFloat;
    else if(std::is_same<T, half>::value) nccl_data_type = ncclHalf;
    else if(std::is_same<T, int>::value) nccl_data_type = ncclInt;
    else if(std::is_same<T, bool>::value) nccl_data_type = ncclInt8;
    else
    {
        printf("[ERROR] nccl_send only support float, half, int and bool. \n");
        exit(-1);
    }
    NCCLCHECK(ncclSend(send_buf, data_size, nccl_data_type, peer, comm, stream));

#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
    cudaDeviceSynchronize();
}

template void nccl_send(const float* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_send(const half* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_send(const int* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_send(const bool* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

template<typename T>
void nccl_recv(T* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if(std::is_same<T, float>::value) nccl_data_type = ncclFloat;
    else if(std::is_same<T, half>::value) nccl_data_type = ncclHalf;
    else if(std::is_same<T, int>::value) nccl_data_type = ncclInt;
    else if(std::is_same<T, bool>::value) nccl_data_type = ncclInt8;
    else
    {
        printf("[ERROR] nccl_recv only support float, half, int and bool. \n");
        exit(-1);
    }
    NCCLCHECK(ncclRecv(recv_buf, data_size, nccl_data_type, peer, comm, stream));

#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
    cudaDeviceSynchronize();
}

template void nccl_recv(float* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_recv(half* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_recv(int* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void nccl_recv(bool* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

template<typename T>
void nccl_broadcast(T* buff, const int data_size, const int root, ParallelParam param, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if(std::is_same<T, bool>::value) nccl_data_type = ncclInt8;
    else
    {
        printf("[ERROR] nccl_broadcast only support bool. \n");
        exit(-1);
    }
    NCCLCHECK(ncclBcast(buff, data_size, nccl_data_type, root, param.nccl_comm, stream));
    
#ifndef NDEBUG
    cudaDeviceSynchronize();
    CUDACHECK(cudaGetLastError());
#endif
    cudaDeviceSynchronize();
}

template void nccl_broadcast(bool* buff, const int data_size, const int root, ParallelParam param, cudaStream_t stream);

#else // PARALLEL_GPT

template<typename T>
void all2all_reduce_sum(const T* send_buf, T* recv_buf, const int data_size,
                        ParallelParam param, cudaStream_t stream){}

template<typename T>
void all2all_gather(const T* send_buf, T* recv_buf, const int data_size,
                    ParallelParam param, cudaStream_t stream){}
#endif

template void all2all_reduce_sum(const float* send_buf, float* recv_buf, const int data_size,
                                 ParallelParam param, cudaStream_t stream);

template void all2all_reduce_sum(const half* send_buf, half* recv_buf, const int data_size,
                                 ParallelParam param, cudaStream_t stream);

template void all2all_gather(const float* send_buf, float* recv_buf, const int data_size,
                             ParallelParam param, cudaStream_t stream);

template void all2all_gather(const half* send_buf, half* recv_buf, const int data_size,
                             ParallelParam param, cudaStream_t stream);
