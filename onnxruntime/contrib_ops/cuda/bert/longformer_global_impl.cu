/*
Copyright (c) NVIDIA Corporation and Microsoft Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cub/cub.cuh>
#include <cub/device/device_partition.cuh>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "longformer_global_impl.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetGlobalScratchSize(int sequence_length) {
  // Global Index scratch layout:
  //   [sequence_index: int S][tmp_storage: int 1024x1]
  return sizeof(int) * (sequence_length + 1024);
}

__global__ void InitSequenceIndexKernel(int* sequence_index, int sequence_length) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sequence_length; i += blockDim.x) {
    sequence_index[i] = i;
  }
}

Status BuildGlobalIndex(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const int* global_attention,
    int batch_size,
    int sequence_length,
    int* global_index,
    int* batch_global_num,
    void* scratch,
    size_t scratch_size) {
  int* sequence_index = (int*)scratch;
  int* tmp_storage = sequence_index + sequence_length;

  const int threads = device_prop.maxThreadsPerBlock;
  int blocks = CeilDiv(sequence_length, threads);
  InitSequenceIndexKernel<<<blocks, threads, 0, stream>>>(sequence_index, sequence_length);

  // Determine temporary device storage size.
  // For int* inputs/outputs, it need 767 bytes. We reserved 1024*4 bytes, which shall be enough.
  size_t temp_storage_bytes = 0;
  CUDA_RETURN_IF_ERROR(cub::DevicePartition::Flagged(
      NULL, temp_storage_bytes, sequence_index,
      global_attention, global_index, batch_global_num, sequence_length, stream));
  if (temp_storage_bytes + sizeof(int) * sequence_length > scratch_size) {
    ORT_THROW("LongformerAttention scratch space is not large enough. Temp storage bytes are", temp_storage_bytes);
  }

  // Find the global attention indices and number of global attention tokens
  for (int i = 0; i < batch_size; ++i) {
    CUDA_RETURN_IF_ERROR(cub::DevicePartition::Flagged(
        reinterpret_cast<void*>(tmp_storage), temp_storage_bytes, sequence_index,
        global_attention + i * sequence_length, global_index + i * sequence_length,
        batch_global_num + i, sequence_length, stream));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
