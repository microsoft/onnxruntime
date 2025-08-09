/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef ENABLE_FP4
#include <cuda_runtime_api.h>
#include <vector>

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"
#include "contrib_ops/cuda/llm/common/quantization.h"

namespace tk = onnxruntime::llm::common;
namespace tkc = onnxruntime::llm::cutlass_extensions;

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {

/*
  This runner supports:
  FP4 inputs (A and B)
  float blockwise scaling factor
  float alpha scalings
  T output (D) where T = {float, half, __nv_bfloat16}

  Activations, biases and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
  Block scaling factor are interleaved.
*/

class CutlassFp4GemmRunnerInterface {
 public:
  CutlassFp4GemmRunnerInterface() {}

  virtual ~CutlassFp4GemmRunnerInterface() {}

  virtual void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
                    float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
                    char* workspace, const size_t workspaceBytes, cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) = 0;

  virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

enum class FP4GemmType {
  W4A4_NVFP4_NVFP4,
  W4A8_MXFP4_MXFP8,
};

template <typename T, FP4GemmType gemmType = FP4GemmType::W4A4_NVFP4_NVFP4>
class CutlassFp4GemmRunner : public virtual CutlassFp4GemmRunnerInterface {
 public:
  CutlassFp4GemmRunner();
  ~CutlassFp4GemmRunner();

  void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
            float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
            char* workspace, const size_t workspaceBytes, cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSize(int const m, int const n, int const k, int const batch_count) override;

  std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

 private:
  size_t dispatchToArch(T* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
                        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
                        char* workspace, const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

  size_t getWorkspaceSizeImpl(int const m, int const n, int const k, int const batch_count);

  int mSm;
  int mMultiProcessorCount;
};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
#endif  // ENABLE_FP4
