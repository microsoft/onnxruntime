/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"
#include "contrib_ops/cuda/llm/common/quantization.h"

#include <cuda_runtime_api.h>
#include <vector>

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {

namespace tk = onnxruntime::llm::common;
namespace tkc = onnxruntime::llm::cutlass_extensions;

/*
  This runner supports:

  Activations and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

class CutlassFusedGatedGemmRunnerInterface {
 public:
  CutlassFusedGatedGemmRunnerInterface() {}

  virtual ~CutlassFusedGatedGemmRunnerInterface() {}

  virtual void gemm(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m,
                    int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
                    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

  virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class CutlassFusedGatedGemmRunner : public virtual CutlassFusedGatedGemmRunnerInterface {
 public:
  CutlassFusedGatedGemmRunner();
  ~CutlassFusedGatedGemmRunner();

  void gemm(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m, int n, int k,
            float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig, char* workspace,
            size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) override;

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSize(int const m, int const n, int const k) override;

  std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

 private:
  size_t dispatchToArch(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m,
                        int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
                        char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

  size_t getWorkspaceSizeImpl(int const m, int const n, int const k);

  int mSm;
};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
