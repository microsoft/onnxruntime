/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "contrib_ops/cuda/llm/gemm_runner.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::llm;

namespace onnxruntime {
namespace contrib {
namespace cuda {

class FlexGemm final : public CudaKernel {
 public:
  FlexGemm(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<IGemmRunner> gemm_runner_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
