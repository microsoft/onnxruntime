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

#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "core/framework/tensor_shape.h"
#include "core/providers/shared_library/provider_api.h"
//#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::llm {
class IGemmRunner {
 public:
  virtual ~IGemmRunner() = default;

  virtual TensorShape GetOutputShape(const onnxruntime::TensorShape& x, const onnxruntime::TensorShape& w) const = 0;

  virtual void Run(const onnxruntime::Tensor* x,
                   const onnxruntime::Tensor* w,
                   onnxruntime::Tensor* y,
                   void* workspace,
                   cudaStream_t stream,
                   cublasHandle_t cublas_handle,
                   cublasLtHandle_t cublasLt_handle) = 0;
};

class GemmRunner: public IGemmRunner {
 public:
  GemmRunner();
  ~GemmRunner() = default;

  void Initialize(int pad_lda, int pad_ldb, int pad_ldc, bool transA, bool transB, int output_dtype, float alpha);

  TensorShape GetOutputShape(const onnxruntime::TensorShape& x, const onnxruntime::TensorShape& w) const override;

  size_t GetWorkspaceSize() const;

  void Run(const onnxruntime::Tensor* x,
           const onnxruntime::Tensor* w,
           onnxruntime::Tensor* y,
           void* workspace,
           cudaStream_t stream,
           cublasHandle_t cublas_handle,
           cublasLtHandle_t cublasLt_handle) override;

  void Configure(const onnxruntime::TensorShape& min_x, const onnxruntime::TensorShape& max_x, const onnxruntime::TensorShape& w);

  static std::unique_ptr<IGemmRunner> Create(int pad_lda,
                                            int pad_ldb,
                                            int pad_ldc,
                                            bool transA,
                                            bool transB,
                                            int output_dtype,
                                            float alpha);

 private:
  // hide the implementation details to help build.
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

}  // namespace onnxruntime::llm
