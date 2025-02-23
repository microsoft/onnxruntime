/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "contrib_ops/cuda/llm/flex_gemm.h"

using namespace onnxruntime::common;

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    FlexGemm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, BFloat16>()),
    FlexGemm);

FlexGemm::FlexGemm(const OpKernelInfo& info) : CudaKernel(info), gemm_runner_(nullptr) {
  bool transA = info.GetAttrOrDefault<int64_t>("transa", 0) == 1;
  bool transB = info.GetAttrOrDefault<int64_t>("transb", 0) == 1;
  int pad_lda = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_lda", 0));
  int pad_ldb = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_ldb", 0));
  int pad_ldc = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_ldc", 0));
  int output_dtype = static_cast<int>(info.GetAttrOrDefault<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  int alpha = info.GetAttrOrDefault<float>("alpha", 1.0f);
  gemm_runner_ = GemmRunner::Create(pad_lda, pad_ldb, pad_ldc, transA, transB, output_dtype, alpha);
}

Status FlexGemm::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);

  TensorShape output_shape = gemm_runner_->GetOutputShape(X->Shape(), W->Shape());
  auto* Y = context->Output(0, output_shape);

  // const int64_t M = computeMDimension(transA_, X->Shape()) - pad_m;
  // if (M == 0) {
  //   return Status::OK();
  // }

  void* workspace = nullptr;
  cudaStream_t stream = this->Stream(context);
  cublasHandle_t cublas_handle = GetCublasHandle(context);
  cublasLtHandle_t cublasLt_handle = CublasLtHandle();
  gemm_runner_->Run(X, W, Y, workspace, stream, cublas_handle, cublasLt_handle);

  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
