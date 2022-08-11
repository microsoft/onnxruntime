// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include "contrib_ops/rocm/bert/tunable_op.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct FastGeluParams : OpParams {
  FastGeluParams(hipStream_t stream, const T* input, const T* bias, T* output, int input_length, int bias_length) :
    OpParams(stream), input(input), bias(bias), output(output), input_length(input_length), bias_length(bias_length) {}

  std::string Signature() const {
    std::string sig = std::to_string(input_length) + "_" + std::to_string(bias_length);
    return sig;
  }

  const T* input;
  const T* bias;
  T* output;
  int input_length;
  int bias_length;
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status FastGeluOp(const FastGeluParams<T>* params) {
  hipLaunchKernelGGL((FastGeluKernelVec<T, ThreadsPerBlock, VecSize>),
                     dim3(CeilingDivision(params->input_length, ThreadsPerBlock * VecSize)),
                     dim3(ThreadsPerBlock),
                     0, params->stream,
                     params->input_length, params->bias_length, params->input, params->bias, params->output);
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, hipGetErrorName(status)));
  return Status::OK();
}

#define ADD_OP(threads_per_block)                               \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 1>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 2>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 4>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 8>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 16>);

template <typename T>
class FastGeluTunableOp : public TunableOp<FastGeluParams<T>> {
 public:
  FastGeluTunableOp() {
    ADD_OP(64);
    ADD_OP(128);
    ADD_OP(192);
    ADD_OP(256);
    ADD_OP(320);
    ADD_OP(384);
    ADD_OP(448);
    ADD_OP(512);
  }

 private:
  bool Condition(const FastGeluParams<T>* fast_gelu_params) override {
    bool condition = (fast_gelu_params->bias_length > 0) && (fast_gelu_params->bias_length % 16 == 0);
    return condition;
  }
};

#undef ADD_OP

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
