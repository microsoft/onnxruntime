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

template<typename T>
struct FastGeluParams : OpParams {
  FastGeluParams(hipStream_t stream, const T* input, const T* bias, T* output, int input_length, int bias_length) :
    OpParams(stream), input(input), bias(bias), output(output), input_length(input_length), bias_length(bias_length) {}

  std::string signature() const {
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
void LaunchFastGelu(hipStream_t stream, const T* input, const T* bias, T* output, int input_length, int bias_length) {
  hipLaunchKernelGGL((FastGeluKernelVec<T, ThreadsPerBlock, VecSize>),
                  dim3(CeilingDivision(input_length, ThreadsPerBlock*VecSize)),
                  dim3(ThreadsPerBlock),
                  0, stream,
                  input_length, bias_length, input, bias, output);
}


template <typename T, int ThreadsPerBlock, int VecSize>
class FastGeluOp : public Op {
 public:
  FastGeluOp() : Op() {}

  void Run(const OpParams* op_params) {
    const FastGeluParams<T>* fast_gelu_params = static_cast<const FastGeluParams<T>*>(op_params);
    LaunchFastGelu<T, ThreadsPerBlock, VecSize>(fast_gelu_params->stream,
                                                fast_gelu_params->input,
                                                fast_gelu_params->bias,
                                                fast_gelu_params->output,
                                                fast_gelu_params->input_length,
                                                fast_gelu_params->bias_length);
  }
};

#define ADD_OP(threads_per_block)                                           \
  ops_.push_back(std::make_unique<FastGeluOp<T, threads_per_block, 1>>());  \
  ops_.push_back(std::make_unique<FastGeluOp<T, threads_per_block, 2>>());  \
  ops_.push_back(std::make_unique<FastGeluOp<T, threads_per_block, 4>>());  \
  ops_.push_back(std::make_unique<FastGeluOp<T, threads_per_block, 8>>());  \
  ops_.push_back(std::make_unique<FastGeluOp<T, threads_per_block, 16>>());

template<typename T>
class FastGeluTunableOp : public TunableOp {
 public:
  FastGeluTunableOp() : TunableOp(15) {
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
  virtual bool Condition(const OpParams* op_params) {
    const FastGeluParams<T>* fast_gelu_params = static_cast<const FastGeluParams<T>*>(op_params);
    bool condition = (fast_gelu_params->bias_length > 0) && (fast_gelu_params->bias_length % 16 == 0);
    return condition;
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
