// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "contrib_ops/rocm/bert/tunable_op.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.h"

using onnxruntime::contrib::rocm::Op;
using onnxruntime::contrib::rocm::OpParams;
using onnxruntime::contrib::rocm::TunableOp;

namespace onnxruntime {

template<typename T>
struct VectorAddParams : OpParams {
  VectorAddParams(hipStream_t stream, const T* x, const T* y, T* z, int n) :
    OpParams(stream), x(x), y(y), z(z), n(n) {}

  std::string signature() const {
    return std::to_string(n);
  }

  const T* x;
  const T* y;
  T* z;
  int n;
};

template <typename T, int ThreadsPerBlock, int VecSize>
class VectorAddOp : public Op {
 public:
  VectorAddOp() : Op() {}

  void Run(const OpParams* op_params) {
    const VectorAddParams<T>* vector_add_params = static_cast<const VectorAddParams<T>*>(op_params);
    LaunchVectorAdd<T, ThreadsPerBlock, VecSize>(vector_add_params->stream,
                                                 vector_add_params->x,
                                                 vector_add_params->y,
                                                 vector_add_params->z,
                                                 vector_add_params->n);
  }
};

#define ADD_OP(threads_per_block)                                            \
  ops_.push_back(std::make_unique<VectorAddOp<T, threads_per_block, 1>>());  \
  ops_.push_back(std::make_unique<VectorAddOp<T, threads_per_block, 2>>());  \
  ops_.push_back(std::make_unique<VectorAddOp<T, threads_per_block, 4>>());  \
  ops_.push_back(std::make_unique<VectorAddOp<T, threads_per_block, 8>>());

template <typename T>
class VectorAddTunableOp : public TunableOp {
 public:
  VectorAddTunableOp() : TunableOp(4) {
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
  virtual bool Condition(const OpParams* /*op_params*/) {
    return true;
  }
};

}  // namespace onnxruntime
