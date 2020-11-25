// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/multi_tensor/common.cuh"

constexpr int PARALLEL_LOADS = 4;
constexpr int WARP_THREAD_COUNT = 32;
constexpr int MAX_BLOCK_COUNT = 288;
constexpr int MAX_TENSOR_COUNT = 128;
constexpr int MAX_BLOCK_THREAD_COUNT = 512;

namespace onnxruntime {
namespace rocm {

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t count);

template <typename TSrc>
class IsFiniteOp final : public RocmKernel {
 public:
  IsFiniteOp(const OpKernelInfo& info) : RocmKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t N);

template <typename TSrc>
class IsAllFiniteOp final : public RocmKernel {
 public:
  IsAllFiniteOp(const OpKernelInfo& info) : RocmKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
struct IsAllFiniteFunctor {
  void operator()(ChunkGroup<1> chunks, bool* output); 
};

}  // namespace rocm
}  // namespace onnxruntime