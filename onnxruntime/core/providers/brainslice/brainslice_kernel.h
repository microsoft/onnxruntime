// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "3rdparty/half.hpp"
#include "CommonFunctions_client.h"
#include "core/providers/brainslice/brainslice_fwd.h"

namespace onnxruntime {
namespace brainslice {
using BS_Half = half_float::half;

class BrainSliceExecutionProvider;
enum class ParameterUsage {
  USE_AS_MATRIX = 0,
  USE_AS_VECTOR
};

struct BrainSliceParameterInitPlan {
  std::unique_ptr<Tensor> tensor;
  ParameterUsage usage;
  size_t axis;
  bool need_transpose;
  ISA_Mem mem_type;
  size_t address;
  ISA_Mem target_mem_type;
};

class BrainSliceOpKernel : public OpKernel {
 public:
  explicit BrainSliceOpKernel(const OpKernelInfo& info);
  virtual ~BrainSliceOpKernel() {}

  virtual Status Compute(OpKernelContext* context) const = 0;

  template <typename T>
  static Status UploadBrainSliceParameter(BrainSliceParameterInitPlan& plan, BrainSliceExecutionProvider* provider, uint32_t* num_tiles);

 protected:
  BrainSliceExecutionProvider* provider_;
  const int64_t native_dim_;
};

template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace brainslice
}  // namespace onnxruntime
