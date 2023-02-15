// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <variant>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include <vector>
#include "core/providers/cpu/element_wise_ranged_transform.h"

using onnxruntime::narrow;
namespace onnxruntime {
namespace contrib {

using AOTFunc = int32_t (*)(const void **, ptrdiff_t, ptrdiff_t, const int64_t *, void **);

// Implement a new one instead of inheriting from ElementWiseRangedTransform so that we can call
// MlasComputeLogistic instead of using Eigen for better perf.
template <typename T>
class AOTanyOp : public OpKernel {
 public:
  AOTanyOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  AOTFunc func_{nullptr};
  int64_t func_type_{0};
  std::vector<TensorShapeVector> output_shapes_;
  std::vector<std::pair<int32_t,int32_t>> dynamic_dims_;
  std::vector<std::vector<uint16_t>> dynamic_dims_pair_;
  std::string func_name_;
};

}  // namespace contrib
}  // namespace onnxruntime
