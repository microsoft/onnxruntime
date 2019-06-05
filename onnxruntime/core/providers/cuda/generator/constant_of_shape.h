// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/generator/constant_of_shape.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

class ConstantOfShape final : public ConstantOfShapeBase, public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::unique_ptr<cuda::IConstantBuffer<float>> constant_float_;
  std::unique_ptr<cuda::IConstantBuffer<double>> constant_double_;
  std::unique_ptr<cuda::IConstantBuffer<half>> constant_half_;
  std::unique_ptr<cuda::IConstantBuffer<bool>> constant_bool_;
  std::unique_ptr<cuda::IConstantBuffer<int8_t>> constant_int8_;
  std::unique_ptr<cuda::IConstantBuffer<int16_t>> constant_int16_;
  std::unique_ptr<cuda::IConstantBuffer<int32_t>> constant_int32_;
  std::unique_ptr<cuda::IConstantBuffer<int64_t>> constant_int64_;
  std::unique_ptr<cuda::IConstantBuffer<uint8_t>> constant_uint8_;
  std::unique_ptr<cuda::IConstantBuffer<uint16_t>> constant_uint16_;
  std::unique_ptr<cuda::IConstantBuffer<uint32_t>> constant_uint32_;
  std::unique_ptr<cuda::IConstantBuffer<uint64_t>> constant_uint64_;
};

}  // namespace cuda
}  // namespace onnxruntime
