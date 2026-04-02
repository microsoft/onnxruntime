// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN

// Plugin build: keep the attribute fetch self-contained while reusing the shared
// ConstantOfShapeCore helpers for default handling and supported type mapping.
// ConstantOfShapeBase still depends on TensorProto/UnpackTensor utilities that the
// plugin build avoids, so the plugin path reads the attribute via the ORT C API instead.
class ConstantOfShape final : public ConstantOfShapeCore, public CudaKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : CudaKernel(info) {
    InitValue(info);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstantOfShape);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  void InitValue(const OpKernelInfo& info) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto ort_info = info.GetKernelInfo();
    try {
      Ort::Value value_tensor = ort_info.GetTensorAttribute("value", allocator);
      auto type_and_shape = value_tensor.GetTensorTypeAndShapeInfo();
      size_t elem_count = type_and_shape.GetElementCount();
      ORT_ENFORCE(elem_count == 1 || elem_count == 0,
                  "The value attribute of ConstantOfShape must be a single-element tensor");
      if (elem_count == 1) {
        SetValueFromOrtTensor<ConstantOfShapeDefaultOutputTypes>(
            type_and_shape.GetElementType(), value_tensor.GetTensorRawData());
      } else {
        SetDefaultValue();
      }
    } catch (const Ort::Exception&) {
      SetDefaultValue();
    }
  }
};

#else  // !BUILD_CUDA_EP_AS_PLUGIN

class ConstantOfShape final : public ConstantOfShapeBase<>, public CudaKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), CudaKernel(info) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstantOfShape);

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

#endif  // BUILD_CUDA_EP_AS_PLUGIN

}  // namespace cuda
}  // namespace onnxruntime
