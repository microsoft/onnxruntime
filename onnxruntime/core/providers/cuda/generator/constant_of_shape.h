// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/cpu/generator/constant_of_shape_base.h"
#endif
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN

// Plugin build: self-contained ConstantOfShape without ConstantOfShapeBase dependency.
// ConstantOfShapeBase uses TensorProto/UnpackTensor utilities not available in the plugin,
// so we read the 'value' attribute via the ORT C API (KernelInfoGetAttribute_tensor) instead.
class ConstantOfShape final : public CudaKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : CudaKernel(info) {
    InitValue(info);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstantOfShape);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 protected:
  void* GetValuePtr() const { return p_value_; }

  static Status PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) {
    const auto* shape_tensor = ctx->Input<Tensor>(0);
    const auto& input_shape = shape_tensor->Shape();
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() > 0, "Must have a valid input shape.");
    const auto span = shape_tensor->DataAsSpan<int64_t>();
    TensorShape output_shape(span);
    (*output_tensor) = ctx->Output(0, output_shape);
    return Status::OK();
  }

 private:
  union SizeBasedValue {
    int8_t int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;
  };

  mutable SizeBasedValue s_value_{};
  mutable void* p_value_ = nullptr;

  void SetValue(size_t size, const void* value) {
    switch (size) {
      case sizeof(int8_t):
        s_value_.int8_ = *(reinterpret_cast<const int8_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int8_));
        break;
      case sizeof(int16_t):
        s_value_.int16_ = *(reinterpret_cast<const int16_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int16_));
        break;
      case sizeof(int32_t):
        s_value_.int32_ = *(reinterpret_cast<const int32_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int32_));
        break;
      case sizeof(int64_t):
        s_value_.int64_ = *(reinterpret_cast<const int64_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int64_));
        break;
      default:
        ORT_THROW("Unsupported value attribute datatype with size: ", size);
    }
  }

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
        const void* data = value_tensor.GetTensorRawData();
        size_t elem_size = GetElementSize(type_and_shape.GetElementType());
        SetValue(elem_size, data);
      } else {
        float f_value = 0.f;
        SetValue(sizeof(float), &f_value);
      }
    } catch (const Ort::Exception&) {
      float f_value = 0.f;
      SetValue(sizeof(float), &f_value);
    }
  }

  static size_t GetElementSize(ONNXTensorElementDataType type) {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return 1;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return 8;
      default:
        ORT_THROW("Unsupported element type for ConstantOfShape: ", static_cast<int>(type));
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
