// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// SHARED_PROVIDER is defined in the in-tree CUDA EP shared library build
// (onnxruntime_providers_cuda). It gates out framework headers that are
// re-exported via the DLL-boundary proxy.  The plugin EP build uses a
// different flag (BUILD_CUDA_EP_AS_PLUGIN) and the force-include adapter
// headers instead.  Both builds need these headers excluded.
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/type_list.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/op_kernel_type_control_utils.h"
#endif

namespace onnxruntime {

// Add bf16 support for ConstantOfShape operator for phimm model.
// Although ONNX don't have bf16 support in opset-9 for ConstantOfShape we add support here:
// https://github.com/onnx/onnx/blob/main/docs/Changelog.md#constantofshape-9
using ConstantOfShapeDefaultOutputTypes =
    TypeList<
        MLFloat16,
        float, double,
        int8_t, int16_t, int32_t, int64_t,
        uint8_t, uint16_t, uint32_t, uint64_t,
        bool, BFloat16>;

using ConstantOfShapeDefaultOutputTypesOpset20 =
    TypeList<
        BFloat16,
        MLFloat16,
        float, double,
#if !defined(DISABLE_FLOAT8_TYPES)
        Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ,
#endif
        int8_t, int16_t, int32_t, int64_t,
        uint8_t, uint16_t, uint32_t, uint64_t,
        bool>;

// Opset 21 added int4 and uint4
// TODO(adrianlizarraga): Implement int4 and uint4 support.
using ConstantOfShapeDefaultOutputTypesOpset21 =
    TypeList<
        BFloat16,
        MLFloat16,
        float, double,
#if !defined(DISABLE_FLOAT8_TYPES)
        Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ,
#endif
        int8_t, int16_t, int32_t, int64_t,
        uint8_t, uint16_t, uint32_t, uint64_t,
        bool>;

// Opset 23 added support for float4e2m1.
// TODO(titaiwang): Add support for float4e2m1.
using ConstantOfShapeDefaultOutputTypesOpset23 =
    TypeList<
        BFloat16,
        MLFloat16,
        float, double,
#if !defined(DISABLE_FLOAT8_TYPES)
        Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ,
#endif
        int8_t, int16_t, int32_t, int64_t,
        uint8_t, uint16_t, uint32_t, uint64_t,
        bool>;

#define ORT_CONSTANT_OF_SHAPE_VALUE_TYPES(M)          \
  M(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)         \
  M(float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)       \
  M(MLFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) \
  M(double, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)     \
  M(int8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)       \
  M(int16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)     \
  M(int32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)     \
  M(int64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)     \
  M(uint8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)     \
  M(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)   \
  M(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)   \
  M(uint64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)   \
  M(BFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)

template <typename T>
struct ConstantOfShapeOrtType;

#define DEFINE_CONSTANT_OF_SHAPE_ORT_TYPE(c_type, ort_type)      \
  template <>                                                    \
  struct ConstantOfShapeOrtType<c_type> {                        \
    static constexpr ONNXTensorElementDataType value = ort_type; \
  };

ORT_CONSTANT_OF_SHAPE_VALUE_TYPES(DEFINE_CONSTANT_OF_SHAPE_ORT_TYPE)

#undef DEFINE_CONSTANT_OF_SHAPE_ORT_TYPE

class ConstantOfShapeCore {
 protected:
  void* GetValuePtr() const { return p_value_; }

  template <typename ContextType>
  static Status PrepareCompute(ContextType* ctx, Tensor** output_tensor) {
    const auto shape_tensor = ctx->template Input<Tensor>(0);
    const auto& input_shape = shape_tensor->Shape();

    // If empty the output is a scalar with empty shape
    // TensorShape::Size() will still return 1 and we will output
    // one value
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() > 0, "Must have a valid input shape.");

    const auto span = shape_tensor->template DataAsSpan<int64_t>();

    TensorShape output_shape(span);
    (*output_tensor) = ctx->Output(0, output_shape);

    return Status::OK();
  }

  void SetDefaultValue() {
    float f_value = 0.f;
    SetValue(sizeof(float), &f_value);
  }

  template <typename EnabledOutputTypeList>
  void SetValueFromOrtTensor(ONNXTensorElementDataType tensor_type, const void* data) {
    bool handled = false;
    switch (tensor_type) {
#define CASE_SET_ORT_VALUE(c_type, ort_type)               \
  case ConstantOfShapeOrtType<c_type>::value: {            \
    if (utils::HasType<EnabledOutputTypeList, c_type>()) { \
      SetValue(sizeof(c_type), data);                      \
      handled = true;                                      \
    }                                                      \
    break;                                                 \
  }
      ORT_CONSTANT_OF_SHAPE_VALUE_TYPES(CASE_SET_ORT_VALUE)
#undef CASE_SET_ORT_VALUE
      default:
        ORT_THROW("Unsupported value attribute datatype: ", static_cast<int>(tensor_type));
    }

    ORT_ENFORCE(handled, "Unsupported value attribute datatype in this build: ", static_cast<int>(tensor_type));
  }

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

 private:
  union SizeBasedValue {
    int8_t int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;
  };
  mutable SizeBasedValue s_value_{};
  mutable void* p_value_ = nullptr;
};

template <typename EnabledOutputTypeList = ConstantOfShapeDefaultOutputTypes>
class ConstantOfShapeBase : public ConstantOfShapeCore {
 protected:
  ConstantOfShapeBase(const OpKernelInfo& info) {
#ifndef SHARED_PROVIDER
    ONNX_NAMESPACE::TensorProto t_proto;
    auto* t_proto_p = &t_proto;
#else
    auto t_proto = ONNX_NAMESPACE::TensorProto::Create();
    auto* t_proto_p = t_proto.get();
#endif
    if (info.GetAttr<ONNX_NAMESPACE::TensorProto>("value", t_proto_p).IsOK()) {
      for (auto dim : t_proto_p->dims()) {
        ORT_ENFORCE(dim == 1, "The value attribute of ConstantOfShape must be a single-element tensor");
      }
      SetValueFromTensorProto(*t_proto_p);
    } else {
      SetDefaultValue();
    }
  }

  void SetValueFromTensorProto(const ONNX_NAMESPACE::TensorProto&);
};

// ort_type parameter unused here but required for ORT_CONSTANT_OF_SHAPE_VALUE_TYPES X-macro conformance.
#define CASE_FETCH_VALUE_DATA(c_type, ort_type)                                          \
  case utils::ToTensorProtoElementType<c_type>(): {                                      \
    if (utils::HasType<EnabledOutputTypeList, c_type>()) {                               \
      c_type val;                                                                        \
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1)); \
      SetValue(sizeof(c_type), reinterpret_cast<void*>(&val));                           \
      handled = true;                                                                    \
    }                                                                                    \
    break;                                                                               \
  }

template <typename EnabledOutputTypeList>
void ConstantOfShapeBase<EnabledOutputTypeList>::SetValueFromTensorProto(const ONNX_NAMESPACE::TensorProto& t_proto) {
  ORT_ENFORCE(utils::HasDataType(t_proto));
  ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(t_proto.data_type()));
  ORT_ENFORCE(!utils::HasExternalData(t_proto),
              "Tensor proto with external data for value attribute is not supported.");
  const auto tensor_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = utils::HasRawData(t_proto) ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = utils::HasRawData(t_proto) ? t_proto.raw_data().size() : 0;
  bool handled = false;
  switch (tensor_type) {
    ORT_CONSTANT_OF_SHAPE_VALUE_TYPES(CASE_FETCH_VALUE_DATA)
    default:
      ORT_THROW("Unsupported value attribute datatype: ", tensor_type);
  }

  ORT_ENFORCE(handled, "Unsupported value attribute datatype in this build: ", tensor_type);
}

#undef CASE_FETCH_VALUE_DATA
#undef ORT_CONSTANT_OF_SHAPE_VALUE_TYPES

}  // namespace onnxruntime
