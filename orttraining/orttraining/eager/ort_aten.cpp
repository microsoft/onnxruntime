// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
#include "ort_tensor.h"
#include <c10/core/TensorImpl.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/InferSize.h>

#include <torch/csrc/jit/ir/ir.h>
#include <c10/util/irange.h>
#include <ATen/WrapDimUtils.h>


namespace torch_ort {
namespace eager {

//#pragma region Helpers
using NodeAttributes = onnxruntime::NodeAttributes;
namespace {
  inline bool is_device_supported(at::DeviceType type) {
    return type == at::kORT || type == at::kCPU;
  }

  inline void assert_tensor_supported(const at::Tensor& tensor) {
    if (tensor.is_sparse()) {
      throw std::runtime_error("ORT copy: sparse not supported");
    }

    if (tensor.is_quantized()) {
      throw std::runtime_error("ORT copy: quantized not supported");
    }

    if (!is_device_supported(tensor.device().type())) {
      throw std::runtime_error("ORT copy: device not supported");
    }
  }
}

at::Tensor aten_tensor_from_ort(
  OrtValue&& ot,
  const at::TensorOptions& options) {
  return at::Tensor(c10::make_intrusive<ORTTensorImpl>(
    std::move(ot),
    options));
}

const std::vector<at::Tensor> aten_tensor_from_ort(
  std::vector<OrtValue>& ortvalues,
  const at::TensorOptions& options) {
    const size_t num_outputs = ortvalues.size();
    std::vector<at::Tensor> atvalues = std::vector<at::Tensor>(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      atvalues[i] = at::Tensor(c10::make_intrusive<ORTTensorImpl>(
        std::move(ortvalues[i]),
        options));
    }
    return atvalues;
}

onnxruntime::MLDataType ort_scalar_type_from_aten(
  at::ScalarType dtype) {
  switch (dtype){
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    case at::kBool:
      return onnxruntime::DataTypeImpl::GetType<bool>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar) {
  return create_ort_value(invoker, scalar, at::kFloat);
}

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar,
  at::ScalarType type) {
  float val = scalar.toFloat();
  OrtValue ort_val;
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(type), onnxruntime::TensorShape({}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ort_val);
  auto* ort_tensor = ort_val.GetMutable<onnxruntime::Tensor>();
  switch (type) {
    case at::ScalarType::Float:
      CopyVectorToTensor<float>(invoker, &val, 1, *ort_tensor);
      break;
    case at::ScalarType::BFloat16: {
      at::BFloat16 valBFloat16 = scalar.toBFloat16();
      Ort::BFloat16_t *valOrtBFloat16 = reinterpret_cast<Ort::BFloat16_t *>(&valBFloat16);
      CopyVectorToTensor<Ort::BFloat16_t>(invoker, valOrtBFloat16, 1, *ort_tensor);
      break;
    }      
    default:
      // TODO: support more types
      // For most at::ScalarType, it should be safe to just call value.to<>
      // on it, but for now we want to explicitly know when we've encountered
      // a new scalar type while bringing up ORT eager mode.
      ORT_THROW("Unsupported: at::ScalarType::", scalar.type());
  }
  return ort_val;
}

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Tensor& tensor) {
  assert_tensor_supported(tensor);

  auto* impl = dynamic_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl) {
    return impl->tensor();
  }

  OrtMemoryInfo *mem_info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
  auto element_type = ort_scalar_type_from_aten(tensor.scalar_type());

  OrtValue ort_tensor;
  onnxruntime::Tensor::InitOrtValue(element_type, onnxruntime::TensorShape(tensor.sizes().vec()), tensor.data_ptr(),
                                    *mem_info, ort_tensor, 0L /* offset = 0 - because tensor.data_ptr() includes the underyling offset */,
                                    tensor.strides().vec());
  return ort_tensor;
}

OrtValue create_ort_value(const at::Tensor& tensor){
  auto& invoker = GetORTInvoker(tensor.device());
  return create_ort_value(invoker, tensor);
}

std::vector<OrtValue> create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  at::TensorList values) {
    auto output = std::vector<OrtValue>{};
    for (auto element: values){
      output.push_back(create_ort_value(element));
    }
    return output;
}

onnx::AttributeProto create_ort_attribute(
  const char* name,
  at::Scalar value) {
  return create_ort_attribute(name, value, value.type());
}

onnx::AttributeProto create_ort_attribute(
  const char* name,
  at::Scalar value,
  at::ScalarType type) {
  onnx::AttributeProto attr;
  attr.set_name(name);
  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::Double:
      attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
      attr.set_f(value.to<double>());
      break;
    case at::ScalarType::Bool:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
      attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
      attr.set_i(value.to<int64_t>());
      break;
    default:
      // For most at::ScalarType, it should be safe to just call value.to<>
      // on it, but for now we want to explicitly know when we've encountered
      // a new scalar type while bringing up ORT eager mode.
      ORT_THROW("Unsupported: at::ScalarType::", value.type());
  }

  return attr;
}

onnx::AttributeProto create_ort_attribute(
  const char* name,
  const char* value) {
  onnx::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  attr.set_s(value);
  return attr;
}

bool IsSupportedType(at::Scalar scalar, const std::vector<at::ScalarType>& valid_types){
  return std::find(valid_types.begin(), valid_types.end(), scalar.type()) != valid_types.end();
}

bool IsSupportedType(at::Tensor tensor, const std::vector<at::ScalarType>& valid_types){
  return std::find(valid_types.begin(), valid_types.end(), tensor.scalar_type()) != valid_types.end();
}

bool IsSupportedType(at::IntArrayRef arrary, const std::vector<at::ScalarType>& valid_types){
  return std::find(valid_types.begin(), valid_types.end(), at::kInt) != valid_types.end() ||
         std::find(valid_types.begin(), valid_types.end(), at::kLong) != valid_types.end();
}

bool IsSupportedType(int64_t val, const std::vector<at::ScalarType>& valid_types){
  return std::find(valid_types.begin(), valid_types.end(), at::kLong) != valid_types.end();
}

bool IsSupportedType(c10::optional<int64_t> val, const std::vector<at::ScalarType>& valid_types){
  return IsSupportedType(val.value(), valid_types);
}

bool IsSupportedType(at::TensorList tensors, const std::vector<at::ScalarType>& valid_types){
  return IsSupportedType(tensors[0], valid_types);
}

ONNX_NAMESPACE::TensorProto_DataType GetONNXTensorProtoDataType(at::ScalarType dtype){
  switch (dtype){
    case at::kFloat:
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case at::kDouble:
      return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    case at::kHalf:
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case at::kBFloat16:
      return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    case at::kInt:
      return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    case at::kShort:
      return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    case at::kLong:
      return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    case at::kBool:
      return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

static c10::optional<at::ScalarType> PromoteScalarTypes(
    const std::vector<at::ScalarType>& types) {
  if (types.empty()) {
    return at::nullopt;
  }
  auto st = types[0];
  for (const auto i : c10::irange(1, types.size())) {
    st = c10::promoteTypes(st, types[i]);
  }
  return st;
}


c10::optional<at::ScalarType> PromoteScalarTypesWithCategory(
    const std::vector<at::ScalarType>& typesFromTensors,
    const std::vector<at::ScalarType>& typesFromScalars) {
  auto typeFromTensor = PromoteScalarTypes(typesFromTensors);
  auto typeFromScalar = PromoteScalarTypes(typesFromScalars);

  auto getTypeCategory = [](c10::ScalarType t) {
    if (c10::kBool == t) {
      return 1;
    }
    if (c10::isIntegralType(t, /*includeBool=*/false)) {
      return 2;
    }
    if (c10::isFloatingType(t)) {
      return 3;
    }
    return 0;
  };

  if (c10::nullopt == typeFromScalar) {
    return typeFromTensor;
  } else if (c10::nullopt == typeFromTensor) {
    return typeFromScalar;
  }

  auto typeCategoryFromTensor = getTypeCategory(typeFromTensor.value());
  auto typeCategoryFromScalar = getTypeCategory(typeFromScalar.value());

  if (typeCategoryFromScalar > typeCategoryFromTensor) {
    return typeFromScalar;
  }
  return typeFromTensor;
}

OrtValue CastToType(onnxruntime::ORTInvoker& invoker, const OrtValue& input, at::ScalarType type){
  std::vector<OrtValue> output(1);
  NodeAttributes attrs(2);
  attrs["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(type), at::ScalarType::Long);

  auto status = invoker.Invoke("Cast", {
    std::move(input),
  }, output, &attrs);

  if (!status.IsOK())
    throw std::runtime_error(
    "ORT return failure status:" + status.ErrorMessage());
  return output[0];  
}

//#pragma endregion

//#pragma region Hand-Implemented ATen Ops

namespace aten {

at::Tensor empty_memory_format(
  at::IntArrayRef size,
  // *,
  c10::optional<at::ScalarType> dtype_opt,
  c10::optional<at::Layout> layout_opt,
  c10::optional<at::Device> device_opt,
  c10::optional<bool> pin_memory,
  c10::optional<at::MemoryFormat> memory_format) {
  ORT_LOG_FN(size, dtype_opt, layout_opt, device_opt, pin_memory, memory_format);

  assert(dtype_opt.has_value());
  assert(device_opt.has_value());

  // TODO: validate options and memory format
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  auto& invoker = GetORTInvoker(*device_opt);
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(*dtype_opt), onnxruntime::TensorShape(size.vec()),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ot);
  return aten_tensor_from_ort(
    std::move(ot),
    at::TensorOptions()
      .device(*device_opt)
      .dtype(*dtype_opt));
}

at::Tensor empty_strided(
  at::IntArrayRef size,
  at::IntArrayRef stride,
  // *
  c10::optional<at::ScalarType> dtype_opt,
  c10::optional<at::Layout> layout_opt,
  c10::optional<at::Device> device_opt,
  c10::optional<bool> pin_memory_opt) {
  ORT_LOG_FN(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // TODO: handle stride
  // TODO: how to handle type conversion
  OrtValue ot;
  assert(device_opt.has_value());
  // TODO: how to support layout
  //assert(!layout_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  auto& invoker = GetORTInvoker(*device_opt);
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(dtype), onnxruntime::TensorShape(size.vec()),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ot);
  return aten_tensor_from_ort(
    std::move(ot),
    at::TensorOptions()
      .device(*device_opt)
      .dtype(dtype));
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
at::Tensor as_strided(
  const at::Tensor& self, 
  at::IntArrayRef size, 
  at::IntArrayRef stride, 
  c10::optional<int64_t> storage_offset) {
  ORT_LOG_FN(self, size, stride, storage_offset);
  auto& invoker = GetORTInvoker(self.device());
  auto ort_input = create_ort_value(invoker, self);
  auto* tensor = ort_input.GetMutable<onnxruntime::Tensor>();

  auto byte_offset = storage_offset.has_value() ? (*storage_offset * tensor->DataType()->Size()) : 0;
  OrtValue ot;
  onnxruntime::Tensor::InitOrtValue(tensor->DataType(), onnxruntime::TensorShape(size.vec()), tensor->MutableDataRaw(),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault)->Info(),
                                    ot, byte_offset, stride.vec());
  return aten_tensor_from_ort(
    std::move(ot),
    self.options());
}

at::Tensor _reshape_alias(
  const at::Tensor& self, 
  at::IntArrayRef size, 
  at::IntArrayRef stride){
  ORT_LOG_FN(self, size, stride);
  // TODO: support stride
  auto& invoker = GetORTInvoker(self.device());
  auto ort_input = create_ort_value(invoker, self);
  return aten_tensor_from_ort(
    reshape_invoke(
      invoker,
      ort_input,
      size,
      // invoke reshape kernel inplace
      true),
    self.options());
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  ORT_LOG_FN(self, size);
  auto& invoker = GetORTInvoker(self.device());
  auto ort_input = create_ort_value(invoker, self);
  return aten_tensor_from_ort(
    reshape_invoke(
      invoker,
      ort_input,
      size,
      // invoke reshape kernel inplace
      true),
    self.options());
}

at::Tensor& copy_(
  at::Tensor& self,
  const at::Tensor& src,
  bool non_blocking) {
  ORT_LOG_FN(self, src, non_blocking);

  assert_tensor_supported(self);
  assert_tensor_supported(src);

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT
    ? self.device()
    : src.device());
  const auto ort_src = create_ort_value(invoker, src);
  auto ort_self = create_ort_value(invoker, self);
  if (self.scalar_type() != src.scalar_type()){
    // invoke cast first
    std::vector<OrtValue> ort_cast_output(1);
    onnxruntime::NodeAttributes attrs(1);
    attrs["to"] = create_ort_attribute(
      "to", (int64_t)GetONNXTensorProtoDataType(self.scalar_type()), at::kLong);

    auto status = invoker.Invoke("Cast", {
      std::move(ort_src),
    }, ort_cast_output, &attrs);
  
    if (!status.IsOK())
      throw std::runtime_error(
        "ORT return failure status:" + status.ErrorMessage());
    
    copy(invoker, ort_cast_output[0], ort_self);
  }
  else{
    copy(invoker, ort_src, ort_self);
  }
  
  return self;
}

at::Tensor _copy_from_and_resize(
  const at::Tensor& self, 
  const at::Tensor& dst){
  ORT_LOG_FN(self, dst);

  assert_tensor_supported(self);
  assert_tensor_supported(dst);

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT
    ? self.device()
    : dst.device());
  const auto ort_self = create_ort_value(invoker, self);
  auto ort_dst = create_ort_value(invoker, dst);

  copy(invoker, ort_self, ort_dst);

  return self;
}

at::Tensor& zero_(at::Tensor& self){
  auto& invoker = GetORTInvoker(self.device());
  auto ort_in_self = create_ort_value(invoker, self);
  OrtValue flag_val;
  //construct a constant tensor
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  onnxruntime::Tensor::InitOrtValue(element_type, onnxruntime::TensorShape({}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), flag_val);
  auto* ort_flag_tensor = flag_val.GetMutable<onnxruntime::Tensor>();
  int64_t one = 1;
  CopyVectorToTensor<int64_t>(invoker, &one, 1, *ort_flag_tensor);

  std::vector<OrtValue> ort_out = {ort_in_self};
  
  auto status = invoker.Invoke(
    "ZeroGradient", {
      std::move(ort_in_self),
      std::move(flag_val)
    }, ort_out, nullptr, onnxruntime::kMSDomain, 1);

  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());

  return self;
}

// TODO: enhance opgen.py to support inplace binary operations.
// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor& add__Tensor(
  at::Tensor& self, 
  const at::Tensor& other, 
  const at::Scalar& alpha) {
  ORT_LOG_FN(self, other, alpha);
  
  if (
    !IsSupportedType(alpha, {at::kDouble,at::kLong,at::kHalf,at::kShort,at::kInt,at::kByte,at::kFloat,at::kBFloat16}) || 
    !IsSupportedType(other, {at::kDouble,at::kLong,at::kHalf,at::kShort,at::kInt,at::kByte,at::kFloat,at::kBFloat16}) || 
    !IsSupportedType(self, {at::kDouble,at::kLong,at::kHalf,at::kShort,at::kInt,at::kByte,at::kFloat,at::kBFloat16})) {
    return at::native::call_fallback_fn<
      &at::native::cpu_fallback,
      ATEN_OP(add__Tensor)>::call(self, other, alpha);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_alpha = create_ort_value(invoker, alpha, other.scalar_type());
  auto ort_input_other = create_ort_value(invoker, other);
  
  std::vector<OrtValue> ort_outputs_0_Mul(1);
  
  auto status = invoker.Invoke("Mul", {
    std::move(ort_input_alpha),
    std::move(ort_input_other),
  }, ort_outputs_0_Mul, nullptr);
  
  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());
  
  auto ort_input_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_1_Add(1);
  ort_outputs_1_Add[0] = ort_input_self;
  
  status = invoker.Invoke("Add", {
    std::move(ort_input_self),
    std::move(ort_outputs_0_Mul[0]),
  }, ort_outputs_1_Add, nullptr);
  
  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());
  
  return self;
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
at::Tensor slice_Tensor(
  const at::Tensor& self, 
  int64_t dim, 
  c10::optional<int64_t> start, 
  c10::optional<int64_t> end, 
  int64_t step) {
  ORT_LOG_FN(self, dim, start, end, step);
  int64_t ndim = self.dim();
  if (ndim == 0) {
    throw std::runtime_error("slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);

  auto& invoker = GetORTInvoker(self.device());
  auto ort_input = create_ort_value(invoker, self);
  auto* ort_tensor = ort_input.GetMutable<onnxruntime::Tensor>();
  auto& shape = ort_tensor->Shape();
  auto strides = ort_tensor->Strides();
  int64_t l_start = start.has_value() ? *start : 0;
  int64_t l_end = end.has_value() ? *end : shape[dim];
  if (l_start < 0) {
    l_start += shape[dim];
  }
  if (l_end < 0) {
    l_end += shape[dim];
  }
  if (l_start < 0) {
    l_start = 0;
  } else if (l_start >= shape[dim]) {
    l_start = shape[dim];
  }
  if (l_end < l_start) {
    l_end = l_start;
  } else if (l_end >= shape[dim]) {
    l_end = shape[dim];
  }

  auto byte_offset = ort_tensor->ByteOffset() + (l_start * strides[dim]) * ort_tensor->DataType()->Size();
  auto len = l_end - l_start;
  onnxruntime::TensorShapeVector new_shape = shape.AsShapeVector();
  onnxruntime::TensorShapeVector new_stride(strides.begin(), strides.end());
  new_shape[dim] = (len + step - 1) / step;  // round-up
  new_stride[dim] *= step;

  OrtValue ot;
  onnxruntime::Tensor::InitOrtValue(
      ort_tensor->DataType(), onnxruntime::TensorShape(new_shape), ort_tensor->MutableDataRaw(),
      invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault)->Info(), ot, byte_offset, new_stride);
  return aten_tensor_from_ort(
    std::move(ot),
    self.options());
}

} // namespace aten

//#pragma endregion

} // namespace eager
} // namespace torch_ort
