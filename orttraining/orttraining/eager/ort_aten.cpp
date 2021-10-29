// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
#include "ort_tensor.h"

namespace torch_ort {
namespace eager {

//#pragma region Helpers

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
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar) {
  // TODO: support more types
  float val = scalar.toFloat();
  OrtValue ort_val;
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    ort_scalar_type_from_aten(at::kFloat),
    {},
    &ort_val);
  auto* ort_tensor = ort_val.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<float>(invoker, {val}, *ort_tensor);
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

  OrtValue ort_tensor;
  CreateMLValue(
    tensor.data_ptr(),
    ort_scalar_type_from_aten(tensor.scalar_type()),
    tensor.sizes().vec(),
    &ort_tensor);
  return ort_tensor;
}

OrtValue create_ort_value(const at::Tensor& tensor){
  auto& invoker = GetORTInvoker(tensor.device());
  return create_ort_value(invoker, tensor);
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

//#pragma endregion

//#pragma region Hand-Implemented ATen Ops

namespace aten {

at::Tensor empty__memory_format(
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
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    ort_scalar_type_from_aten(*dtype_opt),
    size.vec(),
    &ot);

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
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    ort_scalar_type_from_aten(dtype),
    size.vec(),
    &ot);
  return aten_tensor_from_ort(
    std::move(ot),
    at::TensorOptions()
      .device(*device_opt)
      .dtype(dtype));
}

at::Tensor reshape(at::Tensor const& self, at::IntArrayRef shape) {
  ORT_LOG_FN(self, shape);

  auto& invoker = GetORTInvoker(self.device());
  return aten_tensor_from_ort(
    reshape_copy(
      invoker,
      create_ort_value(invoker, self),
      shape.vec()),
    self.options());
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  ORT_LOG_FN(self, size);

  auto& invoker = GetORTInvoker(self.device());
  return aten_tensor_from_ort(
    reshape_copy(
      invoker,
      create_ort_value(invoker, self),
      at::infer_size(
        size,
        self.numel())),
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

  copy(invoker, ort_src, ort_self);

  return self;
}

at::Tensor& zero_(at::Tensor& self){
  auto& invoker = GetORTInvoker(self.device());
  auto ort_in_self = create_ort_value(invoker, self);
  OrtValue flag_val;
  //construct a constant tensor
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, {}, &flag_val);
  auto* ort_flag_tensor = flag_val.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>(invoker, {1}, *ort_flag_tensor);

  std::vector<OrtValue> ort_out(1);

  auto status = invoker.Invoke(
    "ZeroGradient", {
      std::move(ort_in_self),
      std::move(flag_val)
    }, ort_out, nullptr, onnxruntime::kMSDomain, 1);

  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());

  copy(invoker, ort_out[0], ort_in_self);
  return self;
}

} // namespace aten

//#pragma endregion

} // namespace eager
} // namespace torch_ort