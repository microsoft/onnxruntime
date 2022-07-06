// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
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
  at::Scalar value,
  const bool isTensor) {
  if (isTensor){
    onnx::AttributeProto attr;
    attr.set_name(name);
    at::ScalarType type = value.type();
    attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR);
    auto* constant_attribute_tensor_proto = attr.mutable_t();
    constant_attribute_tensor_proto->mutable_dims()->Clear();
    switch (type) {
    case at::ScalarType::Float:
      constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      *constant_attribute_tensor_proto->mutable_float_data()->Add() = value.to<float>();
      break;
    case at::ScalarType::Double:
      constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
      *constant_attribute_tensor_proto->mutable_float_data()->Add() = value.to<double>();
      break;
    case at::ScalarType::Bool:
    case at::ScalarType::Int:
      constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
      *constant_attribute_tensor_proto->mutable_float_data()->Add() = value.to<int>();
      break;
    case at::ScalarType::Long:
      constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      *constant_attribute_tensor_proto->mutable_float_data()->Add() = value.to<int64_t>();
      break;
    default:
      // For most at::ScalarType, it should be safe to just call value.to<>
      // on it, but for now we want to explicitly know when we've encountered
      // a new scalar type while bringing up ORT eager mode.
      ORT_THROW("Unsupported: at::ScalarType::", value.type());
    }
    return attr;
  }
  else{
    return create_ort_attribute(name, value, value.type());
  }
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

/*
 * Utility function for resizing output tensor
 * Only resizes if:
 *   - The shape is different
 *   - The output tensor is empty
 *
 * We do not support resizing non-empty output tensors.
 * PyToch implementation of resize will warn about resizing
 * non-empty and indicate this is deprecated behavior that
 * can / will change.
  *
 * In PyTorch repository see: aten/src/ATen/native/Resize.{h|cpp}
 */
void resize_output(
  onnxruntime::ORTInvoker& invoker,
  ORTTensorImpl* output,
  at::IntArrayRef shape) {
  if (output->sizes().equals(shape)) {
    return;
  }

  if (output->numel() != 0) {
    throw std::runtime_error(
      "resizing a non-empty output tensor is not supported.");
  }

  resize_impl_ort_(invoker, output, shape);
}

//#pragma endregion

/*
 * Resize backing store of a TensorImpl.
 *
 * See notes for implementation details and potential differences from canonical implementations due to constraints in
 * ORT model.
 *
 * If new size is the same size as existing tensor: reshape the existing tensor
 * If new size is larger:  allocate new memory and copy over existing elements. New memory is uninitialized.
 * If new size is smaller: allocate a smaller backing tensor, and copy over
 *                         as many elements as will fit.
 *
 * Notes:
 * There are some implementation details that might deviate from expectations:
 *  - As the Onnxruntime::tensor does not support resize operation, this functionality is supported on the TensorImpl
 *    by swapping out the backing tensor if the size changes.
 *
 *  - In the ORT model the shape of the TensorImpl is defined by the backing onnxruntime::tensor, so it is not supported
 *    to have a TensorImpl with a different shape / size than the backing onnxruntime::tensor. This means when resizing
 *    to a smaller TensorImpl, other implementations might keep the same backing storage, ORT will re-allocate a new
 *    onnxruntime::tensor and copy over as many of the existing elements that fit. Functionally, you will end up with
 *    same output, but the underlying buffer will be re-allocated.
 *
 *    A future change could be to allow ORTTensorImpl to have a different size / shape than the onnxrutime::tensor
 *    backing it, and then we could improve this behavior.
 *
 * The canonical CPU / CUDA implementations in PyTorch repository:
 *     CPU: aten/src/ATen/native/Resize.cpp
 *     CUDA: aten/src/ATen/native/cuda/Resize.cpp
 */
void resize_impl_ort_(
    onnxruntime::ORTInvoker& invoker,
    ORTTensorImpl* self,
    at::IntArrayRef size) {
  auto self_ort_value = self->tensor();

  // If shape and size are the same, then nothing to do
  if (self->sizes() == size) {
    return;
  }

  auto old_shape = onnxruntime::TensorShape(self->sizes());
  auto new_shape = onnxruntime::TensorShape(size);

  if (new_shape.Size() == old_shape.Size()) {
    // Requested size is the same, only shape is different.
    // Just resize existing tensor and return

    OrtValue new_ort_value = reshape_invoke(
              invoker,
              self_ort_value,
              size,
              // invoke reshape kernel inplace
              true);

    // TODO(jamill): Investigate why reshape_invoke kernel does not update inplace
    self->set_tensor(new_ort_value);
  } else {
    // Requested size is different - allocate a new onnxruntime::tensor and update ORTTensorImpl
    // with new backing onnxruntime::tensor.
    auto* self_ort_tensor = self_ort_value.GetMutable<onnxruntime::Tensor>();

    OrtValue new_ort_value;
    onnxruntime::Tensor::InitOrtValue(self_ort_tensor->DataType(), new_shape,
                                      invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                                      new_ort_value);

    auto* new_ort_tensor = new_ort_value.GetMutable<onnxruntime::Tensor>();

    // Copy over existing elements from current tensor as appropriate
    if (self_ort_tensor->SizeInBytes() == 0) {
      // self is empty, nothing to copy over
    } else if (new_ort_tensor->SizeInBytes() > self_ort_tensor->SizeInBytes()) {
      // Copy elements from (smaller) old tensor to (larger) new self tensor

      // See function comments to see details on why we need to create temporary ORTValue here
      // (Copying elements between tensors of different sizes is not supported)
      OrtValue tmp;
      onnxruntime::Tensor::InitOrtValue(new_ort_tensor->DataType(), old_shape,
                                        new_ort_tensor->MutableDataRaw(),
                                        new_ort_tensor->Location(),
                                        tmp);

      copy(invoker, self_ort_value, tmp);
    } else if (new_ort_tensor->SizeInBytes() < self_ort_tensor->SizeInBytes()) {
      // Copy elements from (larger) initial self tensor to (smaller) updated self tensor

      // See function comments to see details on why we need to create temporary ORTValue here
      // (Copying elements between tensors of different sizes is not supported)
      OrtValue tmp;
      onnxruntime::Tensor::InitOrtValue(self_ort_tensor->DataType(), new_shape,
                                        self_ort_tensor->MutableDataRaw(),
                                        self_ort_tensor->Location(),
                                        tmp);

      copy(invoker, tmp, new_ort_value);
    }

    self->set_tensor(new_ort_value);
  }

  return;
}

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

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                         c10::optional<bool> pin_memory_opt) {
  ORT_LOG_FN(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // TODO: how to handle type conversion
  OrtValue ot;
  assert(device_opt.has_value());
  // TODO: how to support layout
  // assert(!layout_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  auto& invoker = GetORTInvoker(*device_opt);
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(dtype), onnxruntime::TensorShape(size.vec()),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ot,
                                    stride.vec());
  return aten_tensor_from_ort(std::move(ot), at::TensorOptions().device(*device_opt).dtype(dtype));
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
    !IsSupportedType(alpha, {at::kDouble, at::kLong, at::kHalf, at::kShort, at::kInt, at::kByte, at::kFloat, at::kBFloat16}) ||
    !IsSupportedType(other, {at::kDouble, at::kLong, at::kHalf, at::kShort, at::kInt, at::kByte, at::kFloat, at::kBFloat16}) ||
    !IsSupportedType(self, {at::kDouble, at::kLong, at::kHalf, at::kShort, at::kInt, at::kByte, at::kFloat, at::kBFloat16})) {
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

// aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& argmax_out(
const at::Tensor& self,
c10::optional<int64_t> dim,
bool keepdim,
// *,
at::Tensor& out) {
  ORT_LOG_FN(self, dim, keepdim, out);

  if (
    !IsSupportedType(self, {at::kLong, at::kShort, at::kHalf, at::kBFloat16, at::kFloat, at::kByte, at::kInt, at::kDouble})) {
    return at::native::call_fallback_fn<
    &at::native::cpu_fallback,
    ATEN_OP(argmax_out)>::call(self, dim, keepdim, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_self =
    create_ort_value(invoker, dim.has_value() ? self : self.reshape({-1}));

  // Remove this hand signature once the generator can support this one line below.
  int64_t l_axis = dim.has_value() ? *dim : 0;

  NodeAttributes attrs(2);
  attrs["axis"] = create_ort_attribute(
  "axis", l_axis, at::ScalarType::Int);
  attrs["keepdims"] = create_ort_attribute(
  "keepdims", keepdim, at::ScalarType::Int);

  std::vector<OrtValue> ort_outputs_0_ArgMax(1);

  auto status = invoker.Invoke("ArgMax", {
  std::move(ort_input_self),
  }, ort_outputs_0_ArgMax, &attrs);

  if (!status.IsOK())
  throw std::runtime_error(
  "ORT return failure status:" + status.ErrorMessage());

  at::TensorOptions tensor_options = out.options();

  // generator also needs to do this to handle the out param!
  out = aten_tensor_from_ort(
  std::move(ort_outputs_0_ArgMax[0]),
  tensor_options);
  return out;
}

// aten::equal(Tensor self, Tensor other) -> bool
bool equal(
  const at::Tensor& self,
  const at::Tensor& other) {
  ORT_LOG_FN(self, other);

  if (
    std::vector<at::ScalarType> supportedTypes =
      {at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kLong, at::kByte, at::kInt, at::kShort, at::kBool};
    !IsSupportedType(self, supportedTypes) ||
    !IsSupportedType(other, supportedTypes)) {
    return at::native::call_fallback_fn<
      &at::native::cpu_fallback,
      ATEN_OP(equal)>::call(self, other);
  }

  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_self = create_ort_value(invoker, self);
  auto ort_input_other = create_ort_value(invoker, other);

  auto& ort_tensor_self = ort_input_self.Get<onnxruntime::Tensor>();
  auto& shape_self = ort_tensor_self.Shape();
  auto& ort_tensor_other = ort_input_other.Get<onnxruntime::Tensor>();
  auto& shape_other = ort_tensor_other.Shape();

  // ensure shape is equal
  if (shape_self != shape_other) return false;

  // to check content, we'll do elementwise comparison
  // then we'll reduce to the mininum value based on false
  // being less than true, so any false will reduce to false.
  std::vector<OrtValue> ort_outputs_0_Equal(1);

  auto equalStatus = invoker.Invoke("Equal", {
    std::move(ort_input_self),
    std::move(ort_input_other),
  }, ort_outputs_0_Equal, nullptr);

  if (!equalStatus.IsOK())
    throw std::runtime_error(
      "ORT Equal return failure status:" + equalStatus.ErrorMessage());

  // now reduce the resulting tensor of bool values to its minimum value (any false)
  NodeAttributes attrs(1);
  attrs["keepdims"] = create_ort_attribute(
    "keepdims", 0, at::ScalarType::Int);

  std::vector<OrtValue> ort_outputs_0_ReduceMin(1);

  // ReduceMin does not support bool or short and CastToType does not support Byte because
  // GetONNXTensorProtoDataType doesn't support byte, which leaves us with int
  OrtValue equalAsInt = CastToType(invoker, ort_outputs_0_Equal[0], at::ScalarType::Int);

  auto reduceStatus = invoker.Invoke("ReduceMin", {
    std::move(equalAsInt),
  }, ort_outputs_0_ReduceMin, &attrs);

  if (!reduceStatus.IsOK())
    throw std::runtime_error(
      "ORT ReduceMin return failure reduceStatus:" + reduceStatus.ErrorMessage());

  auto* ort_tensor = ort_outputs_0_ReduceMin[0].GetMutable<onnxruntime::Tensor>();
  // the first (and only) value of the tensor will be 0 for false else true
  return *(ort_tensor->Data<int>()) != 0;
}

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const at::Tensor& resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    c10::optional<at::MemoryFormat> optional_memory_format) {
  ORT_LOG_FN(self, size, optional_memory_format);
  assert_tensor_supported(self);

  // If self is already desired size, then return early
  if (self.sizes() == size) {
    return self;
  }

  auto& invoker = GetORTInvoker(self.device());
  resize_impl_ort_(
      invoker,
      dynamic_cast<ORTTensorImpl*>(self.unsafeGetTensorImpl()),
      size);
  return self;
}

} // namespace aten

//#pragma endregion

} // namespace eager
} // namespace torch_ort
