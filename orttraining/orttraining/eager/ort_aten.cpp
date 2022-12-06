// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"

#include <ATen/InferSize.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace torch_ort {
namespace eager {

// #pragma region Helpers
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
}  // namespace

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
  switch (dtype) {
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
  return create_ort_value(invoker, scalar, scalar.type());
}

OrtValue create_ort_value(
    onnxruntime::ORTInvoker& invoker,
    const at::Scalar& scalar,
    at::ScalarType type) {
  OrtValue ort_val;
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(type), onnxruntime::TensorShape({}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ort_val);
  auto* ort_tensor = ort_val.GetMutable<onnxruntime::Tensor>();
  switch (type) {
    case at::ScalarType::Float: {
      float val = scalar.toFloat();
      CopyVectorToTensor<float>(invoker, &val, 1, *ort_tensor);
      break;
    }
    case at::ScalarType::BFloat16: {
      at::BFloat16 valBFloat16 = scalar.toBFloat16();
      Ort::BFloat16_t* valOrtBFloat16 = reinterpret_cast<Ort::BFloat16_t*>(&valBFloat16);
      CopyVectorToTensor<Ort::BFloat16_t>(invoker, valOrtBFloat16, 1, *ort_tensor);
      break;
    }
    case at::ScalarType::Double: {
      double val = scalar.toDouble();
      CopyVectorToTensor<double>(invoker, &val, 1, *ort_tensor);
      break;
    }
    case at::ScalarType::Long: {
      int64_t val = scalar.toLong();
      CopyVectorToTensor<int64_t>(invoker, &val, 1, *ort_tensor);
      break;
    }
    default:
      // TODO(unknown): support more types
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

  OrtMemoryInfo* mem_info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
  auto element_type = ort_scalar_type_from_aten(tensor.scalar_type());

  OrtValue ort_tensor;
  onnxruntime::Tensor::InitOrtValue(
      element_type,
      onnxruntime::TensorShape(tensor.sizes().vec()),
      tensor.data_ptr(),
      *mem_info, ort_tensor,
      0L,  // offset = 0 - because tensor.data_ptr() includes the underlying offset
      tensor.strides().vec());
  return ort_tensor;
}

OrtValue create_ort_value(const at::Tensor& tensor) {
  auto& invoker = GetORTInvoker(tensor.device());
  return create_ort_value(invoker, tensor);
}

std::vector<OrtValue> create_ort_value(
    onnxruntime::ORTInvoker& invoker,
    at::TensorList values) {
  auto output = std::vector<OrtValue>{};
  for (auto element : values) {
    output.push_back(create_ort_value(element));
  }
  return output;
}

onnx::AttributeProto create_ort_attribute(
    const char* name,
    at::Scalar value,
    const bool isTensor,
    at::ScalarType type) {
  if (isTensor) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR);
    auto* constant_attribute_tensor_proto = attr.mutable_t();
    constant_attribute_tensor_proto->mutable_dims()->Clear();
    // Creating a 1 dim tensor of size 1, so add that dim now.
    constant_attribute_tensor_proto->add_dims(1);
    switch (type) {
      case at::ScalarType::Float:
        constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        *constant_attribute_tensor_proto->mutable_float_data()->Add() = value.to<float>();
        break;
      case at::ScalarType::Double:
        constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
        *constant_attribute_tensor_proto->mutable_double_data()->Add() = value.to<double>();
        break;
      case at::ScalarType::Bool:
      case at::ScalarType::Int:
        constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
        *constant_attribute_tensor_proto->mutable_int32_data()->Add() = value.to<int>();
        break;
      case at::ScalarType::Long:
        constant_attribute_tensor_proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        *constant_attribute_tensor_proto->mutable_int64_data()->Add() = value.to<int64_t>();
        break;
      default:
        // For most at::ScalarType, it should be safe to just call value.to<>
        // on it, but for now we want to explicitly know when we've encountered
        // a new scalar type while bringing up ORT eager mode.
        ORT_THROW("Unsupported: at::ScalarType::", value.type());
    }
    return attr;
  } else {
    return create_ort_attribute(name, value, value.type());
  }
}

onnx::AttributeProto create_ort_attribute(
    const char* name,
    at::Scalar value,
    const bool isTensor) {
  return create_ort_attribute(name, value, isTensor, value.type());
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

onnx::AttributeProto create_ort_attribute(
    const char* name,
    const std::vector<int64_t> values) {
  onnx::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

  for (size_t i = 0; i < values.size(); i++)
    attr.add_ints(values[i]);

  return attr;
}

bool IsSupportedType(at::Scalar scalar, const std::vector<at::ScalarType>& valid_types) {
  return std::find(valid_types.begin(), valid_types.end(), scalar.type()) != valid_types.end();
}

bool IsSupportedType(at::Tensor tensor, const std::vector<at::ScalarType>& valid_types) {
  return std::find(valid_types.begin(), valid_types.end(), tensor.scalar_type()) != valid_types.end();
}

bool IsSupportedType(at::IntArrayRef arrary, const std::vector<at::ScalarType>& valid_types) {
  return std::find(valid_types.begin(), valid_types.end(), at::kInt) != valid_types.end() ||
         std::find(valid_types.begin(), valid_types.end(), at::kLong) != valid_types.end();
}

bool IsSupportedType(int64_t val, const std::vector<at::ScalarType>& valid_types) {
  return std::find(valid_types.begin(), valid_types.end(), at::kLong) != valid_types.end();
}

bool IsSupportedType(c10::optional<int64_t> val, const std::vector<at::ScalarType>& valid_types) {
  return IsSupportedType(val.value(), valid_types);
}

bool IsSupportedType(at::TensorList tensors, const std::vector<at::ScalarType>& valid_types) {
  return IsSupportedType(tensors[0], valid_types);
}

ONNX_NAMESPACE::TensorProto_DataType GetONNXTensorProtoDataType(at::ScalarType dtype) {
  switch (dtype) {
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

OrtValue CastToType(onnxruntime::ORTInvoker& invoker, const OrtValue& input, at::ScalarType type) {
  std::vector<OrtValue> output(1);
  NodeAttributes attrs(1);
  attrs["to"] = create_ort_attribute(
      "to", GetONNXTensorProtoDataType(type), at::ScalarType::Long);

  auto status = invoker.Invoke("Cast",
                               {std::move(input)},
                               output, &attrs);

  CHECK_STATUS(status);
  return output[0];
}

void CastToType_out(onnxruntime::ORTInvoker& invoker, const OrtValue& input, OrtValue& output, at::ScalarType type) {
  std::vector<OrtValue> output_result(1);
  output_result[0] = output;
  NodeAttributes attrs(1);
  attrs["to"] = create_ort_attribute(
      "to", GetONNXTensorProtoDataType(type), at::ScalarType::Long);

  auto status = invoker.Invoke("Cast",
                               {std::move(input)},
                               output_result, &attrs);

  CHECK_STATUS(status);
}

/*
 * Utility method to calculate the resulting shape of tensor after a reduction operation.
 *
 * @param dimToReduce The dimension to reduce. If null, then shape is 0 dimension.
 * @param keepdim Whether to retain dim or not. Ignored if dimToReduce is null.
 */
inline at::DimVector calculate_reduction_shape(
    const at::Tensor& self,
    c10::optional<int64_t> dimToReduce,
    bool keepdim) {
  at::DimVector shape;

  // If we have dim value, then reduce that dimension.
  // else, return empty shape (corresponding to 0-D tensor)
  if (dimToReduce.has_value()) {
    shape = at::DimVector(self.sizes());
    int64_t effectiveDimToReduce = *dimToReduce;
    at::maybe_wrap_dims_n(&effectiveDimToReduce, 1, self.dim());

    if (keepdim) {
      shape[effectiveDimToReduce] = 1;
    } else {
      shape.erase(shape.begin() + effectiveDimToReduce);
    }
  } else {
    shape = at::DimVector();
  }

  return shape;
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

// #pragma endregion

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

// The following was copied from onnxruntime/core/providers/cuda/math/binary_elementwise_ops.cc
// This method computes the resulting shape from broadcasting.
Status ComputeOutputShape(
    const std::string& node_name,
    const onnxruntime::TensorShape& lhs_shape,
    const onnxruntime::TensorShape& rhs_shape,
    onnxruntime::TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = onnxruntime::TensorShape(output_dims);
  return Status::OK();
}

// Note in the below the user needs to pass in out_shape so they own the memory as IntArrayRef is just a view into it.
at::IntArrayRef BroadcastShape(
    const std::string& node_name,
    OrtValue lhs,
    OrtValue rhs,
    onnxruntime::TensorShape& out_shape) {
  auto& ort_tensor_lhs = lhs.Get<onnxruntime::Tensor>();
  auto& ort_tensor_rhs = rhs.Get<onnxruntime::Tensor>();
  auto status = ComputeOutputShape(node_name, ort_tensor_lhs.Shape(), ort_tensor_rhs.Shape(), out_shape);
  CHECK_STATUS(status);
  auto out_shape_dims = out_shape.GetDims();
  return !out_shape_dims.empty() ? at::IntArrayRef(out_shape_dims.data(), out_shape_dims.size())
                                 : at::IntArrayRef();
}

// #pragma region Hand-Implemented ATen Ops

namespace aten {

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,  // Ignored because there's no ONNX support.
    c10::optional<at::Device> device_opt,  // Will be ORT by the time this is dispatched.
    c10::optional<bool> pin_memory_opt) {  // Ignored because there's no ONNX support.
  ORT_LOG_FN(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  OrtValue ot;
  assert(device_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  auto& invoker = GetORTInvoker(*device_opt);
  onnxruntime::Tensor::InitOrtValue(ort_scalar_type_from_aten(dtype), onnxruntime::TensorShape(size.vec()),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), ot,
                                    stride.vec());
  return aten_tensor_from_ort(
      std::move(ot),
      at::TensorOptions()
          .device(*device_opt)
          .dtype(dtype));
}

at::Tensor empty_memory_format(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {  // Ignored because there's no ONNX support.
  ORT_LOG_FN(size, dtype_opt, layout_opt, device_opt, pin_memory, memory_format);

  // Use the strided impl with default (no strides specified).
  return empty_strided(size, at::IntArrayRef({}), dtype_opt, layout_opt, device_opt, pin_memory);
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
    at::IntArrayRef stride) {
  ORT_LOG_FN(self, size, stride);
  // TODO(unknown): support stride
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

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT ? self.device() : src.device());
  const auto ort_src = create_ort_value(invoker, src);
  auto ort_self = create_ort_value(invoker, self);

  if (self.scalar_type() != src.scalar_type()) {
    if (src.device().type() != at::kORT) {
      // invoke cast first and then copy for non-ORT device types
      auto val = at::native::to(src, self.scalar_type());
      const auto ort_val = create_ort_value(invoker, val);
      copy(invoker, ort_val, ort_self);
    } else {
      // For ORT device type, the cast operation will perform the copy as well
      std::vector<OrtValue> ort_cast_output(1);
      ort_cast_output[0] = ort_self;
      onnxruntime::NodeAttributes attrs(1);
      attrs["to"] = create_ort_attribute(
          "to", (int64_t)GetONNXTensorProtoDataType(self.scalar_type()), at::kLong);

      auto status = invoker.Invoke("Cast",
                                   {std::move(ort_src)},
                                   ort_cast_output, &attrs);

      CHECK_STATUS(status);
    }
  } else {
    copy(invoker, ort_src, ort_self);
  }

  return self;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  ORT_LOG_FN(self, dst);

  assert_tensor_supported(self);
  assert_tensor_supported(dst);

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT ? self.device() : dst.device());
  const auto ort_self = create_ort_value(invoker, self);
  auto ort_dst = create_ort_value(invoker, dst);

  copy(invoker, ort_self, ort_dst);

  return self;
}

at::Tensor& zero_(at::Tensor& self) {
  auto& invoker = GetORTInvoker(self.device());
  auto ort_in_self = create_ort_value(invoker, self);
  OrtValue flag_val;
  // construct a constant tensor
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  onnxruntime::Tensor::InitOrtValue(element_type, onnxruntime::TensorShape({}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), flag_val);
  auto* ort_flag_tensor = flag_val.GetMutable<onnxruntime::Tensor>();
  int64_t one = 1;
  CopyVectorToTensor<int64_t>(invoker, &one, 1, *ort_flag_tensor);

  std::vector<OrtValue> ort_out = {ort_in_self};

  auto status = invoker.Invoke("ZeroGradient",
                               {std::move(ort_in_self), std::move(flag_val)},
                               ort_out, nullptr, onnxruntime::kMSDomain, 1);

  CHECK_STATUS(status);

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

  auto st = {at::kDouble, at::kLong, at::kHalf, at::kShort, at::kInt, at::kByte, at::kFloat, at::kBFloat16};
  if (
      !IsSupportedType(self, st)) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(argmax_out)>::call(self, dim, keepdim, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_self =
      create_ort_value(invoker, dim.has_value() ? self : self.reshape({-1}));

  int64_t l_axis = dim.has_value() ? *dim : 0;
  bool keepdim_effective_value = dim.has_value() ? keepdim : false;

  NodeAttributes attrs(2);
  attrs["axis"] = create_ort_attribute(
      "axis", l_axis, at::ScalarType::Int);
  attrs["keepdims"] = create_ort_attribute(
      "keepdims", keepdim_effective_value, at::ScalarType::Bool);

  std::vector<OrtValue> ort_outputs_0_ArgMax(1);

  // Calculate the size of the out tensor, based on self tensor, dimension input, and keepdim input
  auto shape = calculate_reduction_shape(self, dim, keepdim);

  resize_output(invoker,
                dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()),
                at::IntArrayRef{shape});

  auto ort_input_out = create_ort_value(invoker, out);
  ort_outputs_0_ArgMax[0] = ort_input_out;

  auto status = invoker.Invoke("ArgMax",
                               {std::move(ort_input_self)},
                               ort_outputs_0_ArgMax, &attrs);

  CHECK_STATUS(status);

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

  auto equalStatus = invoker.Invoke("Equal",
                                    {std::move(ort_input_self), std::move(ort_input_other)},
                                    ort_outputs_0_Equal, nullptr);

  CHECK_STATUS(equalStatus);

  // now reduce the resulting tensor of bool values to its minimum value (any false)
  NodeAttributes attrs(1);
  attrs["keepdims"] = create_ort_attribute(
      "keepdims", 0, at::ScalarType::Int);

  std::vector<OrtValue> ort_outputs_0_ReduceMin(1);

  // ReduceMin does not support bool or short and CastToType does not support Byte because
  // GetONNXTensorProtoDataType doesn't support byte, which leaves us with int
  OrtValue equalAsInt = CastToType(invoker, ort_outputs_0_Equal[0], at::ScalarType::Int);

  auto reduceStatus = invoker.Invoke("ReduceMin",
                                     {std::move(equalAsInt)},
                                     ort_outputs_0_ReduceMin, &attrs);

  CHECK_STATUS(reduceStatus);

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

// aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& cat_out(
    at::TensorList tensors,
    int64_t dim,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(tensors, dim, out);

  assert(tensors.size() > 0);
  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kBool, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(tensors, supportedTypes)) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(cat_out)>::call(tensors, dim, out);
  }
  int64_t ndim = tensors[0].dim();
  assert(ndim != 0);
  dim = at::maybe_wrap_dim(dim, ndim);

  auto& invoker = GetORTInvoker(tensors[0].device());

  // IntArrayRef isn't writeable, convert to vector.
  std::vector<int64_t> sizes;
  for (auto s : tensors[0].sizes())
    sizes.push_back(s);

  // Calculate the new size of the dimension being concatenated.
  sizes[dim] = 0;
  for (auto t : tensors)
    sizes[dim] += t.size(dim);

  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), at::IntArrayRef(sizes));
  auto ort_input_out = create_ort_value(invoker, out);

  auto ort_input_0_tensors = create_ort_value(invoker, tensors);

  NodeAttributes attrs_0(1);
  attrs_0["axis"] = create_ort_attribute(
      "axis", dim, at::ScalarType::Int);

  std::vector<OrtValue> ort_outputs_0_Concat(1);
  ort_outputs_0_Concat[0] = ort_input_out;

  auto status = invoker.Invoke("Concat",
                               {std::move(ort_input_0_tensors)},
                               ort_outputs_0_Concat, &attrs_0);
  CHECK_STATUS(status);

  return out;
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor& fill__Scalar(
    at::Tensor& self,
    const at::Scalar& value) {
  ORT_LOG_FN(self, value);

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kHalf, at::kFloat, at::kInt, at::kDouble, at::kByte, at::kShort, at::kLong, at::kBFloat16, at::kBool};
      !IsSupportedType(self, supportedTypes)) {
    std::cout << "fill__Scalar - Fell back to cpu!\n";
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(fill__Scalar)>::call(self, value);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_self = create_ort_value(invoker, self);

  std::vector<OrtValue> ort_outputs_0_Shape(1);

  auto status = invoker.Invoke("Shape",
                               {std::move(ort_input_self)},
                               ort_outputs_0_Shape, nullptr);

  CHECK_STATUS(status);

  std::vector<OrtValue> ort_outputs_1_ConstantOfShape(1);
  ort_outputs_1_ConstantOfShape[0] = ort_input_self;

  NodeAttributes attrs(1);
  attrs["value"] = create_ort_attribute(
      "value", value, true, self.scalar_type());

  status = invoker.Invoke("ConstantOfShape",
                          {std::move(ort_outputs_0_Shape[0])},
                          ort_outputs_1_ConstantOfShape, &attrs);

  CHECK_STATUS(status);

  return self;
}

// aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& nonzero_out(
    const at::Tensor& self,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(self, out);

  auto temp = eager::aten::nonzero(self);

  // resize out, then copy nonzero result into it.
  auto& invoker = GetORTInvoker(self.device());
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), temp.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  auto ort_temp = create_ort_value(invoker, temp);
  copy(invoker, ort_temp, ort_input_out);

  return out;
}

// aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& _log_softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(self, dim, half_to_float, out);

  if (
      !IsSupportedType(self, {at::kBFloat16, at::kDouble, at::kFloat, at::kHalf})) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(_log_softmax_out)>::call(self, dim, half_to_float, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  auto ort_input_0_self = create_ort_value(invoker, self);

  // Check dimensions (according to symbolic_opset9).
  // Onnx only supports log_softmax with dim -1, otherwise transpose required.
  int64_t ndim = self.dim();
  if (dim < 0) {
    dim += ndim;
  }
  bool need_transpose = ndim != dim + 1;

  // Use transpose to switch the needed dimension to -1
  // This requires specifying all of the dimensions in order and then
  // swapping the last one with the one specified.
  std::vector<int64_t> axes;
  std::vector<OrtValue> ort_outputs_0_Transpose(1);
  if (need_transpose) {
    axes.reserve(ndim);
    for (int64_t i = 0; i < ndim; i++)
      axes.push_back(i);

    axes[dim] = ndim - 1;
    axes[ndim - 1] = dim;
    dim = ndim - 1;

    NodeAttributes attrs_0(1);
    attrs_0["perm"] = create_ort_attribute("perm", axes);
    auto status = invoker.Invoke("Transpose",
                                 {std::move(ort_input_0_self)},
                                 ort_outputs_0_Transpose, &attrs_0);
    CHECK_STATUS(status);
  }

  NodeAttributes attrs_1(1);
  attrs_1["axis"] = create_ort_attribute(
      "axis", dim, at::ScalarType::Int);

  std::vector<OrtValue> ort_outputs_1_LogSoftmax(1);
  if (!need_transpose) {
    ort_outputs_1_LogSoftmax[0] = ort_input_out;
  }

  auto status = invoker.Invoke("LogSoftmax",
                               {std::move(need_transpose ? ort_outputs_0_Transpose[0] : ort_input_0_self)},
                               ort_outputs_1_LogSoftmax, &attrs_1);
  CHECK_STATUS(status);

  std::vector<OrtValue> ort_outputs_2_Transpose(1);

  if (need_transpose) {
    ort_outputs_2_Transpose[0] = ort_input_out;

    NodeAttributes attrs_2(1);
    attrs_2["perm"] = create_ort_attribute("perm", axes);

    status = invoker.Invoke("Transpose",
                            {std::move(ort_outputs_1_LogSoftmax[0])},
                            ort_outputs_2_Transpose, &attrs_2);
    CHECK_STATUS(status);
  }

  return out;
}

// aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
// mm is for matrix multiplication and does not broadcast.
// https://pytorch.org/docs/stable/generated/torch.mm.html
at::Tensor& mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(self, mat2, out);

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kDouble, at::kLong, at::kHalf, at::kFloat, at::kBFloat16, at::kInt};
      !IsSupportedType(self, supportedTypes) ||
      !IsSupportedType(mat2, supportedTypes) ||
      // to match cpu device behavior for torch.mm, verify the following and fall back to cpu to generate error message.
      // 1. self and mat2 must be 2-D (matrices)
      self.dim() != 2 || mat2.dim() != 2 ||
      // 2. self and mat2 can be multiplied
      self.sizes()[1] != mat2.sizes()[0] ||
      // 3. self, mat2, and out are of the same type
      self.scalar_type() != out.scalar_type() ||
      self.scalar_type() != mat2.scalar_type()) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(mm_out)>::call(self, mat2, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(), mat2.scalar_type()}, {});

  // resize the output and then create output ort value to be updated.
  // out size is first dimension of self and 2nd dimension of mat2
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), {self.sizes()[0], mat2.sizes()[1]});
  auto ort_input_out = create_ort_value(invoker, out);

  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type) {
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_mat2 = create_ort_value(invoker, mat2);
  if (mat2.scalar_type() != *promoted_type) {
    ort_input_0_mat2 = CastToType(invoker, ort_input_0_mat2, *promoted_type);
  }

  std::vector<OrtValue> ort_outputs_0_MatMul(1);
  ort_outputs_0_MatMul[0] = ort_input_out;

  auto status = invoker.Invoke("MatMul",
                               {std::move(ort_input_0_self), std::move(ort_input_0_mat2)},
                               ort_outputs_0_MatMul, nullptr);
  CHECK_STATUS(status);

  return out;
}

// aten::squeeze(Tensor(a) self) -> Tensor(a)
at::Tensor squeeze(
    const at::Tensor& self) {
  ORT_LOG_FN(self);

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kBool, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(self, supportedTypes))
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(squeeze)>::call(self);

  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_0_self = create_ort_value(invoker, self);

  std::vector<OrtValue> ort_outputs_0_Squeeze(1);

  auto status = invoker.Invoke("Squeeze",
                               {std::move(ort_input_0_self)},
                               ort_outputs_0_Squeeze, nullptr);
  CHECK_STATUS(status);

  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
      std::move(ort_outputs_0_Squeeze[0]),
      tensor_options);
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    // *,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ORT_LOG_FN(self, other, alpha, out);

  auto promoted_type = PromoteScalarTypesWithCategory({other.scalar_type(), self.scalar_type()}, {alpha.type()});

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(alpha, supportedTypes) ||
      !IsSupportedType(other, supportedTypes) ||
      !IsSupportedType(self, supportedTypes) ||
      !c10::canCast(*promoted_type, out.scalar_type())) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(add_out)>::call(self, other, alpha, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_0_alpha = create_ort_value(invoker, alpha);
  if (alpha.type() != *promoted_type) {
    ort_input_0_alpha = CastToType(invoker, ort_input_0_alpha, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type) {
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }

  std::vector<OrtValue> ort_outputs_0_Mul(1);

  auto status = invoker.Invoke("Mul",
                               {std::move(ort_input_0_alpha), std::move(ort_input_0_other)},
                               ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);

  auto ort_input_1_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type) {
    ort_input_1_self = CastToType(invoker, ort_input_1_self, *promoted_type);
  }

  // resize the output and then create output ort value to be updated.
  onnxruntime::TensorShape out_shape;
  resize_output(
      invoker,
      dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()),
      BroadcastShape(__func__, ort_input_1_self, ort_outputs_0_Mul[0], out_shape));

  auto ort_input_out = create_ort_value(invoker, out);

  std::vector<OrtValue> ort_outputs_1_Add(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_1_Add[0] = ort_input_out;
  }

  status = invoker.Invoke("Add",
                          {std::move(ort_input_1_self), std::move(ort_outputs_0_Mul[0])},
                          ort_outputs_1_Add, nullptr);
  CHECK_STATUS(status);

  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_1_Add[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor& sub_out(
    const at::Tensor& self,
    const at::Tensor& other,
    // *,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ORT_LOG_FN(self, other, alpha, out);

  auto promoted_type = PromoteScalarTypesWithCategory({other.scalar_type(), self.scalar_type()}, {alpha.type()});

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(alpha, supportedTypes) ||
      !IsSupportedType(other, supportedTypes) ||
      !IsSupportedType(self, supportedTypes) ||
      !c10::canCast(*promoted_type, out.scalar_type())) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(sub_out)>::call(self, other, alpha, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_0_alpha = create_ort_value(invoker, alpha);
  if (alpha.type() != *promoted_type) {
    ort_input_0_alpha = CastToType(invoker, ort_input_0_alpha, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type) {
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }

  std::vector<OrtValue> ort_outputs_0_Mul(1);

  auto status = invoker.Invoke("Mul",
                               {std::move(ort_input_0_alpha), std::move(ort_input_0_other)},
                               ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);

  auto ort_input_1_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type) {
    ort_input_1_self = CastToType(invoker, ort_input_1_self, *promoted_type);
  }

  // resize the output and then create output ort value to be updated.
  onnxruntime::TensorShape out_shape;
  resize_output(
      invoker,
      dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()),
      BroadcastShape(__func__, ort_input_1_self, ort_outputs_0_Mul[0], out_shape));

  auto ort_input_out = create_ort_value(invoker, out);

  std::vector<OrtValue> ort_outputs_1_Sub(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_1_Sub[0] = ort_input_out;
  }

  status = invoker.Invoke("Sub",
                          {std::move(ort_input_1_self), std::move(ort_outputs_0_Mul[0])},
                          ort_outputs_1_Sub, nullptr);
  CHECK_STATUS(status);

  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_1_Sub[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(self, other, out);

  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(), other.scalar_type()}, {});

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(other, supportedTypes) ||
      !IsSupportedType(self, supportedTypes) ||
      !c10::canCast(*promoted_type, out.scalar_type())) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(mul_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type) {
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type) {
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }

  // resize the output and then create output ort value to be updated.
  onnxruntime::TensorShape out_shape;
  resize_output(
      invoker,
      dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()),
      BroadcastShape(__func__, ort_input_0_self, ort_input_0_other, out_shape));

  auto ort_input_out = create_ort_value(invoker, out);

  std::vector<OrtValue> ort_outputs_0_Mul(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_0_Mul[0] = ort_input_out;
  }

  auto status = invoker.Invoke("Mul",
                               {std::move(ort_input_0_self), std::move(ort_input_0_other)},
                               ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);

  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_0_Mul[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    // *,
    at::Tensor& out) {
  ORT_LOG_FN(self, other, out);

  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(), other.scalar_type()}, {});

  if (
      std::vector<at::ScalarType> supportedTypes =
          {at::kBFloat16, at::kByte, at::kDouble, at::kFloat, at::kHalf, at::kInt, at::kLong, at::kShort};
      !IsSupportedType(other, supportedTypes) ||
      !IsSupportedType(self, supportedTypes) ||
      !c10::canCast(*promoted_type, out.scalar_type())) {
    return at::native::call_fallback_fn<
        &at::native::cpu_fallback,
        ATEN_OP(div_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());

  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type) {
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type) {
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }

  // resize the output and then create output ort value to be updated.
  onnxruntime::TensorShape out_shape;
  resize_output(
      invoker,
      dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()),
      BroadcastShape(__func__, ort_input_0_self, ort_input_0_other, out_shape));

  auto ort_input_out = create_ort_value(invoker, out);

  std::vector<OrtValue> ort_outputs_0_Div(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_0_Div[0] = ort_input_out;
  }

  auto status = invoker.Invoke("Div",
                               {std::move(ort_input_0_self), std::move(ort_input_0_other)},
                               ort_outputs_0_Div, nullptr);
  CHECK_STATUS(status);

  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_0_Div[0], ort_input_out, out.scalar_type());
  }
  return out;
}

}  // namespace aten

// #pragma endregion

}  // namespace eager
}  // namespace torch_ort
