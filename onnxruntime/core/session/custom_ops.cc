// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include <type_traits>
#include <unordered_set>

#include "core/common/gsl.h"
#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/allocator_adapters.h"
#include "core/session/api_utils.h"
#include "core/session/custom_ops.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"

#if !defined(ORT_MINIMAL_BUILD)
static constexpr uint32_t min_ort_version_with_optional_io_support = 8;
static constexpr uint32_t min_ort_version_with_variadic_io_support = 14;
static constexpr uint32_t min_ort_version_with_custom_version = 17;
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
static constexpr uint32_t min_ort_version_with_compute_v2_support = 16;
static constexpr uint32_t min_ort_version_with_shape_inference = 17;
#endif

#if !defined(DISABLE_FLOAT8_TYPES)
#define SUPPORTED_TENSOR_TYPES DataTypeImpl::AllTensorTypesIRv9()
#else
#define SUPPORTED_TENSOR_TYPES DataTypeImpl::AllTensorTypesIRv4()
#endif

#if defined(ORT_MINIMAL_BUILD)
struct OrtShapeInferContext {
  size_t GetInputCount() const { return 0; }
  OrtTensorTypeAndShapeInfo* GetInputTypeShape(size_t) const { return {}; }
  onnxruntime::Status SetOutputTypeShape(size_t, const OrtTensorTypeAndShapeInfo*) const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "OrtShapeInferContext::SetOutputTypeShape not implemented for minimal build");
  }
  const ONNX_NAMESPACE::AttributeProto* GetAttr(const char*) const { return {}; }
};
#else
struct OrtShapeInferContext {
  OrtShapeInferContext(ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
    auto num_inputs = ctx_.getNumInputs();
    for (size_t ith_input = 0; ith_input < num_inputs; ++ith_input) {
      const auto* input_type = ctx_.getInputType(ith_input);
      const auto& value_case = input_type->value_case();
      ORT_ENFORCE(value_case == ONNX_NAMESPACE::TypeProto::kTensorType, "shape inference not yet supported for non-tensor types");
      const auto& shape_proto = input_type->tensor_type().shape();
      const auto& type_proto = input_type->tensor_type();
      auto elem_type = ::onnxruntime::utils::CApiElementTypeFromProtoType(type_proto.elem_type());
      auto tensor_shape = ::onnxruntime::utils::GetTensorShapeFromTensorShapeProto(shape_proto);
      auto symbolic_dims = GetSymbolicDims(shape_proto);
      input_type_shapes_.emplace_back(OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(elem_type, tensor_shape, &symbolic_dims).release());
    }
  }

  ~OrtShapeInferContext() = default;
  size_t GetInputCount() const { return input_type_shapes_.size(); }

  OrtTensorTypeAndShapeInfo* GetInputTypeShape(size_t idx) const {
    return input_type_shapes_.at(idx).get();
  }

  onnxruntime::Status SetOutputTypeShape(size_t index, const OrtTensorTypeAndShapeInfo* info) const {
    ORT_RETURN_IF_NOT(info, "Invalid shape info");
    ONNX_NAMESPACE::TensorShapeProto shape_proto;
    const auto& symbolic_dims = info->dim_params;
    const auto& integer_dims = info->shape.GetDims();
    ORT_RETURN_IF_NOT(symbolic_dims.size() == integer_dims.size(), "symbolic and integer dims mismatch!");
    for (size_t ith = 0; ith < symbolic_dims.size(); ith++) {
      auto* dim_proto = shape_proto.add_dim();
      if (symbolic_dims[ith].size() > 0) {
        dim_proto->set_dim_param(symbolic_dims[ith]);
      } else {
        dim_proto->set_dim_value(integer_dims[ith]);
      }
    }
    ONNX_NAMESPACE::updateOutputShape(ctx_, index, shape_proto);
    return onnxruntime::Status::OK();
  }

  const ONNX_NAMESPACE::AttributeProto* GetAttr(const char* attr_name) const {
    return ctx_.getAttribute(attr_name);
  }

 private:
  static std::vector<std::string> GetSymbolicDims(const ONNX_NAMESPACE::TensorShapeProto& shape_proto) {
    std::vector<std::string> symblic_dims;
    for (int ith = 0; ith < shape_proto.dim_size(); ith++) {
      const auto& dim = shape_proto.dim(ith);
      if (::onnxruntime::utils::HasDimValue(dim)) {
        symblic_dims.emplace_back();
      } else {
        symblic_dims.emplace_back(dim.dim_param());
      }
    }
    return symblic_dims;
  }
  ONNX_NAMESPACE::InferenceContext& ctx_;
  using TypeShapePtr = std::unique_ptr<OrtTensorTypeAndShapeInfo>;
  onnxruntime::InlinedVector<TypeShapePtr> input_type_shapes_;
};
#endif

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetInputCount, _In_ const OrtShapeInferContext* context, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = context->GetInputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetInputTypeShape, _In_ const OrtShapeInferContext* context, _In_ size_t index, _Outptr_ OrtTensorTypeAndShapeInfo** info) {
  API_IMPL_BEGIN
  *info = context->GetInputTypeShape(index);
  if (*info) {
    return nullptr;
  } else {
    return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Failed to fetch type shape info for the index.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetAttribute, _In_ const OrtShapeInferContext* context, _In_ const char* attr_name, _Outptr_ const OrtOpAttr** attr) {
  API_IMPL_BEGIN
  *attr = reinterpret_cast<const OrtOpAttr*>(context->GetAttr(attr_name));
  if (*attr) {
    return nullptr;
  } else {
    return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute does not exist.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReadOpAttr,
                    _In_ const OrtOpAttr* op_attr,
                    _In_ OrtOpAttrType type,
                    _Inout_ void* data,
                    _In_ size_t len,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN

  if (!op_attr) {
    return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Invalid attribute.");
  }

  auto attr = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(op_attr);
  OrtStatusPtr ret = nullptr;
  *out = 0;

  if (type == OrtOpAttrType::ORT_OP_ATTR_FLOAT) {
    if (len < sizeof(float)) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold a float.");
    } else {
      if (attr->has_f()) {
        auto output_f = reinterpret_cast<float*>(data);
        *output_f = attr->f();
      } else {
        ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute has no float value.");
      }
    }
    *out = sizeof(float);

  } else if (type == OrtOpAttrType::ORT_OP_ATTR_FLOATS) {
    const auto& floats = attr->floats();
    auto num_floats = floats.size();

    if (len < sizeof(float) * num_floats) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold the array of floats.");
    } else {
      auto output_f = reinterpret_cast<float*>(data);
      for (auto f : floats) {
        *output_f = f;
        output_f++;
      }
    }
    *out = num_floats * sizeof(float);

  } else if (type == OrtOpAttrType::ORT_OP_ATTR_INT) {
    if (len < sizeof(int)) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold an int64.");
    } else {
      if (attr->has_i()) {
        auto output_i = reinterpret_cast<int64_t*>(data);
        *output_i = attr->i();
      } else {
        ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute has no int64 value.");
      }
    }
    *out = sizeof(int64_t);

  } else if (type == OrtOpAttrType::ORT_OP_ATTR_INTS) {
    const auto& ints = attr->ints();
    auto num_ints = ints.size();

    if (len < sizeof(int64_t) * num_ints) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold the array of int64.");
    } else {
      auto output_i = reinterpret_cast<int64_t*>(data);
      for (auto i : ints) {
        *output_i = i;
        output_i++;
      }
    }
    *out = num_ints * sizeof(int64_t);

  } else if (type == OrtOpAttrType::ORT_OP_ATTR_STRING) {
    const auto& s = attr->s();
    if (len < s.size() + 1) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold the string.");
    } else {
      char* output_c = reinterpret_cast<char*>(data);
      for (char c : s) {
        *output_c++ = c;
      }
      *output_c = '\0';
    }
    *out = s.size() + 1;

  } else if (type == OrtOpAttrType::ORT_OP_ATTR_STRINGS) {
    const auto& ss = attr->strings();
    size_t num_bytes = 0;
    for_each(ss.begin(), ss.end(), [&num_bytes](const std::string& s) { num_bytes += s.size() + 1; });

    if (len < num_bytes) {
      ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Size of data not large enough to hold the array of strings.");
    } else {
      char* output_c = reinterpret_cast<char*>(data);
      for (const auto& s : ss) {
        for (char c : s) {
          *output_c++ = c;
        }
        *output_c++ = '\0';
      }
    }
    *out = num_bytes;

  } else {
    ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Unknown attribute type.");
  }

  return ret;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_SetOutputTypeShape, _In_ const OrtShapeInferContext* context, _In_ size_t index, _In_ const OrtTensorTypeAndShapeInfo* info) {
  API_IMPL_BEGIN
  auto status = context->SetOutputTypeShape(index, info);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return OrtApis::CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out) {
  API_IMPL_BEGIN
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out) {
  API_IMPL_BEGIN
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->InputCount();
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->OutputCount();
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<const OrtValue*>(reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->GetInputMLValue(gsl::narrow_cast<int>(index)));
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out) {
  API_IMPL_BEGIN
  onnxruntime::TensorShape shape(dim_values, dim_count);
  *out = reinterpret_cast<OrtValue*>(reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->OutputMLValue(gsl::narrow_cast<int>(index), shape));
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size) {
  API_IMPL_BEGIN
  std::string value;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<std::string>(name, &value);
  if (status.IsOK()) {
    if (out == nullptr) {  // User is querying the true size of the attribute
      *size = value.size() + 1;
      return nullptr;
    } else if (*size >= value.size() + 1) {
      std::memcpy(out, value.data(), value.size());
      out[value.size()] = '\0';
      *size = value.size() + 1;
      return nullptr;
    } else {  // User has provided a buffer that is not large enough
      *size = value.size() + 1;
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Result buffer is not large enough");
    }
  }
  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 28196 6387)
#endif

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context, _Outptr_ void** out) {
  API_IMPL_BEGIN
  auto* stream = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetComputeStream();
  if (stream)
    *out = stream->GetHandle();
  else
    *out = nullptr;
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetAllocator, _In_ const OrtKernelContext* context, _In_ const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  onnxruntime::AllocatorPtr allocator = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetAllocator(mem_info->device);
  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  std::unique_ptr<onnxruntime::OrtAllocatorImplWrappingIAllocator> p = std::make_unique<onnxruntime::OrtAllocatorImplWrappingIAllocator>(std::move(allocator));
  *out = p.release();
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetResource, _In_ const OrtKernelContext* context, _In_ int resource_version, _In_ int resource_id, _Outptr_ void** resource) {
  API_IMPL_BEGIN
  *resource = {};
  const auto* ctx = reinterpret_cast<const onnxruntime::OpKernelContext*>(context);
  auto* stream = reinterpret_cast<onnxruntime::Stream*>(ctx->GetComputeStream());
  if (!stream) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Failed to fetch a stream hosting the requested resource");
  }
  *resource = stream->GetResource(resource_version, resource_id);
  if (!(*resource)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Requested resource does not exist");
  }
  return nullptr;
  API_IMPL_END
};

#ifdef _WIN32
#pragma warning(pop)
#endif

template <typename T, typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
static Status CopyDataFromVectorToMemory(const std::vector<T>& values, T* out, size_t* size) {
  if (out == nullptr) {  // User is querying the true size of the attribute
    *size = values.size();
    return Status::OK();
  } else if (*size >= values.size()) {
    std::memcpy(out, values.data(), values.size() * sizeof(T));
    *size = values.size();
  } else {  // User has provided a buffer that is not large enough
    *size = values.size();
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Result buffer is not large enough");
  }

  return Status::OK();
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ float* out, _Inout_ size_t* size) {
  API_IMPL_BEGIN
  std::vector<float> values;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<float>(name, values);
  if (status.IsOK()) {
    status = CopyDataFromVectorToMemory<float>(values, out, size);
  }
  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ int64_t* out, _Inout_ size_t* size) {
  API_IMPL_BEGIN
  std::vector<int64_t> values;
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<int64_t>(name, values);
  if (status.IsOK()) {
    status = CopyDataFromVectorToMemory<int64_t>(values, out, size);
  }
  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_tensor, _In_ const OrtKernelInfo* info, _In_z_ const char* name,
                    _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  const auto* op_kinfo = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);

  // Get TensorProto attribute
  onnx::TensorProto tensor_proto;
  auto status = op_kinfo->GetAttr<onnx::TensorProto>(name, &tensor_proto);
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }

  // Determine the tensor's size in bytes.
  size_t req_size = 0;
  status = onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(tensor_proto, &req_size);
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }

  // Create Tensor that owns buffer memory that will be allocated with the provided OrtAllocator.
  onnxruntime::TensorShape tensor_shape = onnxruntime::utils::GetTensorShapeFromTensorProto(tensor_proto);
  const auto* const type = onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  onnxruntime::AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  auto tensorp = std::make_unique<onnxruntime::Tensor>(type, tensor_shape, std::move(alloc_ptr));

  // Deserialize TensorProto into pre-allocated, empty Tensor.
  status = onnxruntime::utils::TensorProtoToTensor(onnxruntime::Env::Default(), nullptr, tensor_proto, *tensorp);
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }

  // Initialize OrtValue from Tensor.
  auto ml_tensor = onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>();
  auto value = std::make_unique<OrtValue>();
  value->Init(tensorp.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetInputCount();
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetOutputCount();
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out,
                    _Inout_ size_t* size) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  const auto input_defs = op_info->node().InputDefs();

  if (index >= input_defs.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo input index is out of bounds");
  }

  auto status = CopyStringToOutputArg(input_defs[index]->Name(),
                                      "Output buffer is not large enough for ::OrtKernelInfo input name", out, size);

  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out,
                    _Inout_ size_t* size) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  const auto output_defs = op_info->node().OutputDefs();

  if (index >= output_defs.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo output index is out of bounds");
  }

  auto status = CopyStringToOutputArg(output_defs[index]->Name(),
                                      "Output buffer is not large enough for ::OrtKernelInfo output name", out, size);

  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  const auto input_defs = op_info->node().InputDefs();

  if (index >= input_defs.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo input index is out of bounds");
  }

  const onnxruntime::NodeArg* node_arg = input_defs[index];
  const ONNX_NAMESPACE::TypeProto* type_proto = node_arg->TypeAsProto();

  if (type_proto == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_GRAPH, "::OrtKernelInfo input does not have a type");
  }

  auto type_info_ret = OrtTypeInfo::FromTypeProto(*type_proto);
  *type_info = type_info_ret.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  const auto output_defs = op_info->node().OutputDefs();

  if (index >= output_defs.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo output index is out of bounds");
  }

  const onnxruntime::NodeArg* node_arg = output_defs[index];
  const ONNX_NAMESPACE::TypeProto* type_proto = node_arg->TypeAsProto();

  if (type_proto == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_GRAPH, "::OrtKernelInfo output does not have a type");
  }

  auto type_info_ret = OrtTypeInfo::FromTypeProto(*type_proto);
  *type_info = type_info_ret.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetConstantInput_tensor, _In_ const OrtKernelInfo* info, _In_ size_t index,
                    _Out_ int* is_constant, _Outptr_ const OrtValue** out) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  *is_constant = static_cast<int>(op_info->TryGetConstantInput(gsl::narrow_cast<int>(index), out));
  return nullptr;
  API_IMPL_END
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetNodeName, _In_ const OrtKernelInfo* info, _Out_ char* out,
                    _Inout_ size_t* size) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);

  auto status = CopyStringToOutputArg(op_info->node().Name(),
                                      "Output buffer is not large enough for ::OrtKernelInfo node name", out, size);

  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetLogger, _In_ const OrtKernelInfo* info, _Outptr_ const OrtLogger** logger) {
  API_IMPL_BEGIN
  const auto* ep = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetExecutionProvider();

  if (ep == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_GRAPH, "::OrtKernelInfo does not have an execution provider");
  }

  const auto* ep_logger = ep->GetLogger();

  if (ep_logger == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_GRAPH,
                                 "::OrtKernelInfo cannot get a valid logger from "
                                 "its execution provider");
  }

  *logger = reinterpret_cast<const OrtLogger*>(ep_logger);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetLogger, _In_ const OrtKernelContext* context, _Outptr_ const OrtLogger** logger) {
  API_IMPL_BEGIN
  const auto& kernel_ctx_logger = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->Logger();

  *logger = reinterpret_cast<const OrtLogger*>(&kernel_ctx_logger);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Logger_LogMessage, _In_ const OrtLogger* logger, OrtLoggingLevel log_severity_level,
                    _In_z_ const char* message, _In_z_ const ORTCHAR_T* file_path, int line_number,
                    _In_z_ const char* func_name) {
  API_IMPL_BEGIN
  const auto& actual_logger = *reinterpret_cast<const onnxruntime::logging::Logger*>(logger);
  const auto severity = static_cast<onnxruntime::logging::Severity>(log_severity_level);
  const auto log_data_type = onnxruntime::logging::DataType::SYSTEM;

  if (actual_logger.OutputIsEnabled(severity, log_data_type)) {
#ifdef _WIN32
    const std::string file_path_str = onnxruntime::ToUTF8String(file_path);
    onnxruntime::CodeLocation location(file_path_str.c_str(), line_number, func_name);
#else
    onnxruntime::CodeLocation location(file_path, line_number, func_name);
#endif

    onnxruntime::logging::Capture(
        actual_logger,
        severity,
        onnxruntime::logging::Category::onnxruntime,
        log_data_type,
        location)
            .Stream()
        << message;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Logger_GetLoggingSeverityLevel, _In_ const OrtLogger* logger, _Out_ OrtLoggingLevel* out) {
  API_IMPL_BEGIN
  const auto& actual_logger = *reinterpret_cast<const onnxruntime::logging::Logger*>(logger);
  *out = static_cast<OrtLoggingLevel>(actual_logger.GetSeverity());
  return nullptr;
  API_IMPL_END
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/customregistry.h"
namespace onnxruntime {

struct CustomOpKernel : OpKernel {
  CustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    if (op_.version > ORT_API_VERSION) {
      ORT_THROW("Unsupported version '" + std::to_string(op_.version) + "' in custom op '" + op.GetName(&op));
    }

    if (op_.version >= min_ort_version_with_compute_v2_support &&
        op_.CreateKernelV2) {
      op_kernel_ = nullptr;
      Ort::ThrowOnError(
          op_.CreateKernelV2(
              &op_,
              OrtGetApiBase()->GetApi(op_.version),
              reinterpret_cast<const OrtKernelInfo*>(&info),
              &op_kernel_));
    } else {
      op_kernel_ = op_.CreateKernel(&op_, OrtGetApiBase()->GetApi(op_.version),
                                    reinterpret_cast<const OrtKernelInfo*>(&info));
    }
  }

  ~CustomOpKernel() override {
    op_.KernelDestroy(op_kernel_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    if (op_.version >= min_ort_version_with_compute_v2_support &&
        op_.KernelComputeV2) {
      auto status_ptr = op_.KernelComputeV2(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
      return ToStatus(status_ptr);
    } else {
      op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
      return Status::OK();
    }
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

#if !defined(ORT_MINIMAL_BUILD)
KernelCreateInfo CreateKernelCreateInfo(const std::string& domain, const OrtCustomOp* op) {
  const size_t input_count = op->GetInputTypeCount(op);
  const size_t output_count = op->GetOutputTypeCount(op);

  KernelDefBuilder def_builder;
  def_builder.SetName(op->GetName(op))
      .SetDomain(domain);

  if (op->version >= min_ort_version_with_custom_version) {
    if (op->GetStartVersion && op->GetEndVersion) {
      def_builder.SinceVersion(op->GetStartVersion(op), op->GetEndVersion(op));
    } else if (op->GetStartVersion) {
      def_builder.SinceVersion(op->GetStartVersion(op));
    } else {
      def_builder.SinceVersion(1);
    }
  } else {
    def_builder.SinceVersion(1);
  }

  // GetInputMemoryType was introduced in ver 13. This check allows custom ops compiled using older versions
  // to work with newer versions (> 12) of the ORT binary.
  if (op->version > 12) {
    for (size_t i = 0; i < input_count; i++) {
      def_builder.InputMemoryType(op->GetInputMemoryType(op, i), gsl::narrow_cast<int>(i));
    }
  }

  for (size_t i = 0; i < input_count; i++) {
    const auto input_type = op->GetInputType(op, i);
    const auto input_name = "Input" + std::to_string(i);
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      def_builder.TypeConstraint(input_name, SUPPORTED_TENSOR_TYPES);
    } else {
      def_builder.TypeConstraint(input_name, DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(input_type))->AsTensorType());
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    const auto output_type = op->GetOutputType(op, i);
    const auto output_name = "Output" + std::to_string(i);
    if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      def_builder.TypeConstraint(output_name, SUPPORTED_TENSOR_TYPES);
    } else {
      def_builder.TypeConstraint(output_name, DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(output_type))->AsTensorType());
    }
  }

  if (const char* provider_type = op->GetExecutionProviderType(op)) {
    def_builder.Provider(provider_type);
  } else {
    def_builder.Provider(onnxruntime::kCpuExecutionProvider);
  }

  KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<CustomOpKernel>(info, *op);
    return Status::OK();
  };

  return KernelCreateInfo(def_builder.Build(), kernel_create_fn);
}

ONNX_NAMESPACE::OpSchema CreateSchema(const std::string& domain, const std::vector<const OrtCustomOp*>& ops) {
  // The function registers the first schema assuming all the other one are the same except the types constraints.
  ORT_ENFORCE(ops.size() > 0, "No kernels to registers.");
  auto op = *ops.begin();
  const size_t input_count = op->GetInputTypeCount(op);
  const size_t output_count = op->GetOutputTypeCount(op);
  int undefined = 0;

  ONNX_NAMESPACE::OpSchema schema(op->GetName(op), "custom op registered at runtime", 0);

  for (size_t i = 0; i < input_count; i++) {
    onnx::OpSchema::FormalParameterOption option = onnx::OpSchema::FormalParameterOption::Single;
    bool is_homogeneous = true;
    int min_arity = 1;

    // The OrtCustomOp interface did not support the methods to query input/output characteristics before
    // ORT API version 8. So, query the relevant methods ONLY from API version 8 onwards.
    if (op->version >= min_ort_version_with_optional_io_support) {
      const auto characteristic = op->GetInputCharacteristic(op, i);

      // Support for optional and variadic inputs/output was added in versions 8 and 14, respectively.
      if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
        option = onnx::OpSchema::FormalParameterOption::Optional;
      } else if ((op->version >= min_ort_version_with_variadic_io_support) &&
                 (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC)) {
        ORT_ENFORCE(i == input_count - 1, "Only the last input to a custom op may be marked variadic.");
        option = onnx::OpSchema::FormalParameterOption::Variadic;
        min_arity = op->GetVariadicInputMinArity(op);
        is_homogeneous = static_cast<bool>(op->GetVariadicInputHomogeneity(op));
      }
    }

    std::unordered_set<ONNXTensorElementDataType> all_types;
    for (auto o : ops) {
      ORT_ENFORCE(i < o->GetInputTypeCount(o), "Another version of operator '", schema.Name(), "'has less inputs.",
                  "onnxruntime allwos the overloading of an operator if all versions have the same number of "
                  "declared inputs.");
      const auto type = o->GetInputType(o, i);
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        // If 'type' is undefined, all types are allowed regardless of what other versions of the same operator
        // define. In that case, all_types is cleared, that's the convention used by the code following this loop
        // to declare all types as possible types.
        all_types.clear();
        break;
      }
      all_types.insert(type);
    }

    std::string input_name = "Input" + std::to_string(i);
    schema.Input(gsl::narrow_cast<int>(i), input_name, "", input_name, option, is_homogeneous, min_arity);

    if (!all_types.empty()) {
      // all_types is not empty then only the types in this container are allowed of this input.
      std::vector<std::string> input_types;
      for (auto type : all_types) {
        const ONNX_NAMESPACE::TypeProto* type_proto =
            DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(type))->GetTypeProto();
        input_types.push_back(*ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto));
      }
      schema.TypeConstraint(input_name, input_types, "defined list of types");
    } else {
      // all_types is empty. As mentioned in the previous loop, all types are allowed.
      schema.TypeConstraint(input_name, DataTypeImpl::ToString(SUPPORTED_TENSOR_TYPES), "all types");
      undefined++;
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    onnx::OpSchema::FormalParameterOption option = onnx::OpSchema::FormalParameterOption::Single;
    bool is_homogeneous = true;
    int min_arity = 1;

    // The OrtCustomOp interface did not support the methods to query input/output characteristics before
    // ORT API version 8. So, query the relevant methods ONLY from API version 8 onwards.
    if (op->version >= min_ort_version_with_optional_io_support) {
      const auto characteristic = op->GetOutputCharacteristic(op, i);

      // Support for optional and variadic inputs/output was added in versions 8 and 14, respectively.
      if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
        option = onnx::OpSchema::FormalParameterOption::Optional;
      } else if ((op->version >= min_ort_version_with_variadic_io_support) &&
                 (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC)) {
        ORT_ENFORCE(i == output_count - 1, "Only the last output to a custom op may be marked variadic.");
        option = onnx::OpSchema::FormalParameterOption::Variadic;
        min_arity = op->GetVariadicOutputMinArity(op);
        is_homogeneous = static_cast<bool>(op->GetVariadicOutputHomogeneity(op));
      }
    }
    const auto type = op->GetOutputType(op, i);
    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
      if (op->GetOutputCharacteristic(op, i) == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED) {
        ORT_ENFORCE(1 == undefined,
                    "There must be one (and only one) dynamic typed input to the custom op. "
                    "Its type info at runtime will be used to infer the type info of this dynamic typed output "
                    "which is required for the success of the model loading step. "
                    "More than one dynamic typed inputs are currently not supported as differing types at runtime means the output type "
                    "cannot be inferred without which model loading cannot proceed.");
      }
    }

    std::unordered_set<ONNXTensorElementDataType> all_types;
    for (auto o : ops) {
      ORT_ENFORCE(i < o->GetOutputTypeCount(o), "Another version of operator '", schema.Name(), "'has less outputs.");
      const auto otype = o->GetOutputType(o, i);
      if (otype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        all_types.clear();
        break;
      }
      all_types.insert(otype);
    }

    std::string output_name = "Output" + std::to_string(i);
    schema.Output(gsl::narrow_cast<int>(i), output_name, "", output_name, option, is_homogeneous, min_arity);

    if (!all_types.empty()) {
      std::vector<std::string> output_types;
      for (auto otype : all_types) {
        const ONNX_NAMESPACE::TypeProto* type_proto =
            DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(otype))->GetTypeProto();
        output_types.push_back(*ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto));
      }
      schema.TypeConstraint(output_name, output_types, "defined list of types");
    } else {
      schema.TypeConstraint(output_name, DataTypeImpl::ToString(SUPPORTED_TENSOR_TYPES), "all types");
      undefined++;
    }
  }

  schema.SetDomain(domain);
  if (op->version >= min_ort_version_with_custom_version && op->GetStartVersion) {
    schema.SinceVersion(op->GetStartVersion(op));
  } else {
    schema.SinceVersion(1);
  }
  schema.AllowUncheckedAttributes();

  if (op->version >= min_ort_version_with_shape_inference && op->InferOutputShapeFn) {
    schema.TypeAndShapeInferenceFunction([op](ONNX_NAMESPACE::InferenceContext& infer_ctx) {
      OrtShapeInferContext ctx(infer_ctx);
      op->InferOutputShapeFn(op, &ctx);
    });
  }
  return schema;
}

Status IsCompatible(const ONNX_NAMESPACE::OpSchema& schema, const OrtCustomOp* op) {
  const size_t input_count = op->GetInputTypeCount(op);
  const size_t output_count = op->GetOutputTypeCount(op);

  // check inputs
  const auto& input_parameters = schema.inputs();
  ORT_RETURN_IF_NOT(input_parameters.size() == input_count, "input count does not match");
  for (size_t i = 0; i < input_parameters.size(); ++i) {
    const auto characteristic = op->GetInputCharacteristic(op, i);
    const auto& formal_parameter = input_parameters[i];
    if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
      ORT_RETURN_IF_NOT(op->version < min_ort_version_with_optional_io_support ||
                            formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Optional,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " input to be of optional type");
    } else if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC) {
      ORT_RETURN_IF_NOT(formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Variadic,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " input to be of variadic type");
      ORT_RETURN_IF_NOT(op->version < min_ort_version_with_variadic_io_support ||
                            formal_parameter.GetIsHomogeneous() == (op->GetVariadicInputHomogeneity(op) != 0),
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " input to keep same homogeneity");
      ORT_RETURN_IF_NOT(formal_parameter.GetMinArity() == op->GetVariadicInputMinArity(op),
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " input to keep same arity");
    } else {
      ORT_RETURN_IF_NOT(formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Single,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " input to be of single type");
    }
  }
  // check outputs
  const auto& output_parameters = schema.outputs();
  ORT_RETURN_IF_NOT(output_parameters.size() == output_count, "output count does not match");
  for (size_t i = 0; i < output_parameters.size(); ++i) {
    const auto characteristic = op->GetOutputCharacteristic(op, i);
    const auto& formal_parameter = output_parameters[i];
    if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
      ORT_RETURN_IF_NOT(formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Optional,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " output to be of optional type");
    } else if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC) {
      ORT_RETURN_IF_NOT(formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Variadic,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " output to be of variadic type");
      ORT_RETURN_IF_NOT(formal_parameter.GetIsHomogeneous() == (op->GetVariadicOutputHomogeneity(op) != 0),
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " output to keep same homogeneity");
      ORT_RETURN_IF_NOT(formal_parameter.GetMinArity() == op->GetVariadicOutputMinArity(op),
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " output to keep same arity");
    } else {
      ORT_RETURN_IF_NOT(formal_parameter.GetOption() == onnx::OpSchema::FormalParameterOption::Single,
                        "custom op schemas mismatch, expecting ", i + 1,
                        i == 0 ? "st" : (i == 1 ? "nd" : "th"),
                        " output to be of single type");
    }
  }
  return Status::OK();
}

void InferOutputTypes(const InlinedVector<const KernelDef*>& kernel_defs,
                      ONNX_NAMESPACE::InferenceContext& infer_ctx) {
  for (const auto& kernel_def : kernel_defs) {
    const auto& type_constraints = kernel_def->TypeConstraints();
    auto num_inputs = infer_ctx.getNumInputs();
    bool matched = true;
    ONNXTensorElementDataType undef = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // first, make sure there is a constraint for every input
    for (size_t i = 0; i < num_inputs && matched; ++i) {
      auto input_name = "Input" + std::to_string(i);
      auto input_type = infer_ctx.getInputType(i);
      if (input_type) {
        auto elem_type = static_cast<ONNXTensorElementDataType>(input_type->tensor_type().elem_type());
        auto tc_iter = type_constraints.find(input_name);
        if (tc_iter != type_constraints.end()) {
          if (tc_iter->second.size() > 1) {
            undef = elem_type;
          } else if (tc_iter->second.size() != 1 || tc_iter->second[0] != DataTypeImpl::TensorTypeFromONNXEnum(elem_type)) {
            matched = false;
          }
        } else {
          matched = false;
        }
      } else {
        matched = false;
      }
    }  // for
    // next, ensure that there is a constraint for every output
    auto num_outputs = infer_ctx.getNumOutputs();
    for (size_t i = 0; i < num_outputs && matched; i++) {
      auto output_name = "Output" + std::to_string(i);
      auto tc_iter = type_constraints.find(output_name);
      if (tc_iter == type_constraints.end() || tc_iter->second.size() < 1) {
        matched = false;
      }
    }
    if (matched) {
      for (size_t i = 0; i < num_outputs; i++) {
        auto output_name = "Output" + std::to_string(i);
        auto output_type = infer_ctx.getOutputType(i);
        auto tc_iter = type_constraints.find(output_name);
        if (tc_iter->second.size() > 1) {
          output_type->mutable_tensor_type()->set_elem_type(undef);
        } else {
          output_type->mutable_tensor_type()->set_elem_type(tc_iter->second[0]->GetTypeProto()->tensor_type().elem_type());
        }
      }
      break;
    }
  }
}
#endif

common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains,
                                    std::shared_ptr<CustomRegistry>& output) {
  output = std::make_shared<CustomRegistry>();

  for (const auto& domain : op_domains) {
#if !defined(ORT_MINIMAL_BUILD)
    std::unordered_map<std::string, ONNX_NAMESPACE::OpSchema> schema_map;
    std::unordered_map<std::string, InlinedVector<const KernelDef*>> kernel_def_map;

    // Domain is not empty - add it to the DomainToVersion ONNX map
    // If domain is empty, it is assumed to be part of the ONNX domain
    if (!domain->domain_.empty()) {
      // Add it to the DomainToVersion ONNX map if it doesn't already exist
      // For example, two sessions using the same session_options should not add the same custom op domain to the version map twice
      auto& domain_to_version_range_instance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
      const auto& domain_to_version_map = domain_to_version_range_instance.Map();

      if (domain_to_version_map.find(domain->domain_) == domain_to_version_map.end()) {
        domain_to_version_range_instance.AddDomainToVersion(domain->domain_, 1, 1000);
      }
    }

    // domain_kernels aggregate all custom operator per names.
    std::unordered_map<std::string, std::vector<const OrtCustomOp*>> domain_kernels;
    for (const auto* op : domain->custom_ops_) {
      // define kernel
      auto it = domain_kernels.find(op->GetName(op));
      if (it == domain_kernels.end()) {
        domain_kernels[op->GetName(op)] = {op};
      } else {
        domain_kernels[op->GetName(op)].push_back(op);
      }
    }

    // Creation of the schemas, one per unique name.
    for (auto& [name, ops] : domain_kernels) {
      auto schema = CreateSchema(domain->domain_, ops);
      ORT_ENFORCE(name == schema.Name());
      schema_map.emplace(schema.Name(), schema);
    }

    // This loops checks that all custom operators sharing the same name are compatible with the defined schema.
    for (const auto* op : domain->custom_ops_) {
      // define kernel
      auto kernel_create_info = CreateKernelCreateInfo(domain->domain_, op);
      kernel_def_map[op->GetName(op)].push_back(kernel_create_info.kernel_def.get());
      ORT_RETURN_IF_ERROR(output->RegisterCustomKernel(kernel_create_info));
      // define schema
      auto schema_map_iter = schema_map.find(op->GetName(op));
      ORT_ENFORCE(schema_map_iter != schema_map.end(),
                  "This condition fail is no schema was defined for this operator as it should have in the previous loop.");
      ORT_RETURN_IF_ERROR(IsCompatible(schema_map_iter->second, op),
                          "All custom operators named '", op->GetName(op), "' are not compatible among themselves. ",
                          "They should have the same number of inputs and outputs, the same characteristics ",
                          "(optional, ...). Only the type can change.");
    }

    std::vector<ONNX_NAMESPACE::OpSchema> schemas;
    for (auto schema_iter : schema_map) {
      schemas.push_back(schema_iter.second);
      InlinedVector<const KernelDef*> kernel_defs = std::move(kernel_def_map[schema_iter.first]);
      auto infer_fn = schemas.back().GetTypeAndShapeInferenceFunction();
      ONNX_NAMESPACE::InferenceFunction extended_infer_fn = [infer_fn, kernel_defs](ONNX_NAMESPACE::InferenceContext& infer_ctx) {
        InferOutputTypes(kernel_defs, infer_ctx);
        if (infer_fn) {
          infer_fn(infer_ctx);
        }
      };
      schemas.back().TypeAndShapeInferenceFunction(extended_infer_fn);
    }

    ORT_RETURN_IF_ERROR(output->RegisterOpSet(schemas,
                                              domain->domain_,
                                              1 /* baseline opset version */,
                                              1000 /* opset version */));
#else
    // For a minimal build, we may not need any of the ONNX schema stuff but we still need to track
    // the type template parameters to be used during the kernel def building step below
    for (const auto* op : domain->custom_ops_) {
      size_t undefined = 0;
      size_t input_count = op->GetInputTypeCount(op);
      for (size_t i = 0; i < input_count; i++) {
        auto type = op->GetInputType(op, i);
        if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
          undefined++;
        }
      }

      KernelDefBuilder def_builder;
      def_builder.SetName(op->GetName(op))
          .SetDomain(domain->domain_)
          .SinceVersion(1);

      // GetInputMemoryType was introduced in ver 13. This check allows custom ops compiled using older versions
      // to work with newer versions (> 12) of the ORT binary.
      if (op->version > 12) {
        for (size_t i = 0; i < input_count; i++) {
          def_builder.InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }

      for (size_t i = 0; i < undefined; i++) {
        def_builder.TypeConstraint("T" + std::to_string(i), SUPPORTED_TENSOR_TYPES);
      }

      if (const char* provider_type = op->GetExecutionProviderType(op)) {
        def_builder.Provider(provider_type);
      } else {
        def_builder.Provider(onnxruntime::kCpuExecutionProvider);
      }

      KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<CustomOpKernel>(info, *op);
        return Status::OK();
      };

      KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);
      ORT_RETURN_IF_ERROR(output->RegisterCustomKernel(create_info));
    }
#endif
  }  // for each domain

  return Status::OK();
}

}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
