// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include <string>
#include <type_traits>
#include <unordered_map>
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
#include "core/platform/threadpool.h"

// NOTE: OrtKernelContext is used by both custom ops and compiled kernels.
// In a minimal build, ORT_EXTENDED_MINIMAL_BUILD is used to enable EPs like CoreML/NNAPI which use compiled kernels,
// and ORT_MINIMAL_BUILD_CUSTOM_OPS is used to allow external custom op libraries to be used.
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#define ENABLE_ORT_KERNEL_CONTEXT_API 1
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#define ENABLE_CUSTOM_OP_API 1
#endif

#if !defined(ORT_MINIMAL_BUILD)
static constexpr uint32_t min_ort_version_with_optional_io_support = 8;
static constexpr uint32_t min_ort_version_with_variadic_io_support = 14;
static constexpr uint32_t min_ort_version_with_custom_version = 17;
#endif

#if ENABLE_CUSTOM_OP_API
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
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtShapeInferContext::SetOutputTypeShape not implemented for minimal build");
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
      ORT_ENFORCE(value_case == ONNX_NAMESPACE::TypeProto::kTensorType,
                  "shape inference not yet supported for non-tensor types");
      const auto& shape_proto = input_type->tensor_type().shape();
      const auto& type_proto = input_type->tensor_type();
      auto elem_type = ::onnxruntime::utils::CApiElementTypeFromProtoType(type_proto.elem_type());
      auto tensor_shape = ::onnxruntime::utils::GetTensorShapeFromTensorShapeProto(shape_proto);
      auto symbolic_dims = GetSymbolicDims(shape_proto);
      input_type_shapes_.emplace_back(
          OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(elem_type, tensor_shape, &symbolic_dims).release());
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

#if ENABLE_ORT_KERNEL_CONTEXT_API
template <typename T>
static OrtStatusPtr ExecuteIfKernelContextApiEnabled(const T& fn) {
  API_IMPL_BEGIN
  return fn();
  API_IMPL_END
}
#else
template <typename T>
static OrtStatusPtr ExecuteIfKernelContextApiEnabled(const T&) {
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "OrtKernelContext API is not enabled in this build");
}
#endif

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->InputCount();
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    *out = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->OutputCount();
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index,
                    _Out_ const OrtValue** out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    const auto* ctx = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context);
    *out = reinterpret_cast<const OrtValue*>(ctx->GetInputMLValue(onnxruntime::narrow<int>(index)));
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index,
                    _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    onnxruntime::TensorShape shape(dim_values, dim_count);
    auto* ctx = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
    *out = reinterpret_cast<OrtValue*>(ctx->OutputMLValue(onnxruntime::narrow<int>(index), shape));
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context,
                    _Outptr_ void** out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    auto* stream = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetComputeStream();
    if (stream)
      *out = stream->GetHandle();
    else
      *out = nullptr;
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetAllocator, _In_ const OrtKernelContext* context,
                    _In_ const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    const auto* ctx = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context);
    onnxruntime::AllocatorPtr allocator = ctx->GetAllocator(mem_info->device);
    if (!allocator) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
    }

    auto p = std::make_unique<onnxruntime::OrtAllocatorImplWrappingIAllocator>(std::move(allocator));
    *out = p.release();
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetResource, _In_ const OrtKernelContext* context,
                    _In_ int resource_version, _In_ int resource_id, _Outptr_ void** resource) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    *resource = {};
    const auto* ctx = reinterpret_cast<const onnxruntime::OpKernelContext*>(context);
    auto* stream = reinterpret_cast<onnxruntime::Stream*>(ctx->GetComputeStream());
    if (!stream) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Failed to fetch a stream hosting the requested resource");
    }
    *resource = stream->GetResource(resource_version, resource_id);
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_ParallelFor, _In_ const OrtKernelContext* context,
                    _In_ void (*fn)(void*, size_t), _In_ size_t total, _In_ size_t num_batch, _In_ void* usr_data) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    if (!context) {
      return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, "Invalid context");
    }
    if (fn && total) {
      const auto* ctx = reinterpret_cast<const onnxruntime::OpKernelContext*>(context);
      auto* tp = ctx->GetOperatorThreadPool();
      if (num_batch) {
        onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
            tp,
            static_cast<std::ptrdiff_t>(total),
            [fn, usr_data](std::ptrdiff_t ith) { fn(usr_data, static_cast<size_t>(ith)); },
            static_cast<std::ptrdiff_t>(num_batch));
      } else {
        onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
            tp,
            static_cast<std::ptrdiff_t>(total),
            [fn, usr_data](std::ptrdiff_t ith) { fn(usr_data, static_cast<size_t>(ith)); });
      }
    }
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetLogger, _In_ const OrtKernelContext* context,
                    _Outptr_ const OrtLogger** logger) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    const auto& kernel_ctx_logger = reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->Logger();

    *logger = reinterpret_cast<const OrtLogger*>(&kernel_ctx_logger);
    return nullptr;
  });
}

// Enabled via ExecuteIfKernelContextApiEnabled due to KernelContext_GetLogger
ORT_API_STATUS_IMPL(OrtApis::Logger_LogMessage, _In_ const OrtLogger* logger, OrtLoggingLevel log_severity_level,
                    _In_z_ const char* message, _In_z_ const ORTCHAR_T* file_path, int line_number,
                    _In_z_ const char* func_name) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
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
  });
}

// Enabled via ExecuteIfKernelContextApiEnabled due to KernelContext_GetLogger
ORT_API_STATUS_IMPL(OrtApis::Logger_GetLoggingSeverityLevel, _In_ const OrtLogger* logger,
                    _Out_ OrtLoggingLevel* out) {
  return ExecuteIfKernelContextApiEnabled([&]() -> OrtStatusPtr {
    const auto& actual_logger = *reinterpret_cast<const onnxruntime::logging::Logger*>(logger);
    *out = static_cast<OrtLoggingLevel>(actual_logger.GetSeverity());
    return nullptr;
  });
}

#if ENABLE_CUSTOM_OP_API
template <typename T>
static OrtStatusPtr ExecuteIfCustomOpsApiEnabled(const T& fn) {
  API_IMPL_BEGIN
  return fn();
  API_IMPL_END
}
#else
template <typename T>
static OrtStatusPtr ExecuteIfCustomOpsApiEnabled(const T&) {
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Custom operator API is not enabled in this build");
}
#endif

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetInputCount, _In_ const OrtShapeInferContext* context,
                    _Out_ size_t* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    *out = context->GetInputCount();
    return nullptr;
  });
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetInputTypeShape, _In_ const OrtShapeInferContext* context,
                    _In_ size_t index, _Outptr_ OrtTensorTypeAndShapeInfo** info) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    *info = context->GetInputTypeShape(index);
    if (*info) {
      return nullptr;
    } else {
      return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                   "Failed to fetch type shape info for the index.");
    }
  });
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_GetAttribute, _In_ const OrtShapeInferContext* context,
                    _In_ const char* attr_name, _Outptr_ const OrtOpAttr** attr) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    *attr = reinterpret_cast<const OrtOpAttr*>(context->GetAttr(attr_name));
    if (*attr) {
      return nullptr;
    } else {
      return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute does not exist.");
    }
  });
}

ORT_API_STATUS_IMPL(OrtApis::ShapeInferContext_SetOutputTypeShape, _In_ const OrtShapeInferContext* context,
                    _In_ size_t index, _In_ const OrtTensorTypeAndShapeInfo* info) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    auto status = context->SetOutputTypeShape(index, info);
    if (status.IsOK()) {
      return nullptr;
    } else {
      return OrtApis::CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
    }
  });
}

ORT_API_STATUS_IMPL(OrtApis::ReadOpAttr, _In_ const OrtOpAttr* op_attr, _In_ OrtOpAttrType type, _Inout_ void* data,
                    _In_ size_t len, _Out_ size_t* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    if (!op_attr) {
      return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Invalid attribute.");
    }

    auto attr = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(op_attr);
    OrtStatusPtr ret = nullptr;
    *out = 0;

    switch (type) {
      case OrtOpAttrType::ORT_OP_ATTR_FLOAT: {
        if (len < sizeof(float)) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold a float.");
        } else {
          if (attr->has_f()) {
            auto output_f = reinterpret_cast<float*>(data);
            *output_f = attr->f();
          } else {
            ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute has no float value.");
          }
        }
        *out = sizeof(float);

        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_FLOATS: {
        const auto& floats = attr->floats();
        auto num_floats = floats.size();

        if (len < sizeof(float) * num_floats) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold the array of floats.");
        } else {
          auto output_f = reinterpret_cast<float*>(data);
          for (auto f : floats) {
            *output_f = f;
            output_f++;
          }
        }
        *out = num_floats * sizeof(float);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_INT: {
        if (len < sizeof(int)) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold an int64.");
        } else {
          if (attr->has_i()) {
            auto output_i = reinterpret_cast<int64_t*>(data);
            *output_i = attr->i();
          } else {
            ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Attribute has no int64 value.");
          }
        }
        *out = sizeof(int64_t);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_INTS: {
        const auto& ints = attr->ints();
        auto num_ints = ints.size();

        if (len < sizeof(int64_t) * num_ints) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold the array of int64.");
        } else {
          auto output_i = reinterpret_cast<int64_t*>(data);
          for (auto i : ints) {
            *output_i = i;
            output_i++;
          }
        }
        *out = num_ints * sizeof(int64_t);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRING: {
        const auto& s = attr->s();
        if (len < s.size() + 1) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold the string.");
        } else {
          char* output_c = reinterpret_cast<char*>(data);
          for (char c : s) {
            *output_c++ = c;
          }
          *output_c = '\0';
        }
        *out = s.size() + 1;
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRINGS: {
        const auto& ss = attr->strings();
        size_t num_bytes = 0;
        for_each(ss.begin(), ss.end(), [&num_bytes](const std::string& s) { num_bytes += s.size() + 1; });

        if (len < num_bytes) {
          ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                      "Size of data not large enough to hold the array of strings.");
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
        break;
      }
      default:
        ret = OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "Unexpected attribute type. ");
    }

    return ret;
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ float* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
    if (status.IsOK())
      return nullptr;
    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ int64_t* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
    if (status.IsOK())
      return nullptr;
    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ char* out, _Inout_ size_t* size) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
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
  });
}

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
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    std::vector<float> values;
    auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<float>(name, values);
    if (status.IsOK()) {
      status = CopyDataFromVectorToMemory<float>(values, out, size);
    }
    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                    _Out_ int64_t* out, _Inout_ size_t* size) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    std::vector<int64_t> values;
    auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttrs<int64_t>(name, values);
    if (status.IsOK()) {
      status = CopyDataFromVectorToMemory<int64_t>(values, out, size);
    }
    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAttribute_tensor, _In_ const OrtKernelInfo* info, _In_z_ const char* name,
                    _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
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
    const auto* type = onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
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
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    *out = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetInputCount();
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    *out = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetOutputCount();
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputName, _In_ const OrtKernelInfo* info, size_t index,
                    _Out_ char* out, _Inout_ size_t* size) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
    const auto input_defs = op_info->node().InputDefs();

    if (index >= input_defs.size()) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo input index is out of bounds");
    }

    auto status = CopyStringToOutputArg(input_defs[index]->Name(),
                                        "Output buffer is not large enough for ::OrtKernelInfo input name", out, size);

    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out,
                    _Inout_ size_t* size) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
    const auto output_defs = op_info->node().OutputDefs();

    if (index >= output_defs.size()) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "::OrtKernelInfo output index is out of bounds");
    }

    auto status = CopyStringToOutputArg(output_defs[index]->Name(),
                                        "Output buffer is not large enough for ::OrtKernelInfo output name",
                                        out, size);

    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetInputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
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
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetOutputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
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
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetConstantInput_tensor, _In_ const OrtKernelInfo* info, _In_ size_t index,
                    _Out_ int* is_constant, _Outptr_ const OrtValue** out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
    *is_constant = static_cast<int>(op_info->TryGetConstantInput(gsl::narrow_cast<int>(index), out));
    return nullptr;
  });
};

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetNodeName, _In_ const OrtKernelInfo* info, _Out_ char* out,
                    _Inout_ size_t* size) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);

    auto status = CopyStringToOutputArg(op_info->node().Name(),
                                        "Output buffer is not large enough for ::OrtKernelInfo node name", out, size);

    return onnxruntime::ToOrtStatus(status);
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfo_GetLogger, _In_ const OrtKernelInfo* info, _Outptr_ const OrtLogger** logger) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
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
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelInfoGetAllocator, _In_ const OrtKernelInfo* info, _In_ OrtMemType mem_type, _Outptr_ OrtAllocator** out) {
  return ExecuteIfCustomOpsApiEnabled([&]() -> OrtStatusPtr {
    onnxruntime::AllocatorPtr allocator = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAllocator(mem_type);
    if (!allocator) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
    }
    auto p = std::make_unique<onnxruntime::OrtAllocatorImplWrappingIAllocator>(std::move(allocator));
    *out = p.release();
    return nullptr;
  });
}

ORT_API_STATUS_IMPL(OrtApis::KernelContext_GetScratchBuffer, _In_ const OrtKernelContext* context, _In_ const OrtMemoryInfo* mem_info, _In_ size_t count_or_bytes, _Outptr_ void** out) {
  if (count_or_bytes == 0) {
    *out = nullptr;
    return nullptr;
  }
  onnxruntime::AllocatorPtr allocator = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetAllocator(mem_info->device);
  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  onnxruntime::Stream* stream = reinterpret_cast<const onnxruntime::OpKernelContext*>(context)->GetComputeStream();
  *out = AllocateBufferWithOptions(*allocator, count_or_bytes, false, stream, stream->GetWaitNotificationFn());
  return nullptr;
};

#if ENABLE_CUSTOM_OP_API
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
      def_builder.TypeConstraint(input_name,
                                 DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(input_type))->AsTensorType());
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    const auto output_type = op->GetOutputType(op, i);
    const auto output_name = "Output" + std::to_string(i);
    if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      def_builder.TypeConstraint(output_name, SUPPORTED_TENSOR_TYPES);
    } else {
      def_builder.TypeConstraint(output_name,
                                 DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(output_type))->AsTensorType());
    }
  }

  if (const char* provider_type = op->GetExecutionProviderType(op)) {
    def_builder.Provider(provider_type);
  } else {
    def_builder.Provider(onnxruntime::kCpuExecutionProvider);
  }

  KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info,
                                         std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<CustomOpKernel>(info, *op);
    return Status::OK();
  };

  return KernelCreateInfo(def_builder.Build(), kernel_create_fn);
}

ONNX_NAMESPACE::OpSchema CreateSchema(const std::string& domain, const std::vector<const OrtCustomOp*>& ops) {
  // The function registers the first schema assuming all the other one are the same except the types constraints.
  ORT_ENFORCE(ops.size() > 0, "No kernels to registers.");
  int undefined = 0;

  // Creation of the schema for the first kernel in ops.
  const OrtCustomOp* op = *ops.begin();
  ONNX_NAMESPACE::OpSchema schema(op->GetName(op), "custom op registered at runtime", 0);

  auto create_type_constraint = [&ops, &schema, &undefined](const OrtCustomOp* op, int count, int i, bool is_input) {
    onnx::OpSchema::FormalParameterOption option = onnx::OpSchema::FormalParameterOption::Single;
    bool is_homogeneous = true;
    int min_arity = 1;

    // The OrtCustomOp interface did not support the methods to query input/output characteristics before
    // ORT API version 8. So, query the relevant methods ONLY from API version 8 onwards.
    if (op->version >= min_ort_version_with_optional_io_support) {
      const auto characteristic = is_input ? op->GetInputCharacteristic(op, i) : op->GetOutputCharacteristic(op, i);

      // Support for optional and variadic inputs/output was added in versions 8 and 14, respectively.
      if (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL) {
        option = onnx::OpSchema::FormalParameterOption::Optional;
      } else if ((op->version >= min_ort_version_with_variadic_io_support) &&
                 (characteristic == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC)) {
        ORT_ENFORCE(i == count - 1, "Only the last ", (is_input ? "input" : "output"),
                    " to a custom op may be marked variadic.");
        option = onnx::OpSchema::FormalParameterOption::Variadic;
        min_arity = is_input ? op->GetVariadicInputMinArity(op) : op->GetVariadicOutputMinArity(op);
        is_homogeneous = static_cast<bool>(is_input
                                               ? op->GetVariadicInputHomogeneity(op)
                                               : op->GetVariadicOutputHomogeneity(op));
      }
    }

    // The loop goes through all operators sharing the same schema to build
    // the minimal type constraints for all of them. All kernels must have
    // the same number of inputs / outputs among themselves to be able to build
    // the type constraints. Any kind of incompatibility between a schema and
    // a kernel is checked by method IsCompatible once the schema is created
    // by this method.
    std::unordered_set<ONNXTensorElementDataType> all_types;
    for (auto o : ops) {
      ORT_ENFORCE(static_cast<size_t>(i) != (is_input ? o->GetInputTypeCount(o) : o->GetOutputTypeCount(o)),
                  "Another version of operator '", schema.Name(),
                  "'has a different number of ", (is_input ? "inputs" : "outputs"),
                  ". onnxruntime allows the overloading of an operator "
                  "if all versions have the same number of declared ",
                  (is_input ? "inputs" : "outputs"), ".");
      const auto type = is_input ? o->GetInputType(o, i) : o->GetOutputType(o, i);
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        // If 'type' is undefined, all types are allowed regardless of what other versions of the same operator
        // define. In that case, all_types is cleared, that's the convention used by the code following this loop
        // to declare all types as possible types.
        all_types.clear();
        break;
      }
      all_types.insert(type);
    }

    std::string prefix = is_input ? "Input" : "Output";
    std::string name = prefix + std::to_string(i);
    if (is_input) {
      schema.Input(gsl::narrow_cast<int>(i), name, "", name, option, is_homogeneous, min_arity);
    } else {
      schema.Output(gsl::narrow_cast<int>(i), name, "", name, option, is_homogeneous, min_arity);
    }

    if (!all_types.empty()) {
      // all_types is not empty then only the types in this container are allowed of this input.
      std::vector<std::string> types;
      for (auto type : all_types) {
        const ONNX_NAMESPACE::TypeProto* type_proto =
            DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(type))->GetTypeProto();
        types.push_back(*ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto));
      }
      schema.TypeConstraint(name, types, "defined list of types");
    } else {
      // all_types is empty. As mentioned in the previous loop, all types are allowed.
      schema.TypeConstraint(name, DataTypeImpl::ToString(SUPPORTED_TENSOR_TYPES), "all types");
      undefined++;
    }
  };

  const size_t input_count = op->GetInputTypeCount(op);
  for (size_t i = 0; i < input_count; i++) {
    create_type_constraint(op, static_cast<int>(input_count), static_cast<int>(i), true);
  }

  const size_t output_count = op->GetOutputTypeCount(op);
  for (size_t i = 0; i < output_count; i++) {
    const auto type = op->GetOutputType(op, i);
    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
      if (op->GetOutputCharacteristic(op, i) == OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED) {
        ORT_ENFORCE(1 == undefined,
                    "There must be one (and only one) dynamic typed input to the custom op. "
                    "Its type info at runtime will be used to infer the type info of this dynamic typed output "
                    "which is required for the success of the model loading step. "
                    "More than one dynamic typed inputs are currently not supported as differing types at runtime "
                    "means the output type cannot be inferred without which model loading cannot proceed.");
      }
    }
    create_type_constraint(op, static_cast<int>(output_count), static_cast<int>(i), false);
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

// This function attempts to do its best for older custom ops (most of them) who do not have
// they own type and shape inference function. However, it falls short in some cases, and we leave
// those for the user to handle in their own inference function.
static void InferOutputTypes(const ONNX_NAMESPACE::OpSchema& schema, gsl::span<const KernelDef* const> kernel_defs,
                             ONNX_NAMESPACE::InferenceContext& infer_ctx) {
  const auto& inputs = schema.inputs();
  const auto node_input_num = infer_ctx.getNumInputs();

  const KernelDef* def_selected = nullptr;
  bool is_variadic_input = false;
  bool is_homogeneous_input = false;
  int32_t output_propagate{0};

  for (size_t kernel_index = 0;
       kernel_index < kernel_defs.size() && def_selected == nullptr;
       ++kernel_index) {
    const auto* kernel_def = kernel_defs[kernel_index];
    const auto& type_constraints = kernel_def->TypeConstraints();
    def_selected = kernel_def;

    for (size_t i = 0; i < node_input_num; ++i) {
      const auto input_type = infer_ctx.getInputType(i);

      // Guard against variadic parameter index
      const size_t schema_input_index = (i < inputs.size()) ? i : inputs.size() - 1;
      const auto& param = inputs[schema_input_index];
      const auto& input_name = param.GetName();
      if (input_type == nullptr) {
        if (param.GetOption() == ONNX_NAMESPACE::OpSchema::FormalParameterOption::Optional)
          continue;

        ORT_THROW("[CustomOP type inferencing error]: kernel Input: ", input_name,
                  " is absent, but not optional. Op : ", schema.Name());
      }

      is_variadic_input = (param.GetOption() == ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic);
      is_homogeneous_input = param.GetIsHomogeneous();

      if (!is_variadic_input || is_homogeneous_input) {
        auto hit = type_constraints.find(input_name);
        if (hit != type_constraints.end()) {
          const auto& types = hit->second;
          // For custom ops kernel constraints are never empty
          assert(!types.empty());
          if (!std::any_of(types.cbegin(), types.cend(),
                           [input_type](const DataTypeImpl* type) {
                             return type->IsCompatible(*input_type);
                           })) {
            def_selected = nullptr;
            output_propagate = 0;
            break;
          }

          // If we have multiple types possible from the constraints,
          // record the last type and use it to guess the output type if
          // output may have different types. Works well for symmetric single input/outputs
          // otherwise give up and let the user supply their own function
          if (types.size() > 1) {
            output_propagate = input_type->tensor_type().elem_type();
          }
        } else {
          ORT_THROW("[CustomOP type inferencing error]: no type constraint found for input: ",
                    input_name, " Op: ", schema.Name());
        }
      }
    }
  }

  if (def_selected == nullptr) {
    ORT_THROW("[CustomOP type inferencing error]: no kernel def matches node inputs for Op: ", schema.Name());
  }

  const auto& outputs = schema.outputs();
  const auto node_output_num = infer_ctx.getNumOutputs();
  const auto& selected_type_constraints = def_selected->TypeConstraints();

  for (size_t i = 0; i < node_output_num; ++i) {
    auto output_type = infer_ctx.getOutputType(i);
    // Account for variadic outputs
    const size_t schema_output_index = (i < outputs.size()) ? i : outputs.size() - 1;
    const auto& param = outputs[schema_output_index];
    const auto& output_name = param.GetName();

    const bool is_variadic_output = (param.GetOption() == ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic);
    const bool is_homogeneous = param.GetIsHomogeneous();

    // We give up on variadic non-homogeneous outputs
    // Let the user handle it in their inference function
    if (is_variadic_output && !is_homogeneous) {
      break;
    }

    auto hit = selected_type_constraints.find(output_name);
    if (hit != selected_type_constraints.end()) {
      const auto& types = hit->second;
      assert(!types.empty());

      if (types.size() == 1) {
        // Use the constraint type
        output_type->mutable_tensor_type()->set_elem_type(
            types[0]->GetTypeProto()->tensor_type().elem_type());
      } else if (!is_variadic_input || is_homogeneous_input) {
        // If not variadic or homogeneous, and there are multiple types possible, guess from the last input type
        // as this works for symmetric varied single input/outputs
        // otherwise give up and let the user supply their own function
        output_type->mutable_tensor_type()->set_elem_type(output_propagate);
      }
    } else {
      ORT_THROW("[CustomOP type inferencing error]: no type constraint found for output: ",
                output_name, " Op: ", schema.Name());
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
      // For example, two sessions using the same session_options should not add the same custom op domain
      // to the version map twice
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
      // schema.Name() is equal to ops[0]->GetName(ops[0]) and op->GetName(op) is the value
      // used as a key for dictionary domain_kernels, therefore name == schema.Name().
      schema_map.emplace(schema.Name(), schema);

      // This loops checks that all custom operators sharing the same name are compatible with the defined schema.
      for (const auto* op : ops) {
        // define kernel
        auto kernel_create_info = CreateKernelCreateInfo(domain->domain_, op);
        kernel_def_map[op->GetName(op)].push_back(kernel_create_info.kernel_def.get());
        ORT_RETURN_IF_ERROR(output->RegisterCustomKernel(kernel_create_info));
        // If IsCompatible returns false, then all custom operators named
        // 'op->GetName(op)' are not compatible among themselves.
        // They should have the same number of inputs and outputs, the same characteristics,
        // (optional, ...). Only the type can change.
        ORT_RETURN_IF_ERROR(IsCompatible(schema, op));
      }
    }

    std::vector<ONNX_NAMESPACE::OpSchema> schemas;
    for (auto& [name, schema] : schema_map) {
      schemas.push_back(schema);
      auto infer_fn = schemas.back().GetTypeAndShapeInferenceFunction();
      ONNX_NAMESPACE::InferenceFunction extended_infer_fn =
          [sch = schema, infer_fn = std::move(infer_fn),
           kernel_defs = std::move(kernel_def_map[name])](ONNX_NAMESPACE::InferenceContext& infer_ctx) {
            InferOutputTypes(sch, kernel_defs, infer_ctx);
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
#endif  // ENABLE_CUSTOM_OP_API
