// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_adapters.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/framework/allocator.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/environment.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/ort_mutex.h"
#include "core/common/string_helper.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
}
#endif

#ifdef ENABLE_TRAINING_APIS
#include "orttraining/training_api/include/onnxruntime_training_c_api.h"
#include "orttraining/training_api/ort_training_apis.h"
#endif

#ifdef USE_CANN
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/cann/cann_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CANN* TryGetProviderInfo_CANN();
}
#endif

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/dnnl/dnnl_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_Dnnl* TryGetProviderInfo_Dnnl();
}
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
const OrtDmlApi* GetOrtDmlApi(_In_ uint32_t version) NO_EXCEPTION;
#endif

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
#include "onnxruntime_extensions.h"
#endif
#if defined(_MSC_VER) && !defined(__clang__)
// The warning is: "Do not assign the result of an allocation or a function call with an owner<T> return value to a raw pointer, use owner<T> instead(i .11)."
// But this file is for C API. It can't use unique_ptr/shared_ptr in function signature.
#pragma warning(disable : 26400)
#endif
using namespace onnxruntime::logging;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::narrow;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel logging_level, _In_ const char* logid,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnv, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status, tp_options);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                    OrtLoggingLevel logging_level, _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status, tp_options);
  return ToOrtStatus(status);
  API_IMPL_END
}

// enable platform telemetry
ORT_API_STATUS_IMPL(OrtApis::EnableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().EnableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::DisableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().DisableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateEnvWithCustomLogLevel, _In_ OrtEnv* ort_env,
                    OrtLoggingLevel log_severity_level) {
  API_IMPL_BEGIN
  LoggingManager* default_logging_manager = ort_env->GetLoggingManager();
  int severity_level = static_cast<int>(log_severity_level);
  default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity_level));
  return nullptr;
  API_IMPL_END
}

ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len,
                                _Inout_ OrtAllocator* allocator, OrtValue& value) {
  TensorShape tensor_shape(shape, shape_len);
  AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  Tensor::InitOrtValue(ml_type, tensor_shape, std::move(alloc_ptr), value);
  return nullptr;
}

ORT_STATUS_PTR CreateTensorImplForSeq(MLDataType elem_type, const int64_t* shape, size_t shape_len, Tensor& out) {
  OrtAllocator* allocator;
  // TODO(pranav): what allocator should be used to create the tensor here?
  // for the sake of simplicity of the API using the default one here
  ORT_API_RETURN_IF_ERROR(OrtApis::GetAllocatorWithDefaultOptions(&allocator));
  AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  TensorShape tensor_shape(shape, shape_len);
  out = Tensor(elem_type, tensor_shape, std::move(alloc_ptr));
  return nullptr;
}

/**
 *
 * this function will create a copy of the allocator info
 */
ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                                void* p_data, size_t p_data_len, OrtValue& ort_value) {
  TensorShape tensor_shape(shape, shape_len);
  if (std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(), [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }

  auto elem_count = narrow<size_t>(tensor_shape.Size());
  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(ml_type->Size(), elem_count, &size_to_allocate)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "size overflow");
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }
  Tensor::InitOrtValue(ml_type, tensor_shape, p_data, *info, ort_value);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();
  auto value = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(ml_type, shape, shape_len, info, p_data, p_data_len, *value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();
  auto value = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(ml_type, shape, shape_len, allocator, *value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSparseTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* dense_shape,
                    size_t dense_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto sparse_tensor_type = DataTypeImpl::SparseTensorTypeFromONNXEnum(type);
  auto element_type = sparse_tensor_type->GetElementType();
  assert(element_type->AsPrimitiveDataType() != nullptr);
  TensorShape shape(dense_shape, dense_shape_len);
  if (std::any_of(shape.GetDims().begin(), shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }

  auto alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  auto value = std::make_unique<OrtValue>();
  SparseTensor::InitOrtValue(element_type, shape, std::move(alloc_ptr), *value);
  *out = value.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(dense_shape);
  ORT_UNUSED_PARAMETER(dense_shape_len);
  ORT_UNUSED_PARAMETER(type);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

namespace {
#if !defined(DISABLE_SPARSE_TENSORS)
std::unique_ptr<IDataTransfer> GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) {
  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::CPU) {
    return std::make_unique<CPUDataTransfer>();
  }
#ifdef USE_CUDA
  if (src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU) {
    if (auto* provider_info = TryGetProviderInfo_CUDA()) {
      return provider_info->CreateGPUDataTransfer();
    }
  }
#endif
  ORT_THROW("Not able to find appropriate IDataTransfer to copy sparse data");
}

SparseTensor& ValidateFillInputArgs(OrtValue* v, const TensorShape& values_shape, const OrtMemoryInfo* data_mem_info) {
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*v);
  if (sparse_tensor.IsDataTypeString()) {
    if ((data_mem_info->device.Type() != OrtDevice::CPU) || sparse_tensor.Location().device.Type() != OrtDevice::CPU) {
      ORT_THROW("Strings can only reside in CPU memory");
    }
  }
  if (std::any_of(values_shape.GetDims().begin(), values_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    ORT_THROW("tried Filling sparse tensor with negative value in values shape");
  }

  return sparse_tensor;
}

union PtrConvert {
  explicit PtrConvert(const void* p_p) : p(p_p) {}
  const void* p;
  const char** strings;
};

#endif  // !defined(DISABLE_SPARSE_TENSORS)
}  // namespace

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorCoo, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_data, size_t indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);

  auto values_size = narrow<size_t>(values_t_shape.Size());
  auto indices_span = gsl::make_span(indices_data, indices_num);

  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCooStrings(values_size, conv.strings, indices_span));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCooData(*data_transfer, *data_mem_info, values_size,
                                                 values, indices_span));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(indices_data);
  ORT_UNUSED_PARAMETER(indices_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorCsr, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* inner_indices_data, size_t inner_indices_num,
                    _In_ const int64_t* outer_indices_data, size_t outer_indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);
  auto values_size = narrow<size_t>(values_t_shape.Size());

  auto inner_indices_span = gsl::make_span(inner_indices_data, inner_indices_num);
  auto outer_indices_span = gsl::make_span(outer_indices_data, outer_indices_num);
  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCsrStrings(values_size, conv.strings, inner_indices_span, outer_indices_span));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCsrData(*data_transfer, *data_mem_info, values_size,
                                                 values, inner_indices_span, outer_indices_span));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(inner_indices_data);
  ORT_UNUSED_PARAMETER(inner_indices_num);
  ORT_UNUSED_PARAMETER(outer_indices_data);
  ORT_UNUSED_PARAMETER(outer_indices_num);
  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorBlockSparse, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_shape_data, size_t indices_shape_len,
                    _In_ const int32_t* indices_data) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);

  TensorShape indices_t_shape(indices_shape_data, indices_shape_len);
  if (std::any_of(indices_t_shape.GetDims().begin(), indices_t_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    ORT_THROW("tried Filling sparse tensor with negative value in block sparse indices shape");
  }

  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeBlockSparseStrings(values_t_shape, conv.strings, indices_t_shape, indices_data));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeBlockSparseData(*data_transfer, *data_mem_info, values_t_shape,
                                                         values, indices_t_shape, indices_data));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(indices_shape_data);
  ORT_UNUSED_PARAMETER(indices_shape_len);
  ORT_UNUSED_PARAMETER(indices_data);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSparseTensorWithValuesAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                    _In_ const int64_t* dense_shape, size_t dense_shape_len,
                    _In_ const int64_t* values_shape, size_t values_shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto sparse_tensor_type = DataTypeImpl::SparseTensorTypeFromONNXEnum(type);
  auto element_type = sparse_tensor_type->GetElementType();
  assert(element_type->AsPrimitiveDataType() != nullptr);
  if (utils::IsDataTypeString(element_type)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Can not use strings in pre-allocated memory."
                                 " Use CreateSparseTensorAsOrtValue() to allocate memory inside and copy");
  }
  TensorShape tensor_dense_shape(dense_shape, dense_shape_len);
  TensorShape tensor_values_shape(values_shape, values_shape_len);
  if (std::any_of(tensor_values_shape.GetDims().begin(), tensor_values_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }
  auto value = std::make_unique<OrtValue>();
  SparseTensor::InitOrtValue(element_type, tensor_dense_shape, tensor_values_shape, p_data, *info, *value);
  *out = value.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(info);
  ORT_UNUSED_PARAMETER(p_data);
  ORT_UNUSED_PARAMETER(dense_shape);
  ORT_UNUSED_PARAMETER(dense_shape_len);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(type);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseCooIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* indices_data, size_t indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto v = reinterpret_cast<::OrtValue*>(ort_value);
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*v);
  auto indices_span = (indices_num == 0 || indices_data == nullptr)
                          ? gsl::span<int64_t>()
                          : gsl::make_span(indices_data, indices_num);

  ORT_THROW_IF_ERROR(sparse_tensor.UseCooIndices(indices_span));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(indices_data);
  ORT_UNUSED_PARAMETER(indices_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseCsrIndices, _Inout_ OrtValue* ort_value,
                    _Inout_ int64_t* inner_data, size_t inner_num,
                    _Inout_ int64_t* outer_data, size_t outer_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  auto inner_span = (inner_num == 0 || inner_data == nullptr)
                        ? gsl::span<int64_t>()
                        : gsl::make_span(inner_data, inner_num);
  auto outer_span = (outer_num == 0 || outer_data == nullptr)
                        ? gsl::span<int64_t>()
                        : gsl::make_span(outer_data, outer_num);
  ORT_THROW_IF_ERROR(sparse_tensor.UseCsrIndices(inner_span, outer_span));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(inner_data);
  ORT_UNUSED_PARAMETER(inner_num);
  ORT_UNUSED_PARAMETER(outer_data);
  ORT_UNUSED_PARAMETER(outer_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseBlockSparseIndices, _Inout_ OrtValue* ort_value, const int64_t* indices_shape,
                    size_t indices_shape_len, _Inout_ int32_t* indices_data) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  TensorShape ind_shape(indices_shape, indices_shape_len);
  ORT_THROW_IF_ERROR(sparse_tensor.UseBlockSparseIndices(ind_shape, indices_data));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(indices_shape);
  ORT_UNUSED_PARAMETER(indices_shape_len);
  ORT_UNUSED_PARAMETER(indices_data);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorFormat, _In_ const OrtValue* ort_value, _Out_ enum OrtSparseFormat* out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto v = reinterpret_cast<const ::OrtValue*>(ort_value);
  if (!v->IsAllocated()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "the ort_value must contain a constructed tensor");
  }
  const auto& sparse_tensor = v->Get<SparseTensor>();
  *out = static_cast<OrtSparseFormat>(sparse_tensor.Format());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorValues, _In_ const OrtValue* ort_value, _Outptr_ const void** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  const auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  if (sparse_tensor.IsDataTypeString()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Use GetStringTensor*() API to retrieve strings");
  }
  const auto& values = sparse_tensor.Values();
  *out = values.DataRaw();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out) {
  API_IMPL_BEGIN
  auto custom_op_domain = std::make_unique<OrtCustomOpDomain>();
  custom_op_domain->domain_ = domain;
  *out = custom_op_domain.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op) {
  API_IMPL_BEGIN
  custom_op_domain->custom_ops_.emplace_back(op);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddCustomOpDomain, _Inout_ OrtSessionOptions* options,
                    _In_ OrtCustomOpDomain* custom_op_domain) {
  API_IMPL_BEGIN
  options->custom_op_domains_.emplace_back(custom_op_domain);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, _Outptr_ void** library_handle) {
  API_IMPL_BEGIN

  auto path_str = ToPathString(library_path);
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().LoadDynamicLibrary(path_str, false, library_handle));
  if (!*library_handle)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

  RegisterCustomOpsFn RegisterCustomOps;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().GetSymbolFromLibrary(*library_handle, "RegisterCustomOps",
                                                                      (void**)&RegisterCustomOps));
  if (!RegisterCustomOps)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

  return RegisterCustomOps(options, OrtGetApiBase());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary_V2, _Inout_ OrtSessionOptions* options,
                    _In_ const ORTCHAR_T* library_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  ORT_API_RETURN_IF_STATUS_NOT_OK(options->RegisterCustomOpsLibrary(library_name));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(library_name);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Custom operator libraries are not supported in this build");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsUsingFunction, _Inout_ OrtSessionOptions* options,
                    _In_ const char* registration_func_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (!registration_func_name) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "RegisterCustomOpsUsingFunction: Registration function name must be specified.");
  }

  RegisterCustomOpsFn RegisterCustomOps;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().GetSymbolFromLibrary(nullptr, registration_func_name,
                                                                      (void**)&RegisterCustomOps));
  if (!RegisterCustomOps) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "RegisterCustomOpsUsingFunction: Registration function was not found");
  }

  return RegisterCustomOps(options, OrtGetApiBase());
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(registration_func_name);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Custom operator libraries are not supported in this build");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EnableOrtCustomOps, _Inout_ OrtSessionOptions* options) {
  API_IMPL_BEGIN

  if (options) {
#ifdef ENABLE_EXTENSION_CUSTOM_OPS
    return RegisterCustomOps(options, OrtGetApiBase());
#else
    return OrtApis::CreateStatus(ORT_FAIL, "EnableOrtCustomOps: Custom operators in onnxruntime-extensions are not enabled");
#endif
  }
  return nullptr;

  API_IMPL_END
}

namespace {
// provider either model_path, or modal_data + model_data_length.
static ORT_STATUS_PTR CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                                _In_ const OrtEnv* env,
                                                _In_opt_z_ const ORTCHAR_T* model_path,
                                                _In_opt_ const void* model_data,
                                                size_t model_data_length,

                                                std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  // quick check here to decide load path. InferenceSession will provide error message for invalid values.
  // TODO: Could move to a helper
  const Env& os_env = Env::Default();  // OS environment (!= ORT environment)
  bool load_config_from_model =
      os_env.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar) == "1";

  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    if (model_path != nullptr) {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Add custom domains
  if (options && !options->custom_op_domains_.empty()) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddCustomOpDomains(options->custom_op_domains_));
  }
#endif

  // Finish load
  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load());
#endif
  } else {
    if (model_path != nullptr) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_path));
    } else {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_data, static_cast<int>(model_data_length)));
    }
  }

  return nullptr;
}

static ORT_STATUS_PTR InitializeSession(_In_ const OrtSessionOptions* options,
                                        _In_ std::unique_ptr<::onnxruntime::InferenceSession>& sess,
                                        _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr) {
  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      provider_list.push_back(std::move(provider));
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Initialize());

  return nullptr;
}

static ORT_STATUS_PTR EvictSession(_In_ onnxruntime::InferenceSession* session) {
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->Evict());
  return nullptr;
}

}  // namespace

ORT_API_STATUS_IMPL(OrtApis::CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data,
                    size_t model_data_length, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data, model_data_length, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EvictSession, _In_ OrtSession* session) {
  API_IMPL_BEGIN

  OrtStatus* status = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(::EvictSession(reinterpret_cast<onnxruntime::InferenceSession*>(session)));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);

  gsl::span<const char* const> input_names_span(input_names, input_len);
  gsl::span<const OrtValue* const> input_span(input, input_len);
  gsl::span<const char* const> output_name_span(output_names, output_names_len);
  gsl::span<OrtValue*> output_span(output, output_names_len);

  Status status;
  if (run_options) {
    status = session->Run(*run_options,
                          input_names_span,
                          input_span,
                          output_name_span,
                          output_span);
  } else {
    const RunOptions default_run_options;
    status = session->Run(default_run_options,
                          input_names_span,
                          input_span,
                          output_name_span,
                          output_span);
  }
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunAsync, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output,
                    _In_ RunAsyncCallbackFn run_async_callback, _In_opt_ void* user_data) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);

  gsl::span<const char* const> input_names_span(input_names, input_len);
  gsl::span<const OrtValue* const> input_span(input, input_len);
  gsl::span<const char* const> output_name_span(output_names, output_names_len);
  gsl::span<OrtValue*> output_span(output, output_names_len);

  return ToOrtStatus(session->RunAsync(run_options,
                                       input_names_span,
                                       input_span,
                                       output_name_span,
                                       output_span,
                                       run_async_callback,
                                       user_data));
  API_IMPL_END
}

struct OrtIoBinding {
  std::unique_ptr<::onnxruntime::IOBinding> binding_;
  explicit OrtIoBinding(std::unique_ptr<::onnxruntime::IOBinding>&& binding) : binding_(std::move(binding)) {}
  OrtIoBinding(const OrtIoBinding&) = delete;
  OrtIoBinding& operator=(const OrtIoBinding&) = delete;
};

ORT_API_STATUS_IMPL(OrtApis::RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options,
                    _In_ const OrtIoBinding* binding_ptr) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions default_run_options;
    status = session->Run(default_run_options, *binding_ptr->binding_);
  } else {
    status = session->Run(*run_options, *binding_ptr->binding_);
  }
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::unique_ptr<::onnxruntime::IOBinding> binding;
  auto status = session->NewIOBinding(&binding);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  *out = std::make_unique<OrtIoBinding>(std::move(binding)).release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseIoBinding, _Frees_ptr_opt_ OrtIoBinding* binding_ptr) {
  delete binding_ptr;
}

ORT_API_STATUS_IMPL(OrtApis::BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindInput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* mem_info_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, mem_info_ptr->device);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Out_ char** buffer, _Outptr_result_maybenull_ size_t** lengths, _Out_ size_t* count) {
  API_IMPL_BEGIN
  const auto& output_names = binding_ptr->binding_->GetOutputNames();
  if (output_names.empty()) {
    *buffer = nullptr;
    *lengths = nullptr;
    *count = 0U;
    return nullptr;
  }

  IAllocatorUniquePtr<size_t> lengths_alloc(reinterpret_cast<size_t*>(allocator->Alloc(allocator, output_names.size() * sizeof(size_t))),
                                            [allocator](size_t* p) { if(p) allocator->Free(allocator, p); });

  if (!lengths_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "lengths allocation failed");
  }

  size_t total_len = 0;
  auto* len_ptr = lengths_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    total_len += sz;
    *len_ptr++ = sz;
  }

  IAllocatorUniquePtr<char> buffer_alloc(reinterpret_cast<char*>(allocator->Alloc(allocator, total_len * sizeof(char))),
                                         [allocator](char* p) { if(p) allocator->Free(allocator, p); });

  if (!buffer_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "string buffer allocation failed");
  }

  char* buf_ptr = buffer_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    memcpy(buf_ptr, n.data(), sz);
    buf_ptr += sz;
  }

  *buffer = buffer_alloc.release();
  *lengths = lengths_alloc.release();
  *count = output_names.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Outptr_result_maybenull_ OrtValue*** output, _Out_ size_t* output_count) {
  API_IMPL_BEGIN
  const auto& outputs = binding_ptr->binding_->GetOutputs();
  if (outputs.empty()) {
    *output = nullptr;
    *output_count = 0U;
    return nullptr;
  }

  // Used to destroy and de-allocate on exception
  IAllocatorUniquePtr<OrtValue*> ortvalues_alloc(reinterpret_cast<OrtValue**>(allocator->Alloc(allocator, outputs.size() * sizeof(OrtValue*))),
                                                 [allocator](OrtValue** p) { if (p) allocator->Free(allocator, p); });
  if (!ortvalues_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "Output buffer allocation failed");
  }

  InlinedVector<std::unique_ptr<OrtValue>> value_dups;
  value_dups.reserve(outputs.size());

  for (const auto& out_value : outputs) {
    value_dups.push_back(std::make_unique<OrtValue>(out_value));
  }

  // The rest is noexcept
  OrtValue** out_ptr = ortvalues_alloc.get();
  for (auto& v : value_dups) {
    *out_ptr++ = v.release();
  }

  *output = ortvalues_alloc.release();
  *output_count = outputs.size();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ClearBoundInputs, _Inout_ OrtIoBinding* binding_ptr) {
  binding_ptr->binding_->ClearInputs();
}

ORT_API(void, OrtApis::ClearBoundOutputs, _Inout_ OrtIoBinding* binding_ptr) {
  binding_ptr->binding_->ClearOutputs();
}

ORT_API_STATUS_IMPL(OrtApis::SynchronizeBoundInputs, _Inout_ OrtIoBinding* binding_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->SynchronizeInputs();
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SynchronizeBoundOutputs, _Inout_ OrtIoBinding* binding_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->SynchronizeOutputs();
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::IsTensor, _In_ const OrtValue* value, _Out_ int* out) {
  auto v = reinterpret_cast<const ::OrtValue*>(value);
  *out = v->IsTensor() ? 1 : 0;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::HasValue, _In_ const OrtValue* value, _Out_ int* out) {
  auto v = reinterpret_cast<const ::OrtValue*>(value);
  *out = v->IsAllocated() ? 1 : 0;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::IsSparseTensor, _In_ const OrtValue* value, _Out_ int* out) {
#if !defined(DISABLE_SPARSE_TENSORS)
  auto v = reinterpret_cast<const ::OrtValue*>(value);
  *out = v->IsSparseTensor() ? 1 : 0;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(value);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  // Uncomment when WinML fixed their code
  // if (tensor->IsDataTypeString()) {
  //  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "this API does not support strings");
  //}
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len != len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array doesn't equal tensor size");
  }
  for (size_t i = 0; i != len; ++i) {
    // allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  const auto len = static_cast<size_t>(tensor->Shape().Size());
  if (index >= len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "element index is out of bounds");
  }

  dst[index] = s;

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetResizedStringTensorElementBuffer, _Inout_ OrtValue* value,
                    _In_ size_t index, _In_ size_t length_in_bytes, _Inout_ char** buffer) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  const auto len = static_cast<size_t>(tensor->Shape().Size());

  if (index >= len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "element index is out of bounds");
  }

  auto& s = dst[index];
  s.resize(length_in_bytes);
  *buffer = s.data();
  return nullptr;
  API_IMPL_END
}

namespace {

OrtStatusPtr GetTensorStringSpan(const ::OrtValue& v, gsl::span<const std::string>& span) {
  if (!v.IsAllocated()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtValue should contain a Tensor or a Sparse Tensor");
  }
  gsl::span<const std::string> str_span;
  int64_t items = 0;
  // Data type will be enforced on DataAsSpan() call.
  if (v.IsTensor()) {
    const auto& tensor = v.Get<onnxruntime::Tensor>();
    items = tensor.Shape().Size();
    if (items >= 0) {
      str_span = tensor.DataAsSpan<std::string>();
    }
  }
#if !defined(DISABLE_SPARSE_TENSORS)
  else if (v.IsSparseTensor()) {
    const auto& sparse_tensor = v.Get<SparseTensor>();
    if (sparse_tensor.Format() == onnxruntime::SparseFormat::kUndefined) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Sparse Tensor does not contain sparse data");
    }
    items = sparse_tensor.Values().Shape().Size();
    if (items >= 0) {
      str_span = sparse_tensor.Values().DataAsSpan<std::string>();
    }
  }
#endif
  else {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "This API supports Tensors or SparseTensors");
  }

  if (items < 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "shape is invalid");
  }
  span = str_span;
  return nullptr;
}
}  // namespace

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* out) {
  API_IMPL_BEGIN
  gsl::span<const std::string> str_span;
  if (auto* status = GetTensorStringSpan(*value, str_span)) {
    return status;
  }

  size_t ret = 0;
  for (const auto& s : str_span) {
    ret += s.size();
  }

  *out = ret;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out) {
  API_IMPL_BEGIN
  gsl::span<const std::string> str_span;
  if (auto* status = GetTensorStringSpan(*value, str_span)) {
    return status;
  }

  if (index < str_span.size()) {
    *out = str_span[index].size();
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "index is out of bounds");
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                    size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len) {
  API_IMPL_BEGIN

  gsl::span<const std::string> str_span;
  if (auto* status = GetTensorStringSpan(*value, str_span)) {
    return status;
  }

  if (offsets_len != str_span.size()) {
    return OrtApis::CreateStatus(ORT_FAIL, "offsets buffer is not equal to tensor size");
  }

  size_t total_size = 0;
  for (const auto& str : str_span) {
    total_size += str.size();
  }

  if (s_len < total_size) {
    return OrtApis::CreateStatus(ORT_FAIL, "output buffer is too small. Use GetStringTensorDataLength.");
  }

  size_t f = 0;
  char* p = static_cast<char*>(s);
  for (const auto& str : str_span) {
    memcpy(p, str.data(), str.size());
    p += str.size();
    *offsets++ = f;
    f += str.size();
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorElement, _In_ const OrtValue* value,
                    size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s) {
  API_IMPL_BEGIN
  gsl::span<const std::string> str_span;
  if (auto* status = GetTensorStringSpan(*value, str_span)) {
    return status;
  }

  if (index < str_span.size()) {
    const auto& str = str_span[index];
    if (s_len < str.size()) {
      return OrtApis::CreateStatus(ORT_FAIL, "buffer size is too small for string element");
    }
    memcpy(s, str.data(), str.size());
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "element index is out of bounds");
  }
  return nullptr;
  API_IMPL_END
}

#define ORT_C_API_RETURN_IF_ERROR(expr)                 \
  do {                                                  \
    auto _status = (expr);                              \
    if ((!_status.IsOK())) return ToOrtStatus(_status); \
  } while (0)

#define DEFINE_RELEASE_ORT_OBJECT_FUNCTION(INPUT_TYPE, REAL_TYPE)                       \
  ORT_API(void, OrtApis::Release##INPUT_TYPE, _Frees_ptr_opt_ Ort##INPUT_TYPE* value) { \
    delete reinterpret_cast<REAL_TYPE*>(value);                                         \
  }

using DefListResult = std::pair<Status, const InputDefList*>;
using GetDefListFn = DefListResult (*)(const ::onnxruntime::InferenceSession*);
const auto get_inputs_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetModelInputs(); };
const auto get_outputs_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetModelOutputs(); };
const auto get_overridable_initializers_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetOverridableInitializers(); };

static ORT_STATUS_PTR GetNodeDefListCountHelper(const OrtSession* sess, GetDefListFn get_fn, size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_inputs_fn, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_outputs_fn, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_overridable_initializers_fn, out);
}

static ORT_STATUS_PTR GetNodeDefTypeInfoHelper(const OrtSession* sess, GetDefListFn get_fn, size_t index,
                                               _Outptr_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second->size() <= index)
    return OrtApis::CreateStatus(ORT_FAIL, "out of index");
  const ONNX_NAMESPACE::TypeProto* type_proto = (*p.second)[index]->TypeAsProto();
  auto type_info = OrtTypeInfo::FromTypeProto(*type_proto);
  *out = type_info.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_inputs_fn, index, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_outputs_fn, index, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_overridable_initializers_fn, index, out);
}

char* onnxruntime::StrDup(const std::string& str, OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static ORT_STATUS_PTR GetNodeDefNameImpl(_In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                                         GetDefListFn get_fn, _Outptr_ char** output) {
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second == nullptr)
    return OrtApis::CreateStatus(ORT_FAIL, "internal error");
  const InputDefList& defs = *p.second;
  if (index >= defs.size())
    return OrtApis::CreateStatus(ORT_FAIL, "index out of range");
  *output = StrDup(defs[index]->Name(), allocator);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  auto profile_file_name = session->EndProfiling();
  *out = StrDup(profile_file_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetModelMetadata, _In_ const OrtSession* sess,
                    _Outptr_ OrtModelMetadata** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto p = session->GetModelMetadata();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = reinterpret_cast<OrtModelMetadata*>(std::make_unique<ModelMetadata>(*p.second).release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetProducerName,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto producer_name = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->producer_name;
  *value = StrDup(producer_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetGraphName,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto graph_name = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->graph_name;
  *value = StrDup(graph_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetDomain,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto domain = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->domain;
  *value = StrDup(domain, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetDescription,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto description = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->description;
  *value = StrDup(description, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetGraphDescription,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto description = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->graph_description;
  *value = StrDup(description, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value) {
  API_IMPL_BEGIN
  auto custom_metadata_map =
      reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->custom_metadata_map;

  std::string temp(key);

  auto iter = custom_metadata_map.find(temp);

  if (iter == custom_metadata_map.end()) {
    *value = nullptr;
  } else {
    *value = StrDup(iter->second, allocator);
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetCustomMetadataMapKeys,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys) {
  API_IMPL_BEGIN
  const auto& custom_metadata_map =
      reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->custom_metadata_map;

  auto count = custom_metadata_map.size();
  if (count == 0) {
    *keys = nullptr;
  } else {
    // To guard against overflow in the next step where we compute bytes to allocate
    SafeInt<size_t> alloc_count(count);

    InlinedVector<Ort::AllocatedStringPtr> string_holders;
    string_holders.reserve(count);

    auto deletor = Ort::detail::AllocatedFree(allocator);
    // alloc_count * sizeof(...) will throw if there was an overflow which will be caught in API_IMPL_END
    // and be returned to the user as a status
    char** p = reinterpret_cast<char**>(allocator->Alloc(allocator, alloc_count * sizeof(char*)));
    assert(p != nullptr);

    // StrDup may throw
    std::unique_ptr<void, decltype(deletor)> array_guard(p, deletor);

    int64_t i = 0;
    for (const auto& e : custom_metadata_map) {
      auto* s = StrDup(e.first, allocator);
      string_holders.push_back(Ort::AllocatedStringPtr(s, deletor));
      p[i++] = s;
    }

    for (auto& s : string_holders) {
      s.release();
    }

    *keys = p;
    array_guard.release();
  }

  *num_keys = static_cast<int64_t>(count);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetVersion,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Out_ int64_t* value) {
  API_IMPL_BEGIN
  *value = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->version;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_inputs_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_outputs_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_overridable_initializers_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out) {
  API_IMPL_BEGIN
  *out = ptr->Alloc(ptr, size);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorFree, _Inout_ OrtAllocator* ptr, void* p) {
  API_IMPL_BEGIN
  ptr->Free(ptr, p);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out) {
  API_IMPL_BEGIN
  *out = ptr->Info(ptr);
  return nullptr;
  API_IMPL_END
}

template <typename T>
ORT_STATUS_PTR OrtGetNumSequenceElements(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<T>();
  *out = data.size();
  return nullptr;
}

#if !defined(DISABLE_ML_OPS)
static constexpr int NUM_MAP_INDICES = 2;
#endif

static ORT_STATUS_PTR OrtGetValueCountImpl(const OrtValue* value, size_t* out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    *out = NUM_MAP_INDICES;
    return nullptr;
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    // Note: keep these in sync with the registered types in data_types.h
    if (value->IsTensorSequence()) {
      *out = value->Get<TensorSeq>().Size();
      return nullptr;
    } else {
#if !defined(DISABLE_ML_OPS)
      utils::ContainerChecker c_checker(value->Type());
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        return OrtGetNumSequenceElements<VectorMapStringToFloat>(value, out);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        return OrtGetNumSequenceElements<VectorMapInt64ToFloat>(value, out);
      } else {
        return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
      }
#else
      return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
    }
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out) {
  API_IMPL_BEGIN
  return OrtGetValueCountImpl(value, out);
  API_IMPL_END
}

namespace c_api_internal {

#if !defined(DISABLE_ML_OPS)
///////////////////
// OrtGetValueImplSeqOfMap
template <typename T>
static ORT_STATUS_PTR OrtGetValueImplSeqOfMap(const OrtValue* p_ml_value, int index, _Outptr_ OrtValue** out) {
  using TKey = typename T::value_type::key_type;
  using TVal = typename T::value_type::mapped_type;
  using MapType = std::map<TKey, TVal>;
  auto& data_vec = p_ml_value->Get<T>();
  auto& data_elem = data_vec.at(index);
  auto copy_data_elem = std::make_unique<MapType>(data_elem);
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<MapType>();
  value->Init(copy_data_elem.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}
#endif

ORT_STATUS_PTR PopulateTensorWithData(Tensor& tensor, bool is_string, _In_ const void* data_elem, size_t num_elems,
                                      size_t elem_size) {
  auto len = narrow<size_t>(tensor.Shape().Size());
  if (num_elems < len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array is too short");
  }
  if (!is_string) {
    memcpy(tensor.MutableDataRaw(), data_elem, elem_size * num_elems);
  } else {
    const std::string* strings = reinterpret_cast<const std::string*>(data_elem);
    auto str_span = gsl::make_span(strings, num_elems);
    auto* dst = tensor.MutableData<std::string>();
    std::copy(str_span.begin(), str_span.end(), dst);
  }
  return nullptr;
}

ORT_STATUS_PTR CreateTensorAndPopulate(MLDataType element_type, const int64_t* shape, size_t shape_len,
                                       const void* data, size_t num_elements, _Inout_ OrtAllocator* allocator, OrtValue& result) {
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(element_type, shape, shape_len, allocator, result));
  ORT_API_RETURN_IF_ERROR(PopulateTensorWithData(*result.GetMutable<Tensor>(), utils::IsDataTypeString(element_type),
                                                 data, num_elements, element_type->Size()));
  return nullptr;
}

}  // namespace c_api_internal
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6101)
#endif

static ORT_STATUS_PTR OrtGetValueImplSeqOfTensors(_In_ const OrtValue* p_ml_value, int index, _Inout_ OrtAllocator* allocator,
                                                  _Outptr_ OrtValue** out) {
  const auto& data = p_ml_value->Get<TensorSeq>();
  const auto& one_tensor = data.Get(index);
  const auto& tensor_shape = one_tensor.Shape();
  auto result = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(c_api_internal::CreateTensorAndPopulate(one_tensor.DataType(), tensor_shape.GetDims().data(),
                                                                  tensor_shape.NumDimensions(), one_tensor.DataRaw(),
                                                                  narrow<size_t>(one_tensor.Shape().Size()),
                                                                  allocator, *result));
  *out = result.release();
  return nullptr;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static ORT_STATUS_PTR OrtGetValueImplSeq(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                         _Outptr_ OrtValue** out) {
  // Note: keep these in sync with the registered types in data_types.h
  if (value->IsTensorSequence()) {
    return OrtGetValueImplSeqOfTensors(value, index, allocator, out);
  } else {
#if !defined(DISABLE_ML_OPS)
    utils::ContainerChecker c_checker(value->Type());
    if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
      return c_api_internal::OrtGetValueImplSeqOfMap<VectorMapStringToFloat>(value, index, out);
    } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
      return c_api_internal::OrtGetValueImplSeqOfMap<VectorMapInt64ToFloat>(value, index, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
}

#if !defined(DISABLE_ML_OPS)
template <typename T>
static ORT_STATUS_PTR OrtGetValueImplMapHelper(_In_ const OrtValue* p_ml_value, int index,
                                               _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out) {
  using namespace onnxruntime::utils;
  using TKey = typename T::key_type;
  using TVal = typename T::mapped_type;
  auto& data = p_ml_value->Get<T>();
  int64_t num_kv_pairs = data.size();
#if defined(_WIN32) && !defined(_M_AMD64)
  ORT_ENFORCE(static_cast<uint64_t>(num_kv_pairs) < std::numeric_limits<size_t>::max());
#endif
  const std::vector<int64_t> dims{num_kv_pairs};
  auto result = std::make_unique<OrtValue>();
  std::vector<TKey> vec_keys;
  std::vector<TVal> vec_vals;
  const void* data_ptr;
  size_t data_size;
  MLDataType element_type;
  switch (index) {
    case 0: {  // user is requesting keys
      element_type = DataTypeImpl::TensorTypeFromONNXEnum(GetONNXTensorElementDataType<TKey>())->GetElementType();
      vec_keys.reserve(static_cast<size_t>(num_kv_pairs));
      std::transform(data.cbegin(), data.cend(), std::back_inserter(vec_keys), [](const auto& k) { return k.first; });
      data_ptr = vec_keys.data();
      data_size = vec_keys.size();
    } break;
    case 1: {  // user is requesting values
      element_type = DataTypeImpl::TensorTypeFromONNXEnum(GetONNXTensorElementDataType<TVal>())->GetElementType();
      vec_vals.reserve(static_cast<size_t>(num_kv_pairs));
      std::transform(data.cbegin(), data.cend(), std::back_inserter(vec_vals), [](const auto& k) { return k.second; });
      data_ptr = vec_vals.data();
      data_size = vec_vals.size();
    } break;
    default:
      return OrtApis::CreateStatus(ORT_FAIL, "Invalid index requested for map type.");
  }
  ORT_API_RETURN_IF_ERROR(c_api_internal::CreateTensorAndPopulate(element_type, dims.data(), dims.size(), data_ptr,
                                                                  data_size, allocator, *result));
  *out = result.release();
  return nullptr;
}

static ORT_STATUS_PTR OrtGetValueImplMap(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                         _Outptr_ OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  utils::ContainerChecker c_checker(type);
  if (c_checker.IsMap()) {
    if (c_checker.IsMapOf<std::string, std::string>()) {
      return OrtGetValueImplMapHelper<MapStringToString>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, int64_t>()) {
      return OrtGetValueImplMapHelper<MapStringToInt64>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, float>()) {
      return OrtGetValueImplMapHelper<MapStringToFloat>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, double>()) {
      return OrtGetValueImplMapHelper<MapStringToDouble>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, std::string>()) {
      return OrtGetValueImplMapHelper<MapInt64ToString>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, int64_t>()) {
      return OrtGetValueImplMapHelper<MapInt64ToInt64>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, float>()) {
      return OrtGetValueImplMapHelper<MapInt64ToFloat>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, double>()) {
      return OrtGetValueImplMapHelper<MapInt64ToDouble>(p_ml_value, index, allocator, out);
    }
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
}
#endif

static ORT_STATUS_PTR OrtGetValueImpl(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                      _Outptr_ OrtValue** out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    return OrtGetValueImplMap(value, index, allocator, out);
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtGetValueImplSeq(value, index, allocator, out);
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  return OrtGetValueImpl(value, index, allocator, out);
  API_IMPL_END
}

///////////////////
// OrtCreateValue

#if !defined(DISABLE_ML_OPS)
template <typename T>
static ORT_STATUS_PTR OrtCreateValueImplSeqHelperMap(const OrtValue* const* in, size_t num_values,
                                                     _Outptr_ OrtValue** out) {
  using SeqType = std::vector<T>;
  auto seq_ptr = std::make_unique<SeqType>();
  seq_ptr->reserve(num_values);
  for (size_t idx = 0; idx < num_values; ++idx) {
    auto& m = reinterpret_cast<const OrtValue*>(in[idx])->Get<T>();
    seq_ptr->push_back(m);
  }
  // create OrtValue with this vector
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<SeqType>();
  value->Init(seq_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}
#endif

static ORT_STATUS_PTR OrtCreateValueImplSeqHelper(const OrtValue* const* in, size_t num_values,
                                                  _Outptr_ OrtValue** out) {
  using namespace c_api_internal;
  auto dtype = in[0]->Get<Tensor>().DataType();
  auto seq_ptr = std::make_unique<TensorSeq>(dtype);
  seq_ptr->Reserve(num_values);

  for (size_t idx = 0; idx < num_values; ++idx) {
    ORT_ENFORCE(in[idx]->IsTensor(), "Expecting all elements to be tensors. Got: ", DataTypeImpl::ToString(in[idx]->Type()));
    auto tensor_elem_type = in[idx]->Get<Tensor>().DataType();

    // sequences must have tensors of the same data type
    if (tensor_elem_type != dtype) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "Sequences must have tensors of the same data type. There was at least one tensor in the input that was different.");
    }

    seq_ptr->Add(*in[idx]);
  }

  // create OrtValue with this vector
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<TensorSeq>();
  value->Init(seq_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

static ORT_STATUS_PTR OrtCreateValueImplSeq(_In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                                            _Outptr_ OrtValue** out) {
  // We only support limited sequence types. For the sake of simplicity the type of the first
  // OrtValue* in OrtValue** will determine the type of the vector used to create the output OrtValue
  // this type should be either a tensor of limited types or map of limited types
  const OrtValue* ovfirst = in[0];
  ONNXType first_value_type;
  if (auto status = OrtApis::GetValueType(ovfirst, &first_value_type))
    return status;
  // in onnxruntime type registrations we can support only a fixed vector types
  // this check ensures that the input conforms to that
  if (!(first_value_type == ONNX_TYPE_TENSOR || first_value_type == ONNX_TYPE_MAP)) {
    return OrtApis::CreateStatus(ORT_FAIL, "Each element of the sequence should be either tensor or map.");
  }
  // check if all OrtValues in the input array are of the same type
  // this is because even though the ONNX spec and this API spec supports heterogenous sequences,
  // only a fixed types are registered in onnxruntime
  for (size_t i = 0; i < num_values; ++i) {
    const OrtValue* ov = in[i];
    ONNXType ov_type;
    if (auto status = OrtApis::GetValueType(ov, &ov_type))
      return status;
    if (ov_type != first_value_type) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "At least one element in the sequence is of a type different from others.");
    }
  }

  // finally create the output vector/MLValue
  auto first_mlvalue = reinterpret_cast<const OrtValue*>(ovfirst);
  if (first_value_type == ONNX_TYPE_TENSOR) {
    return OrtCreateValueImplSeqHelper(in, num_values, out);
  } else if (first_value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    auto map_type = first_mlvalue->Type();
    utils::ContainerChecker c_checker(map_type);
    if (c_checker.IsMapOf<std::string, float>()) {
      return OrtCreateValueImplSeqHelperMap<MapStringToFloat>(in, num_values, out);
    }
    if (c_checker.IsMapOf<int64_t, float>()) {
      return OrtCreateValueImplSeqHelperMap<MapInt64ToFloat>(in, num_values, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
    }
#else
    ORT_UNUSED_PARAMETER(first_mlvalue);
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif

  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Unsupported input type");
  }
}

#if !defined(DISABLE_ML_OPS)
template <typename KeyType, typename ValueType>
static OrtStatus* OrtCreateMapMLValue(const Tensor& key_tensor, const Tensor& value_tensor, _Outptr_ OrtValue** out) {
  using MapType = std::map<KeyType, ValueType>;
  auto map_ptr = std::make_unique<MapType>();
  // iterate through the key and value tensors and populate map
  auto key_data = key_tensor.Data<KeyType>();
  auto value_data = value_tensor.Data<ValueType>();
  auto len = key_tensor.Shape().Size();
  ORT_ENFORCE(len >= 0 && static_cast<uint64_t>(len) < std::numeric_limits<size_t>::max());
  size_t num_kv_pairs = static_cast<size_t>(key_tensor.Shape().Size());
  for (size_t n = 0; n < num_kv_pairs; ++n, ++key_data, ++value_data) {
    map_ptr->insert({*key_data, *value_data});
  }
  // create ort_value with this map
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<MapType>();
  value->Init(map_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename KeyType>
static ORT_STATUS_PTR OrtCreateValueImplMapHelper(const Tensor& key_tensor, const Tensor& value_tensor,
                                                  _Outptr_ OrtValue** out) {
  auto value_type = value_tensor.DataType()->AsPrimitiveDataType();
  ORT_ENFORCE(value_type != nullptr, "Tensor must always contain primitive types. Found: ",
              DataTypeImpl::ToString(value_tensor.DataType()));

  switch (value_type->GetDataType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return OrtCreateMapMLValue<KeyType, std::string>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return OrtCreateMapMLValue<KeyType, int64_t>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return OrtCreateMapMLValue<KeyType, float>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return OrtCreateMapMLValue<KeyType, double>(key_tensor, value_tensor, out);
      break;
    default:
      break;
  }

  std::string msg("Value type is not supported yet: ");
  msg += DataTypeImpl::ToString(value_tensor.DataType());
  return OrtApis::CreateStatus(ORT_FAIL, msg.c_str());
}

static ORT_STATUS_PTR OrtCreateValueImplMap(const OrtValue* const* in, size_t num_values, _Outptr_ OrtValue** out) {
  if (num_values != NUM_MAP_INDICES) {
    return OrtApis::CreateStatus(ORT_FAIL, "For map type num_values MUST be 2");
  }

  const OrtValue* ort_keys = in[0];
  auto p_key_ml_value = reinterpret_cast<const OrtValue*>(ort_keys);
  auto& key_tensor = p_key_ml_value->Get<Tensor>();

  const OrtValue* ort_values = in[1];
  auto p_value_ml_value = reinterpret_cast<const OrtValue*>(ort_values);
  auto& value_tensor = p_value_ml_value->Get<Tensor>();

  // as per data_types.h, we only support maps of primitive data types.
  if (key_tensor.Shape().NumDimensions() > 1 || value_tensor.Shape().NumDimensions() > 1) {
    return OrtApis::CreateStatus(ORT_FAIL, "Either the key tensor or the value tensor has NumDimensions > 1");
  }

  // since maps are represented by key and value tensors, their sizes have to be the same.
  if (key_tensor.Shape().Size() != value_tensor.Shape().Size()) {
    return OrtApis::CreateStatus(ORT_FAIL, "Key and value tensors have unequal number of elements.");
  }

  if (key_tensor.IsDataTypeString()) {
    return OrtCreateValueImplMapHelper<std::string>(key_tensor, value_tensor, out);
  }
  if (key_tensor.IsDataType<int64_t>()) {
    return OrtCreateValueImplMapHelper<int64_t>(key_tensor, value_tensor, out);
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Key type is not supported yet.");
}
#endif

static ORT_STATUS_PTR OrtCreateValueImpl(_In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                                         enum ONNXType value_type, _Outptr_ OrtValue** out) {
  if (num_values <= 0) {
    return OrtApis::CreateStatus(ORT_FAIL, "Number of values should be at least 1.");
  }
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    return OrtCreateValueImplMap(in, num_values, out);
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtCreateValueImplSeq(in, num_values, out);
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
}

ORT_API_STATUS_IMPL(OrtApis::CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                    enum ONNXType value_type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  return OrtCreateValueImpl(in, num_values, value_type, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                    _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::string dtype("opaque(");
  dtype.append(domain_name).append(",").append(type_name).append(")");
  MLDataType ml_type = DataTypeImpl::GetDataType(dtype);
  ORT_ENFORCE(ml_type != nullptr,
              "Specified domain and type names combination does not refer to a registered opaque type");
  const auto* non_tensor_base = ml_type->AsNonTensorType();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  std::unique_ptr<OrtValue> ort_val = std::make_unique<OrtValue>();
  non_tensor_base->FromDataContainer(data_container, data_container_size, *ort_val);
  *out = ort_val.release();
  API_IMPL_END
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size) {
  API_IMPL_BEGIN
  std::string dtype("opaque(");
  dtype.append(domain_name).append(",").append(type_name).append(")");
  MLDataType ml_type = DataTypeImpl::GetDataType(dtype);
  ORT_ENFORCE(ml_type != nullptr,
              "Specified domain and type names combination does not refer to a registered opaque type");
  const auto* non_tensor_base = ml_type->AsNonTensorType();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  non_tensor_base->ToDataContainer(*in, data_container_size, data_container);
  API_IMPL_END
  return nullptr;
}

namespace {

struct ProviderBuffer {
  char** buffer_;
  char* next_write_;

  ProviderBuffer(char** buf, size_t p_count) {
    buffer_ = buf;
    next_write_ = DataStart(p_count);
  }

  char* DataStart(size_t p_count) { return reinterpret_cast<char*>(buffer_ + p_count); }
  // Return next buffer ptr
  void Append(const std::string& provider, size_t p_index) {
    // Maximum provider name length is now enforced at GetAvailableExecutionProviderNames()
    const size_t to_copy = provider.size();
#ifdef _MSC_VER
    memcpy_s(next_write_, to_copy, provider.data(), to_copy);
#elif defined(__APPLE__)
    memcpy(next_write_, provider.data(), to_copy);
#else
    memcpy(next_write_, provider.data(), to_copy);
#endif
    next_write_[to_copy] = 0;
    buffer_[p_index] = next_write_;
    next_write_ += to_copy + 1;
  }
};
}  // namespace

ORT_API_STATUS_IMPL(OrtApis::GetAvailableProviders, _Outptr_ char*** out_ptr,
                    _In_ int* providers_length) {
  API_IMPL_BEGIN
  const auto& available_providers = GetAvailableExecutionProviderNames();
  const size_t available_count = available_providers.size();

  if (available_count == 0) {
    out_ptr = nullptr;
    *providers_length = 0;
    return OrtApis::CreateStatus(ORT_FAIL, "Invalid build with no providers available");
  }

  size_t output_len = 0;
  for (const auto& p : available_providers) {
    output_len += p.size() + 1;
  }

  // We allocate and construct the buffer in char* to hold all the string pointers
  // followed by the actual string data. We allocate in terms of char* to make it convinient and avoid casts.
  const size_t ptrs_num = (sizeof(char*) * available_count + output_len + (sizeof(char*) - 1)) / sizeof(char*);
  auto total_buffer = std::make_unique<char*[]>(ptrs_num);
  ProviderBuffer provider_buffer(total_buffer.get(), available_count);

  for (size_t p_index = 0; p_index < available_count; p_index++) {
    provider_buffer.Append(available_providers[p_index], p_index);
  }

  *providers_length = narrow<int>(available_count);
  *out_ptr = total_buffer.release();
  API_IMPL_END
  return nullptr;
}

// This is a cleanup API, it should never return any failure
// so any no-throw code can rely on it.
ORT_API_STATUS_IMPL(OrtApis::ReleaseAvailableProviders, _In_ char** ptr,
                    _In_ int /* providers_length */) {
  API_IMPL_BEGIN
  // take possession of the memory and deallocate it
  std::unique_ptr<char*[]> g(ptr);
  API_IMPL_END
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetExecutionProviderApi,
                    [[maybe_unused]] _In_ const char* provider_name,
                    [[maybe_unused]] _In_ uint32_t version,
                    _Outptr_ const void** provider_api) {
  API_IMPL_BEGIN

  *provider_api = nullptr;
#ifdef USE_DML
  if (strcmp(provider_name, "DML") == 0) {
    *provider_api = GetOrtDmlApi(version);
    if (*provider_api == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Specified version is not supported for the DirectML provider.");
    }
    return NULL;
  }
#endif

  return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Specified provider is not supported.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count,
                    _Outptr_ void** out) {
  TENSOR_READWRITE_API_BEGIN

  if (tensor->IsDataTypeString()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "this API does not support strings");
  }

  const auto& tensor_shape = tensor->Shape();
  const auto num_dimensions = tensor_shape.NumDimensions();
  if (location_values_count != num_dimensions) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "location dimensions do not match shape size");
  }

  for (size_t i = 0; i < location_values_count; i++) {
    if (location_values[i] >= tensor_shape[i] || location_values[i] < 0) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "invalid location range");
    }
  }

  // compute strides
  // TensorPitches p;
  std::vector<int64_t> strides(num_dimensions);
  {
    int64_t stride = 1;
    for (size_t dim = num_dimensions; dim > 0; --dim) {
      strides[dim - 1] = stride;
      stride *= tensor_shape[dim - 1];
    }
  }

  // For Scalers the offset would always be zero
  int64_t offset = 0;
  for (size_t i = 0; i < num_dimensions; i++) {
    offset += location_values[i] * strides[i];
  }

  auto data = reinterpret_cast<char*>(tensor->MutableDataRaw()) + tensor->DataType()->Size() * offset;
  *out = data;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().SetLanguageProjection(static_cast<uint32_t>(projection));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetProfilingStartTimeNs, _In_ const OrtSession* sess, _Out_ uint64_t* out) {
  API_IMPL_BEGIN
  const auto* session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto profiling_start_time = session->GetProfiling().GetStartTimeNs();
  *out = static_cast<uint64_t>(profiling_start_time);
  return nullptr;
  API_IMPL_END
}

// End support for non-tensor types

ORT_API_STATUS_IMPL(OrtApis::CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
                    int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out) {
  API_IMPL_BEGIN
  auto cfg = std::make_unique<OrtArenaCfg>();
  cfg->max_mem = max_mem;
  cfg->arena_extend_strategy = arena_extend_strategy;
  cfg->initial_chunk_size_bytes = initial_chunk_size_bytes;
  cfg->max_dead_bytes_per_chunk = max_dead_bytes_per_chunk;
  cfg->max_dead_bytes_per_chunk = -1L;
  *out = cfg.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys, _In_reads_(num_keys) const size_t* arena_config_values,
                    _In_ size_t num_keys, _Outptr_ OrtArenaCfg** out) {
  API_IMPL_BEGIN
  auto cfg = std::make_unique<OrtArenaCfg>();

  for (size_t i = 0; i < num_keys; ++i) {
    if (strcmp(arena_config_keys[i], "max_mem") == 0) {
      cfg->max_mem = arena_config_values[i];
    } else if (strcmp(arena_config_keys[i], "arena_extend_strategy") == 0) {
      cfg->arena_extend_strategy = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "initial_chunk_size_bytes") == 0) {
      cfg->initial_chunk_size_bytes = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "max_dead_bytes_per_chunk") == 0) {
      cfg->max_dead_bytes_per_chunk = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "initial_growth_chunk_size_bytes") == 0) {
      cfg->initial_growth_chunk_size_bytes = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "max_power_of_two_extend_bytes") == 0) {
      cfg->max_power_of_two_extend_bytes = static_cast<int64_t>(arena_config_values[i]);
    } else {
      std::ostringstream oss;
      oss << "Invalid key found: " << arena_config_keys[i];

      return CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
    }
  }

  *out = cfg.release();
  return nullptr;
  API_IMPL_END
}

// Allow using raw new/delete because this is for C.
ORT_API(void, OrtApis::ReleaseArenaCfg, _Frees_ptr_opt_ OrtArenaCfg* ptr) {
  std::unique_ptr<OrtArenaCfg> g(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out) {
  API_IMPL_BEGIN
  std::unique_ptr<PrepackedWeightsContainer> container = std::make_unique<PrepackedWeightsContainer>();
  *out = reinterpret_cast<OrtPrepackedWeightsContainer*>(container.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleasePrepackedWeightsContainer, _Frees_ptr_opt_ OrtPrepackedWeightsContainer* ptr) {
  delete reinterpret_cast<PrepackedWeightsContainer*>(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess, prepacked_weights_container));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data,
                                                      model_data_length, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess, prepacked_weights_container));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorMemoryInfo, _In_ const OrtValue* value, _Outptr_ const OrtMemoryInfo** memory_info) {
  TENSOR_READ_API_BEGIN
  *memory_info = &tensor.Location();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetCustomCreateThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  API_IMPL_BEGIN
  options->value.custom_create_thread_fn = ort_custom_create_thread_fn;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetCustomThreadCreationOptions, _Inout_ OrtSessionOptions* options, _In_ void* ort_custom_thread_creation_options) {
  API_IMPL_BEGIN
  options->value.custom_thread_creation_options = ort_custom_thread_creation_options;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetCustomJoinThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  API_IMPL_BEGIN
  options->value.custom_join_thread_fn = ort_custom_join_thread_fn;
  return nullptr;
  API_IMPL_END
}

ORT_API(const OrtTrainingApi*, OrtApis::GetTrainingApi, uint32_t version) {
#ifdef ENABLE_TRAINING_APIS
  if (version >= 13 && version <= ORT_API_VERSION)
    return OrtTrainingApis::GetTrainingApi(version);

  fprintf(stderr, "The given version [%u] is not supported. Training api only supports version 13 to %u.\n",
          version, ORT_API_VERSION);
  return nullptr;
#else

  ORT_UNUSED_PARAMETER(version);

  return nullptr;
#endif
}

static constexpr OrtApiBase ort_api_base = {
    &OrtApis::GetApi,
    &OrtApis::GetVersionString};

/* Rules on how to add a new Ort API version

In general, NEVER remove or rearrange the members in this structure unless a new version is being created. The
goal is for newer shared libraries of the Onnx Runtime to work with binaries targeting the previous versions.
In order to do that we need to ensure older binaries get the older interfaces they are expecting.

If the next version of the OrtApi only adds members, new members can be added at the end of the OrtApi structure
without breaking anything. In this case, rename the ort_api_# structure in a way that shows the range of versions
it supports, for example 'ort_api_1_to_2', and then GetApi can return the same structure for a range of versions.

If methods need to be removed or rearranged, then make a copy of the OrtApi structure and name it 'OrtApi#to#'.
The latest Api should always be named just OrtApi. Then make a copy of the latest ort_api_* structure below and
name it ort_api_# to match the latest version number supported, you'll need to be sure the structure types match
the API they're for (the compiler should complain if this isn't correct).

If there is no desire to have the headers still expose the older APIs (clutter, documentation, etc) then the
definition should be moved to a file included by this file so that it's still defined here for binary compatibility
but isn't visible in public headers.

So for example, if we wanted to just add some new members to the ort_api_1_to_2, we'd take the following steps:

    In include\onnxruntime\core\session\onnxruntime_c_api.h we'd just add the members to the end of the structure

    In this file, we'd correspondingly add the member values to the end of the ort_api_1_to_2 structure, and also rename
    it to ort_api_1_to_3.

    Then in GetApi we'd make it return ort_api_1_to_3 for versions 1 through 3.

Second example, if we wanted to add and remove some members, we'd do this:

    In include\onnxruntime\core\session\onnxruntime_c_api.h we'd make a copy of the OrtApi structure and name the
    old one OrtApi1to2. In the new OrtApi we'd add or remove any members that we desire.

    In this file, we'd create a new copy of ort_api_1_to_2 called ort_api_3 and make the corresponding changes that were
    made to the new OrtApi.

    In GetApi we now make it return ort_api_3 for version 3.
*/

static constexpr OrtApi ort_api_1_to_17 = {
    // NOTE: The ordering of these fields MUST not change after that version has shipped since existing binaries depend on this ordering.

    // Shipped as version 1 - DO NOT MODIFY (see above text for more information)
    &OrtApis::CreateStatus,
    &OrtApis::GetErrorCode,
    &OrtApis::GetErrorMessage,

    &OrtApis::CreateEnv,
    &OrtApis::CreateEnvWithCustomLogger,
    &OrtApis::EnableTelemetryEvents,
    &OrtApis::DisableTelemetryEvents,

    &OrtApis::CreateSession,
    &OrtApis::CreateSessionFromArray,
    &OrtApis::Run,

    &OrtApis::CreateSessionOptions,
    &OrtApis::SetOptimizedModelFilePath,
    &OrtApis::CloneSessionOptions,
    &OrtApis::SetSessionExecutionMode,
    &OrtApis::EnableProfiling,
    &OrtApis::DisableProfiling,
    &OrtApis::EnableMemPattern,
    &OrtApis::DisableMemPattern,
    &OrtApis::EnableCpuMemArena,
    &OrtApis::DisableCpuMemArena,
    &OrtApis::SetSessionLogId,
    &OrtApis::SetSessionLogVerbosityLevel,
    &OrtApis::SetSessionLogSeverityLevel,
    &OrtApis::SetSessionGraphOptimizationLevel,
    &OrtApis::SetIntraOpNumThreads,
    &OrtApis::SetInterOpNumThreads,

    &OrtApis::CreateCustomOpDomain,
    &OrtApis::CustomOpDomain_Add,
    &OrtApis::AddCustomOpDomain,
    &OrtApis::RegisterCustomOpsLibrary,

    &OrtApis::SessionGetInputCount,
    &OrtApis::SessionGetOutputCount,
    &OrtApis::SessionGetOverridableInitializerCount,
    &OrtApis::SessionGetInputTypeInfo,
    &OrtApis::SessionGetOutputTypeInfo,
    &OrtApis::SessionGetOverridableInitializerTypeInfo,
    &OrtApis::SessionGetInputName,
    &OrtApis::SessionGetOutputName,
    &OrtApis::SessionGetOverridableInitializerName,

    &OrtApis::CreateRunOptions,
    &OrtApis::RunOptionsSetRunLogVerbosityLevel,
    &OrtApis::RunOptionsSetRunLogSeverityLevel,
    &OrtApis::RunOptionsSetRunTag,
    &OrtApis::RunOptionsGetRunLogVerbosityLevel,
    &OrtApis::RunOptionsGetRunLogSeverityLevel,
    &OrtApis::RunOptionsGetRunTag,
    &OrtApis::RunOptionsSetTerminate,
    &OrtApis::RunOptionsUnsetTerminate,

    &OrtApis::CreateTensorAsOrtValue,
    &OrtApis::CreateTensorWithDataAsOrtValue,
    &OrtApis::IsTensor,
    &OrtApis::GetTensorMutableData,

    &OrtApis::FillStringTensor,
    &OrtApis::GetStringTensorDataLength,
    &OrtApis::GetStringTensorContent,

    &OrtApis::CastTypeInfoToTensorInfo,
    &OrtApis::GetOnnxTypeFromTypeInfo,
    &OrtApis::CreateTensorTypeAndShapeInfo,
    &OrtApis::SetTensorElementType,

    &OrtApis::SetDimensions,
    &OrtApis::GetTensorElementType,
    &OrtApis::GetDimensionsCount,
    &OrtApis::GetDimensions,
    &OrtApis::GetSymbolicDimensions,
    &OrtApis::GetTensorShapeElementCount,
    &OrtApis::GetTensorTypeAndShape,
    &OrtApis::GetTypeInfo,
    &OrtApis::GetValueType,
    &OrtApis::CreateMemoryInfo,
    &OrtApis::CreateCpuMemoryInfo,
    &OrtApis::CompareMemoryInfo,
    &OrtApis::MemoryInfoGetName,
    &OrtApis::MemoryInfoGetId,
    &OrtApis::MemoryInfoGetMemType,
    &OrtApis::MemoryInfoGetType,
    &OrtApis::AllocatorAlloc,
    &OrtApis::AllocatorFree,
    &OrtApis::AllocatorGetInfo,
    &OrtApis::GetAllocatorWithDefaultOptions,
    &OrtApis::AddFreeDimensionOverride,
    &OrtApis::GetValue,
    &OrtApis::GetValueCount,
    &OrtApis::CreateValue,
    &OrtApis::CreateOpaqueValue,
    &OrtApis::GetOpaqueValue,

    &OrtApis::KernelInfoGetAttribute_float,
    &OrtApis::KernelInfoGetAttribute_int64,
    &OrtApis::KernelInfoGetAttribute_string,
    &OrtApis::KernelContext_GetInputCount,
    &OrtApis::KernelContext_GetOutputCount,
    &OrtApis::KernelContext_GetInput,
    &OrtApis::KernelContext_GetOutput,

    &OrtApis::ReleaseEnv,
    &OrtApis::ReleaseStatus,
    &OrtApis::ReleaseMemoryInfo,
    &OrtApis::ReleaseSession,
    &OrtApis::ReleaseValue,
    &OrtApis::ReleaseRunOptions,
    &OrtApis::ReleaseTypeInfo,
    &OrtApis::ReleaseTensorTypeAndShapeInfo,
    &OrtApis::ReleaseSessionOptions,
    &OrtApis::ReleaseCustomOpDomain,
    // End of Version 1 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetDenotationFromTypeInfo,
    &OrtApis::CastTypeInfoToMapTypeInfo,
    &OrtApis::CastTypeInfoToSequenceTypeInfo,
    &OrtApis::GetMapKeyType,
    &OrtApis::GetMapValueType,
    &OrtApis::GetSequenceElementType,
    &OrtApis::ReleaseMapTypeInfo,
    &OrtApis::ReleaseSequenceTypeInfo,
    &OrtApis::SessionEndProfiling,
    &OrtApis::SessionGetModelMetadata,
    &OrtApis::ModelMetadataGetProducerName,
    &OrtApis::ModelMetadataGetGraphName,
    &OrtApis::ModelMetadataGetDomain,
    &OrtApis::ModelMetadataGetDescription,
    &OrtApis::ModelMetadataLookupCustomMetadataMap,
    &OrtApis::ModelMetadataGetVersion,
    &OrtApis::ReleaseModelMetadata,
    // End of Version 2 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::CreateEnvWithGlobalThreadPools,
    &OrtApis::DisablePerSessionThreads,
    &OrtApis::CreateThreadingOptions,
    &OrtApis::ReleaseThreadingOptions,
    &OrtApis::ModelMetadataGetCustomMetadataMapKeys,
    &OrtApis::AddFreeDimensionOverrideByName,
    // End of Version 3 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetAvailableProviders,
    &OrtApis::ReleaseAvailableProviders,
    // End of Version 4 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetStringTensorElementLength,
    &OrtApis::GetStringTensorElement,
    &OrtApis::FillStringTensorElement,
    &OrtApis::AddSessionConfigEntry,

    // IoBinding and above are propagated in the same order to C# API
    // Do not move
    &OrtApis::CreateAllocator,
    &OrtApis::ReleaseAllocator,
    &OrtApis::RunWithBinding,
    &OrtApis::CreateIoBinding,
    &OrtApis::ReleaseIoBinding,
    &OrtApis::BindInput,
    &OrtApis::BindOutput,
    &OrtApis::BindOutputToDevice,
    &OrtApis::GetBoundOutputNames,
    &OrtApis::GetBoundOutputValues,
    &OrtApis::ClearBoundInputs,
    &OrtApis::ClearBoundOutputs,
    &OrtApis::TensorAt,
    &OrtApis::CreateAndRegisterAllocator,
    &OrtApis::SetLanguageProjection,
    &OrtApis::SessionGetProfilingStartTimeNs,
    &OrtApis::SetGlobalIntraOpNumThreads,
    &OrtApis::SetGlobalInterOpNumThreads,
    &OrtApis::SetGlobalSpinControl,
    // End of Version 5 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::AddInitializer,
    &OrtApis::CreateEnvWithCustomLoggerAndGlobalThreadPools,
    &OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
    &OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
    &OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
    &OrtApis::SetGlobalDenormalAsZero,
    &OrtApis::CreateArenaCfg,
    &OrtApis::ReleaseArenaCfg,
    // End of Version 6 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::ModelMetadataGetGraphDescription,
    &OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
    &OrtApis::SetCurrentGpuDeviceId,
    &OrtApis::GetCurrentGpuDeviceId,
    // End of Version 7 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::KernelInfoGetAttributeArray_float,
    &OrtApis::KernelInfoGetAttributeArray_int64,
    &OrtApis::CreateArenaCfgV2,
    &OrtApis::AddRunConfigEntry,
    &OrtApis::CreatePrepackedWeightsContainer,
    &OrtApis::ReleasePrepackedWeightsContainer,
    &OrtApis::CreateSessionWithPrepackedWeightsContainer,
    &OrtApis::CreateSessionFromArrayWithPrepackedWeightsContainer,
    // End of Version 8 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2,
    &OrtApis::CreateTensorRTProviderOptions,
    &OrtApis::UpdateTensorRTProviderOptions,
    &OrtApis::GetTensorRTProviderOptionsAsString,
    &OrtApis::ReleaseTensorRTProviderOptions,
    &OrtApis::EnableOrtCustomOps,
    &OrtApis::RegisterAllocator,
    &OrtApis::UnregisterAllocator,
    &OrtApis::IsSparseTensor,
    &OrtApis::CreateSparseTensorAsOrtValue,
    &OrtApis::FillSparseTensorCoo,
    &OrtApis::FillSparseTensorCsr,
    &OrtApis::FillSparseTensorBlockSparse,
    &OrtApis::CreateSparseTensorWithValuesAsOrtValue,
    &OrtApis::UseCooIndices,
    &OrtApis::UseCsrIndices,
    &OrtApis::UseBlockSparseIndices,
    &OrtApis::GetSparseTensorFormat,
    &OrtApis::GetSparseTensorValuesTypeAndShape,
    &OrtApis::GetSparseTensorValues,
    &OrtApis::GetSparseTensorIndicesTypeShape,
    &OrtApis::GetSparseTensorIndices,
    // End of Version 9 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::HasValue,
    &OrtApis::KernelContext_GetGPUComputeStream,
    &OrtApis::GetTensorMemoryInfo,
    &OrtApis::GetExecutionProviderApi,
    &OrtApis::SessionOptionsSetCustomCreateThreadFn,
    &OrtApis::SessionOptionsSetCustomThreadCreationOptions,
    &OrtApis::SessionOptionsSetCustomJoinThreadFn,
    &OrtApis::SetGlobalCustomCreateThreadFn,
    &OrtApis::SetGlobalCustomThreadCreationOptions,
    &OrtApis::SetGlobalCustomJoinThreadFn,
    &OrtApis::SynchronizeBoundInputs,
    &OrtApis::SynchronizeBoundOutputs,
    // End of Version 10 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::SessionOptionsAppendExecutionProvider_CUDA_V2,
    &OrtApis::CreateCUDAProviderOptions,
    &OrtApis::UpdateCUDAProviderOptions,
    &OrtApis::GetCUDAProviderOptionsAsString,
    &OrtApis::ReleaseCUDAProviderOptions,
    &OrtApis::SessionOptionsAppendExecutionProvider_MIGraphX,
    // End of Version 11 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::AddExternalInitializers,
    &OrtApis::CreateOpAttr,
    &OrtApis::ReleaseOpAttr,
    &OrtApis::CreateOp,
    &OrtApis::InvokeOp,
    &OrtApis::ReleaseOp,
    &OrtApis::SessionOptionsAppendExecutionProvider,
    &OrtApis::CopyKernelInfo,
    &OrtApis::ReleaseKernelInfo,
    // End of Version 12 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetTrainingApi,
    &OrtApis::SessionOptionsAppendExecutionProvider_CANN,
    &OrtApis::CreateCANNProviderOptions,
    &OrtApis::UpdateCANNProviderOptions,
    &OrtApis::GetCANNProviderOptionsAsString,
    &OrtApis::ReleaseCANNProviderOptions,
    // End of Version 13 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::MemoryInfoGetDeviceType,
    &OrtApis::UpdateEnvWithCustomLogLevel,
    &OrtApis::SetGlobalIntraOpThreadAffinity,
    &OrtApis::RegisterCustomOpsLibrary_V2,
    &OrtApis::RegisterCustomOpsUsingFunction,
    &OrtApis::KernelInfo_GetInputCount,
    &OrtApis::KernelInfo_GetOutputCount,
    &OrtApis::KernelInfo_GetInputName,
    &OrtApis::KernelInfo_GetOutputName,
    &OrtApis::KernelInfo_GetInputTypeInfo,
    &OrtApis::KernelInfo_GetOutputTypeInfo,
    &OrtApis::KernelInfoGetAttribute_tensor,
    &OrtApis::HasSessionConfigEntry,
    &OrtApis::GetSessionConfigEntry,
    // End of Version 14 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::SessionOptionsAppendExecutionProvider_Dnnl,
    &OrtApis::CreateDnnlProviderOptions,
    &OrtApis::UpdateDnnlProviderOptions,
    &OrtApis::GetDnnlProviderOptionsAsString,
    &OrtApis::ReleaseDnnlProviderOptions,
    &OrtApis::KernelInfo_GetNodeName,
    &OrtApis::KernelInfo_GetLogger,
    &OrtApis::KernelContext_GetLogger,
    &OrtApis::Logger_LogMessage,
    &OrtApis::Logger_GetLoggingSeverityLevel,
    &OrtApis::KernelInfoGetConstantInput_tensor,
    &OrtApis::CastTypeInfoToOptionalTypeInfo,
    &OrtApis::GetOptionalContainedTypeInfo,
    &OrtApis::GetResizedStringTensorElementBuffer,
    &OrtApis::KernelContext_GetAllocator,
    &OrtApis::GetBuildInfoString,
    // End of Version 15 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::CreateROCMProviderOptions,
    &OrtApis::UpdateROCMProviderOptions,
    &OrtApis::GetROCMProviderOptionsAsString,
    &OrtApis::ReleaseROCMProviderOptions,
    &OrtApis::CreateAndRegisterAllocatorV2,
    &OrtApis::RunAsync,
    &OrtApis::UpdateTensorRTProviderOptionsWithValue,
    &OrtApis::GetTensorRTProviderOptionsByName,
    &OrtApis::UpdateCUDAProviderOptionsWithValue,
    &OrtApis::GetCUDAProviderOptionsByName,
    &OrtApis::KernelContext_GetResource,
    // End of Version 16 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::SetUserLoggingFunction,
    &OrtApis::ShapeInferContext_GetInputCount,
    &OrtApis::ShapeInferContext_GetInputTypeShape,
    &OrtApis::ShapeInferContext_GetAttribute,
    &OrtApis::ShapeInferContext_SetOutputTypeShape,
    &OrtApis::SetSymbolicDimensions,
    &OrtApis::ReadOpAttr,
    &OrtApis::EvictSession,
};

// OrtApiBase can never change as there is no way to know what version of OrtApiBase is returned by OrtGetApiBase.
static_assert(sizeof(OrtApiBase) == sizeof(void*) * 2, "New methods can't be added to OrtApiBase as it is not versioned");
static_assert(offsetof(OrtApiBase, GetApi) / sizeof(void*) == 0, "These functions cannot be reordered");
static_assert(offsetof(OrtApiBase, GetVersionString) / sizeof(void*) == 1, "These functions cannot be reordered");
static_assert(std::is_same_v<decltype(OrtApiBase::GetApi), const OrtApi*(ORT_API_CALL*)(uint32_t)NO_EXCEPTION>, "This function's signature can never change");
static_assert(std::is_same_v<decltype(OrtApiBase::GetVersionString), const char*(ORT_API_CALL*)(void)NO_EXCEPTION>, "This function's signature can never change");

// Asserts to do a some checks to ensure older Versions of the OrtApi never change (will detect an addition or deletion but not if they cancel out each other)
// If any of these asserts hit, read the above 'Rules on how to add a new Ort API version'
static_assert(offsetof(OrtApi, ReleaseCustomOpDomain) / sizeof(void*) == 101, "Size of version 1 API cannot change");
static_assert(offsetof(OrtApi, ReleaseModelMetadata) / sizeof(void*) == 118, "Size of version 2 API cannot change");
static_assert(offsetof(OrtApi, AddFreeDimensionOverrideByName) / sizeof(void*) == 124,
              "Size of version 3 API cannot change");
static_assert(offsetof(OrtApi, ReleaseAvailableProviders) / sizeof(void*) == 126,
              "Size of version 4 API cannot change");
static_assert(offsetof(OrtApi, SetGlobalSpinControl) / sizeof(void*) == 149, "Size of version 5 API cannot change");
static_assert(offsetof(OrtApi, ReleaseArenaCfg) / sizeof(void*) == 157, "Size of version 6 API cannot change");
static_assert(offsetof(OrtApi, GetCurrentGpuDeviceId) / sizeof(void*) == 161, "Size of version 7 API cannot change");
static_assert(offsetof(OrtApi, CreateSessionFromArrayWithPrepackedWeightsContainer) / sizeof(void*) == 169, "Size of version 8 API cannot change");
static_assert(offsetof(OrtApi, GetSparseTensorIndices) / sizeof(void*) == 191, "Size of version 9 API cannot change");
static_assert(offsetof(OrtApi, SynchronizeBoundOutputs) / sizeof(void*) == 203, "Size of version 10 API cannot change");
static_assert(offsetof(OrtApi, SessionOptionsAppendExecutionProvider_MIGraphX) / sizeof(void*) == 209, "Size of version 11 API cannot change");
static_assert(offsetof(OrtApi, ReleaseKernelInfo) / sizeof(void*) == 218, "Size of version 12 API cannot change");
static_assert(offsetof(OrtApi, ReleaseCANNProviderOptions) / sizeof(void*) == 224, "Size of version 13 API cannot change");
static_assert(offsetof(OrtApi, GetSessionConfigEntry) / sizeof(void*) == 238, "Size of version 14 API cannot change");
static_assert(offsetof(OrtApi, GetBuildInfoString) / sizeof(void*) == 254, "Size of version 15 API cannot change");
static_assert(offsetof(OrtApi, KernelContext_GetResource) / sizeof(void*) == 265, "Size of version 16 API cannot change");
static_assert(offsetof(OrtApi, SetUserLoggingFunction) / sizeof(void*) == 266, "Size of version 17 API cannot change");

// So that nobody forgets to finish an API version, this check will serve as a reminder:
static_assert(std::string_view(ORT_VERSION) == "1.17.0",
              "ORT_Version change detected, please follow below steps to ensure OrtApi is updated properly");
// 1. Update the hardcoded version string in above static_assert to silence it
// 2. If there were any APIs added to ort_api_1_to_17 above:
//    a. Add the 'End of version #' markers (pattern above should be obvious)
//    b. Add a static_assert in the directly above list of version sizes to ensure nobody adds any more functions to the just shipped API version

ORT_API(const OrtApi*, OrtApis::GetApi, uint32_t version) {
  if (version >= 1 && version <= ORT_API_VERSION)
    return &ort_api_1_to_17;

  fprintf(stderr,
          "The requested API version [%u] is not available, only API versions [1, %u] are supported in this build."
          " Current ORT Version is: %s\n",
          version, ORT_API_VERSION, ORT_VERSION);

  return nullptr;  // Unsupported version
}

ORT_API(const char*, OrtApis::GetVersionString) {
  return ORT_VERSION;
}

ORT_API(const char*, OrtApis::GetBuildInfoString) {
  return ORT_BUILD_INFO;
}

const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
  return &ort_api_base;
}

ORT_API(void, OrtApis::ReleaseEnv, OrtEnv* value) {
  OrtEnv::Release(value);
}

DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Value, OrtValue)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(RunOptions, OrtRunOptions)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Session, ::onnxruntime::InferenceSession)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(ModelMetadata, ::onnxruntime::ModelMetadata)
