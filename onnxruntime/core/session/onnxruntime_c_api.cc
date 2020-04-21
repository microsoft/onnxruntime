// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
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

using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
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

// Ignore warnings that arise due to casting const OrtApi1to2* to const OrtApi* because
// OrtApi has more members than OrtApi1to2 and GCC/Clang warn about missing member initializations
// that we don't need to worry about
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status) return _status;      \
  } while (0)

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, default_warning_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnv, OrtLoggingLevel default_warning_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, default_warning_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithGlobalThreadPools, OrtLoggingLevel default_warning_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, default_warning_level, logid};
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

ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len,
                                _Inout_ OrtAllocator* allocator, std::unique_ptr<Tensor>* out) {
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    shapes[i] = shape[i];
  }
  std::shared_ptr<IAllocator> alloc_ptr = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  *out = onnxruntime::make_unique<Tensor>(ml_type, onnxruntime::TensorShape(shapes), alloc_ptr);
  return nullptr;
}

ORT_STATUS_PTR CreateTensorImplForSeq(MLDataType elem_type, const int64_t* shape, size_t shape_len, Tensor& out) {
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    shapes[i] = shape[i];
  }
  OrtAllocator* allocator;
  // TODO(pranav): what allocator should be used to create the tensor here?
  // for the sake of simplicity of the API using the default one here
  auto st = OrtApis::GetAllocatorWithDefaultOptions(&allocator);
  if (st) {
    return st;
  }
  std::shared_ptr<IAllocator> alloc_ptr = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  out = Tensor(elem_type, onnxruntime::TensorShape(shapes), alloc_ptr);
  return nullptr;
}

/**
 *
 * this function will create a copy of the allocator info
 */
ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                                void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= static_cast<size_t>(shape[i]);
    shapes[i] = shape[i];
  }

  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(ml_type->Size(), elem_count, &size_to_allocate)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "size overflow");
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = onnxruntime::make_unique<Tensor>(ml_type, onnxruntime::TensorShape(shapes), p_data, *info);
  return nullptr;
}

namespace c_api_internal {

template <class T>
inline ORT_STATUS_PTR CallCreateTensorImpl(const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                                           void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  auto ml_value = DataTypeImpl::GetType<T>();
  return CreateTensorImpl(ml_value, shape, shape_len, info, p_data, p_data_len, out);
}

template <class T>
inline ORT_STATUS_PTR CallCreateTensorImpl(const int64_t* shape, size_t shape_len, _Inout_ OrtAllocator* allocator,
                                           std::unique_ptr<Tensor>* out) {
  auto ml_type = DataTypeImpl::GetType<T>();
  return CreateTensorImpl(ml_type, shape, shape_len, allocator, out);
}

}  // namespace c_api_internal

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<float>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<std::string>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<bool>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<MLFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<BFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<double>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  auto value = onnxruntime::make_unique<OrtValue>();
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value->Init(tensor.release(),
              ml_tensor,
              ml_tensor->GetDeleteFunc());
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<float>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int64_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint64_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<std::string>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<bool>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<MLFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<BFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<double>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  auto value = onnxruntime::make_unique<OrtValue>();
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value->Init(tensor.release(),
              ml_tensor,
              ml_tensor->GetDeleteFunc());
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out) {
  API_IMPL_BEGIN
  auto custom_op_domain = onnxruntime::make_unique<OrtCustomOpDomain>();
  custom_op_domain->domain_ = domain;
  *out = custom_op_domain.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op) {
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

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle) {
  API_IMPL_BEGIN

  Env::Default().LoadDynamicLibrary(library_path, library_handle);
  if (!*library_handle)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

  OrtStatus*(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

  Env::Default().GetSymbolFromLibrary(*library_handle, "RegisterCustomOps", (void**)&RegisterCustomOps);
  if (!RegisterCustomOps)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

  return RegisterCustomOps(options, OrtGetApiBase());
  API_IMPL_END
}

namespace {
ORT_STATUS_PTR LoadAndInitializeSession(_In_ const OrtEnv* /*env*/, _In_ const OrtSessionOptions* options,
                                        _In_ std::unique_ptr<::onnxruntime::InferenceSession>& sess,
                                        _Outptr_ OrtSession** out) {
  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      if (provider->Type() == kDmlExecutionProvider) {
        if (options->value.enable_mem_pattern) {
          // TODO Instead of returning an error, should we set mem pattern to false here and log a warning saying so?
          // Doing so would be inconsistent with the Python API that doesn't go through this code path.
          return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Mem pattern should be disabled when using DML execution provider.");
        }
        if (options->value.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
          return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Sequential execution should be enabled when using DML execution provider.");
        }
      }
      provider_list.push_back(std::move(provider));
    }
  }

  Status status;
  if (options) {
    if (!options->custom_op_domains_.empty()) {
      status = sess->AddCustomOpDomains(options->custom_op_domains_);
      if (!status.IsOK())
        return ToOrtStatus(status);
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      status = sess->RegisterExecutionProvider(std::move(provider));
      if (!status.IsOK())
        return ToOrtStatus(status);
    }
  }

  status = sess->Load();
  if (!status.IsOK())
    return ToOrtStatus(status);

  status = sess->Initialize();
  if (!status.IsOK())
    return ToOrtStatus(status);

  *out = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
}
}  // namespace

ORT_API_STATUS_IMPL(OrtApis::CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  try {
    sess = onnxruntime::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment(), model_path);
  } catch (const std::exception& e) {
    return OrtApis::CreateStatus(ORT_FAIL, e.what());
  }
  return LoadAndInitializeSession(env, options, sess, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  try {
    sess = onnxruntime::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment(), model_data, static_cast<int>(model_data_length));
  } catch (const std::exception& e) {
    return OrtApis::CreateStatus(ORT_FAIL, e.what());
  }
  return LoadAndInitializeSession(env, options, sess, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  const int queue_id = 0;

  std::vector<std::string> feed_names(input_len);
  std::vector<OrtValue> feeds(input_len);

  for (size_t i = 0; i != input_len; ++i) {
    if (input_names[i] == nullptr || input_names[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input name cannot be empty");
    }

    feed_names[i] = input_names[i];
    auto& ort_value = feeds[i] = *reinterpret_cast<const ::OrtValue*>(input[i]);

    if (ort_value.Fence()) ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
  }

  // Create output feed
  std::vector<std::string> output_names(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output_names1[i] == nullptr || output_names1[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "output name cannot be empty");
    }
    output_names[i] = output_names1[i];
  }

  std::vector<OrtValue> fetches(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output[i] != nullptr) {
      ::OrtValue& value = *(output[i]);
      if (value.Fence())
        value.Fence()->BeforeUsingAsOutput(onnxruntime::kCpuExecutionProvider, queue_id);
      fetches[i] = value;
    }
  }
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions op;
    status = session->Run(op, feed_names, feeds, output_names, &fetches);
  } else {
    status = session->Run(*run_options, feed_names, feeds, output_names, &fetches);
  }

  if (!status.IsOK())
    return ToOrtStatus(status);
  for (size_t i = 0; i != output_names_len; ++i) {
    ::OrtValue& value = fetches[i];
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (output[i] == nullptr) {
      output[i] = new OrtValue(value);
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::IsTensor, _In_ const OrtValue* value, _Out_ int* out) {
  auto v = reinterpret_cast<const ::OrtValue*>(value);
  *out = v->IsTensor() ? 1 : 0;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  //TODO: test if it's a string tensor
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len < len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i != len; ++i) {
    //allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* out) {
  TENSOR_READ_API_BEGIN
  const auto* src = tensor.Data<std::string>();
  int64_t len = tensor.Shape().Size();
  if (len >= 0) {
    size_t ret = 0;
    for (int64_t i = 0; i != len; ++i) {
      ret += src[i].size();
    }
    *out = ret;
  } else
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                    size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (offsets_len < len) {
    return OrtApis::CreateStatus(ORT_FAIL, "space is not enough");
  }
  {
    size_t ret = 0;
    for (size_t i = 0; i != len; ++i) {
      ret += input[i].size();
    }
    if (s_len < ret) {
      return OrtApis::CreateStatus(ORT_FAIL, "space is not enough");
    }
  }
  size_t f = 0;
  char* p = static_cast<char*>(s);
  for (size_t i = 0; i != offsets_len; ++i, ++offsets) {
    memcpy(p, input[i].data(), input[i].size());
    p += input[i].size();
    *offsets = f;
    f += input[i].size();
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
  return OrtTypeInfo::FromTypeProto(type_proto, out);
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

static char* StrDup(const std::string& str, _Inout_ OrtAllocator* allocator) {
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
  *out = reinterpret_cast<OrtModelMetadata*>(new ModelMetadata(*p.second));
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

    // alloc_count * sizeof(...) will throw if there was an overflow which will be caught in API_IMPL_END
    // and be returned to the user as a status
    char** p = reinterpret_cast<char**>(allocator->Alloc(allocator, alloc_count * sizeof(char*)));
    assert(p != nullptr);
    auto map_iter = custom_metadata_map.cbegin();
    int64_t i = 0;
    while (map_iter != custom_metadata_map.cend()) {
      p[i++] = StrDup(map_iter->first, allocator);
      ++map_iter;
    }
    *keys = p;
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

static const int NUM_MAP_INDICES = 2;

template <typename T>
ORT_STATUS_PTR OrtGetNumSequenceElements(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<T>();
  *out = data.size();
  return nullptr;
}

template <>
ORT_STATUS_PTR OrtGetNumSequenceElements<TensorSeq>(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<TensorSeq>();
  *out = data.Size();
  return nullptr;
}

static ORT_STATUS_PTR OrtGetValueCountImpl(const OrtValue* value, size_t* out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
    *out = NUM_MAP_INDICES;
    return nullptr;
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    auto v = reinterpret_cast<const OrtValue*>(value);
    auto type = v->Type();
    // Note: keep these in sync with the registered types in data_types.h
    if (type->IsTensorSequenceType()) {
      return OrtGetNumSequenceElements<TensorSeq>(v, out);
    } else {
      utils::ContainerChecker c_checker(type);
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        return OrtGetNumSequenceElements<VectorMapStringToFloat>(v, out);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        return OrtGetNumSequenceElements<VectorMapInt64ToFloat>(v, out);
      } else {
        return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
      }
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

///////////////////
// OrtGetValueImplSeqOfMap
template <typename T>
static ORT_STATUS_PTR OrtGetValueImplSeqOfMap(const OrtValue* p_ml_value, int index, _Outptr_ OrtValue** out) {
  using TKey = typename T::value_type::key_type;
  using TVal = typename T::value_type::mapped_type;
  using MapType = std::map<TKey, TVal>;
  auto& data_vec = p_ml_value->Get<T>();
  auto& data_elem = data_vec.at(index);
  auto copy_data_elem = onnxruntime::make_unique<MapType>(data_elem);
  auto value = onnxruntime::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<MapType>();
  value->Init(copy_data_elem.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

ORT_STATUS_PTR PopulateTensorWithData(_Inout_ OrtValue* oval, _In_ const void* data_elem, size_t num_elems,
                                      size_t elem_size) {
  void* raw_data = nullptr;
  auto st = OrtApis::GetTensorMutableData(oval, &raw_data);
  if (st) {
    return st;
  }
  memcpy(raw_data, data_elem, elem_size * num_elems);
  return nullptr;
}

ORT_STATUS_PTR PopulateTensorWithData(_Inout_ OrtValue* oval, _In_reads_(num_elems) const std::string* data_elem,
                                      size_t num_elems, size_t /* elem_size */) {
  auto v = reinterpret_cast<OrtValue*>(oval);
  auto tensor = v->GetMutable<Tensor>();
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (num_elems < len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i < len; ++i) {
    dst[i] = data_elem[i];
  }
  return nullptr;
}

namespace c_api_internal {
template <class TensorElemType>
struct CallGetValueImpl {
  ORT_STATUS_PTR operator()(_Inout_ OrtAllocator* allocator, const onnxruntime::Tensor& tensor,
                            _Outptr_ OrtValue** out) const {
    const auto& shape = tensor.Shape();
    const auto* tensor_data = tensor.Data<TensorElemType>();
    OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, shape.GetDims().data(), shape.NumDimensions(),
                                                    onnxruntime::utils::GetONNXTensorElementDataType<TensorElemType>(), out);
    //TODO: check overflow before doing static_cast
    return st ? st : PopulateTensorWithData(*out, tensor_data, static_cast<size_t>(shape.Size()), sizeof(TensorElemType));
  }
};

// Return status instead of throwing if unsupported type specified
struct UnsupportedReturnFailStatus {
  ORT_STATUS_PTR operator()(int32_t dt_type) const {
    std::string msg("Unsupported tensor element type in the input: ");
    msg.append(std::to_string(dt_type));
    return OrtApis::CreateStatus(ORT_FAIL, msg.c_str());
  }
};
}  // namespace c_api_internal
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6101)
#endif
ORT_STATUS_PTR OrtGetValueImplSeqOfTensors(_In_ const OrtValue* p_ml_value, int index, _In_opt_ OrtAllocator* allocator,
                                           _Outptr_ OrtValue** out) {
  auto& data = p_ml_value->Get<TensorSeq>();
  auto& one_tensor = data.Get(index);

  using namespace c_api_internal;
  utils::MLTypeCallDispatcherRet<OrtStatusPtr, CallGetValueImpl, float, double, MLFloat16, BFloat16, bool, std::string,
                                 int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
      t_disp(one_tensor.GetElementType());
  return t_disp.template InvokeWithUnsupportedPolicy<UnsupportedReturnFailStatus>(allocator, one_tensor, out);
}

#ifdef _MSVC_VER
#pragma warning(pop)
#endif

static ORT_STATUS_PTR OrtGetValueImplSeq(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                         _Outptr_ OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  if (type->IsTensorSequenceType()) {
    return OrtGetValueImplSeqOfTensors(p_ml_value, index, allocator, out);
  } else {
    utils::ContainerChecker c_checker(type);
    if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
      return OrtGetValueImplSeqOfMap<VectorMapStringToFloat>(p_ml_value, index, out);
    } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
      return OrtGetValueImplSeqOfMap<VectorMapInt64ToFloat>(p_ml_value, index, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
    }
  }
}

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
  switch (index) {
    case 0: {  // user is requesting keys
      std::vector<TKey> vec;
      vec.reserve(static_cast<size_t>(num_kv_pairs));
      for (const auto& kv : data) {
        vec.push_back(kv.first);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TKey>(), out);
      return st ? st : PopulateTensorWithData(*out, vec.data(), static_cast<size_t>(num_kv_pairs), sizeof(TKey));
    }
    case 1: {  // user is requesting values
      std::vector<TVal> vec;
      vec.reserve(static_cast<size_t>(num_kv_pairs));
      for (const auto& kv : data) {
        vec.push_back(kv.second);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TVal>(), out);
      return st ? st : PopulateTensorWithData(*out, vec.data(), static_cast<size_t>(num_kv_pairs), sizeof(TVal));
    }
    default:
      return OrtApis::CreateStatus(ORT_FAIL, "Invalid index requested for map type.");
  }
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

static ORT_STATUS_PTR OrtGetValueImpl(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                      _Outptr_ OrtValue** out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
    return OrtGetValueImplMap(value, index, allocator, out);
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
template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelperMap(const OrtValue* const* in, size_t num_values,
                                                 _Outptr_ OrtValue** out) {
  using SeqType = std::vector<T>;
  auto seq_ptr = onnxruntime::make_unique<SeqType>();
  seq_ptr->reserve(num_values);
  for (size_t idx = 0; idx < num_values; ++idx) {
    auto& m = reinterpret_cast<const OrtValue*>(in[idx])->Get<T>();
    seq_ptr->push_back(m);
  }
  // create OrtValue with this vector
  auto value = onnxruntime::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<SeqType>();
  value->Init(seq_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename TensorElemType>
static OrtStatus* OrtCreateValueImplSeqHelperTensor(const Tensor& tensor,
                                                    Tensor& out) {
  auto data = tensor.Data<TensorElemType>();
  if (!data) {
    return OrtApis::CreateStatus(ORT_FAIL, "Encountered nullptr.");
  }

  auto elem_type = DataTypeImpl::GetType<TensorElemType>();
  OrtStatus* st = CreateTensorImplForSeq(elem_type, tensor.Shape().GetDims().data(), tensor.Shape().NumDimensions(), out);
  if (st) {
    return st;
  }

  //TODO: check the cast below
  size_t num_elems = static_cast<size_t>(tensor.Shape().Size());
  auto* out_data = out.MutableData<TensorElemType>();
  for (size_t i = 0; i < num_elems; ++i) {
    *out_data++ = *data++;
  }
  return nullptr;
}

namespace c_api_internal {

template <class T>
struct CallCreateValueImpl {
  OrtStatus* operator()(const onnxruntime::Tensor& one_tensor, onnxruntime::Tensor& out) const {
    return OrtCreateValueImplSeqHelperTensor<T>(one_tensor, out);
  }
};

}  // namespace c_api_internal

static ORT_STATUS_PTR OrtCreateValueImplSeqHelper(const OrtValue* const* in, size_t num_values,
                                                  _Outptr_ OrtValue** out) {
  using namespace c_api_internal;
  std::vector<Tensor> tensors;
  tensors.resize(num_values);
  auto dtype = static_cast<const OrtValue*>(in[0])->Get<Tensor>().DataType();

  for (size_t idx = 0; idx < num_values; ++idx) {
    ORT_ENFORCE(in[idx]->IsTensor(), "Expecting all elements to be tensors. Got: ", DataTypeImpl::ToString(in[idx]->Type()));
    auto& one_tensor = static_cast<const OrtValue*>(in[idx])->Get<Tensor>();
    auto tensor_elem_type = one_tensor.DataType();

    // sequences must have tensors of the same data type
    if (idx > 0 && (tensor_elem_type != dtype)) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "Sequences must have tensors of the same data type. There was at least one tensor in the input that was different.");
    }

    OrtStatus* st{};
    utils::MLTypeCallDispatcherRet<OrtStatus*, CallCreateValueImpl, bool, float, double, std::string,
                                   MLFloat16, BFloat16, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
        t_disp(one_tensor.GetElementType());

    st = t_disp.InvokeWithUnsupportedPolicy<UnsupportedReturnFailStatus>(one_tensor, tensors[idx]);

    if (st) {
      return st;
    }
  }
  // create OrtValue with this vector
  auto value = onnxruntime::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<TensorSeq>();
  auto seq_ptr = onnxruntime::make_unique<TensorSeq>(dtype);
  seq_ptr->SetElements(std::move(tensors));
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
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Unsupported input type");
  }
}

template <typename KeyType, typename ValueType>
static OrtStatus* OrtCreateMapMLValue(const Tensor& key_tensor, const Tensor& value_tensor, _Outptr_ OrtValue** out) {
  using MapType = std::map<KeyType, ValueType>;
  auto map_ptr = onnxruntime::make_unique<MapType>();
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
  auto value = onnxruntime::make_unique<OrtValue>();
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

static ORT_STATUS_PTR OrtCreateValueImpl(_In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                                         enum ONNXType value_type, _Outptr_ OrtValue** out) {
  if (num_values <= 0) {
    return OrtApis::CreateStatus(ORT_FAIL, "Number of values should be at least 1.");
  }
  if (value_type == ONNX_TYPE_MAP) {
    return OrtCreateValueImplMap(in, num_values, out);
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
  const auto* non_tensor_base = ml_type->AsNonTensorTypeBase();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  std::unique_ptr<OrtValue> ort_val(new OrtValue);
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
  const auto* non_tensor_base = ml_type->AsNonTensorTypeBase();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  non_tensor_base->ToDataContainer(*in, data_container_size, data_container);
  API_IMPL_END
  return nullptr;
}

// End support for non-tensor types

static constexpr OrtApiBase ort_api_base = {
    &OrtApis::GetApi,
    &OrtApis::GetVersionString,
};

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

struct OrtApi1to2 {
  // Shipped as version 1 - DO NOT MODIFY

  /**
* \param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
*/
  OrtStatus*(ORT_API_CALL* CreateStatus)(OrtErrorCode code, _In_ const char* msg)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  OrtErrorCode(ORT_API_CALL* GetErrorCode)(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
 * \param status must not be NULL
 * \return The error message inside the `status`. Do not free the returned value.
 */
  const char*(ORT_API_CALL* GetErrorMessage)(_In_ const OrtStatus* status)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
     * \param out Should be freed by `OrtReleaseEnv` after use
     */
  ORT_API2_STATUS(CreateEnv, OrtLoggingLevel default_logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  /**
   * \param out Should be freed by `OrtReleaseEnv` after use
   */
  ORT_API2_STATUS(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                  OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  // Platform telemetry events are on by default since they are lightweight.  You can manually turn them off.
  ORT_API2_STATUS(EnableTelemetryEvents, _In_ const OrtEnv* env);
  ORT_API2_STATUS(DisableTelemetryEvents, _In_ const OrtEnv* env);

  // TODO: document the path separator convention? '/' vs '\'
  // TODO: should specify the access characteristics of model_path. Is this read only during the
  // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
  // and continue to access throughout the OrtSession lifetime?
  //  What sort of access is needed to model_path : read or read/write?
  ORT_API2_STATUS(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(input_len) const char* const* input_names,
                  _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                  _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                  _Inout_updates_all_(output_names_len) OrtValue** output);

  /**
    * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseSessionOptions after use
    */
  ORT_API2_STATUS(CreateSessionOptions, _Outptr_ OrtSessionOptions** options);

  // Set filepath to save optimized model after graph level transformations.
  ORT_API2_STATUS(SetOptimizedModelFilePath, _Inout_ OrtSessionOptions* options,
                  _In_ const ORTCHAR_T* optimized_model_filepath);

  // create a copy of an existing OrtSessionOptions
  ORT_API2_STATUS(CloneSessionOptions, _In_ const OrtSessionOptions* in_options,
                  _Outptr_ OrtSessionOptions** out_options);

  // Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
  // has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
  // See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
  ORT_API2_STATUS(SetSessionExecutionMode, _Inout_ OrtSessionOptions* options, ExecutionMode execution_mode);

  // Enable profiling for this session.
  ORT_API2_STATUS(EnableProfiling, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
  ORT_API2_STATUS(DisableProfiling, _Inout_ OrtSessionOptions* options);

  // Enable the memory pattern optimization.
  // The idea is if the input shapes are the same, we could trace the internal memory allocation
  // and generate a memory pattern for future request. So next time we could just do one allocation
  // with a big chunk for all the internal memory allocation.
  // Note: memory pattern optimization is only available when SequentialExecution enabled.
  ORT_API2_STATUS(EnableMemPattern, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableMemPattern, _Inout_ OrtSessionOptions* options);

  // Enable the memory arena on CPU
  // Arena may pre-allocate memory for future usage.
  // set this option to false if you don't want it.
  ORT_API2_STATUS(EnableCpuMemArena, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableCpuMemArena, _Inout_ OrtSessionOptions* options);

  // < logger id to use for session output
  ORT_API2_STATUS(SetSessionLogId, _Inout_ OrtSessionOptions* options, const char* logid);

  // < applies to session load, initialization, etc
  ORT_API2_STATUS(SetSessionLogVerbosityLevel, _Inout_ OrtSessionOptions* options, int session_log_verbosity_level);
  ORT_API2_STATUS(SetSessionLogSeverityLevel, _Inout_ OrtSessionOptions* options, int session_log_severity_level);

  ORT_API2_STATUS(SetSessionGraphOptimizationLevel, _Inout_ OrtSessionOptions* options,
                  GraphOptimizationLevel graph_optimization_level);

  // Sets the number of threads used to parallelize the execution within nodes
  // A value of 0 means ORT will pick a default
  ORT_API2_STATUS(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);

  // Sets the number of threads used to parallelize the execution of the graph (across nodes)
  // If sequential execution is enabled this value is ignored
  // A value of 0 means ORT will pick a default
  ORT_API2_STATUS(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

  /*
  Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
  */
  ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

  /*
     * Add custom ops to the OrtCustomOpDomain
     *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
    */
  ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op);

  /*
     * Add a custom op domain to the OrtSessionOptions
     *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
    */
  ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);

  /*
     * Loads a DLL named 'library_path' and looks for this entry point:
     *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
     * It then passes in the provided session options to this function along with the api base.
     * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
     * session options are destroyed, or if an error occurs and it is non null.
  */
  ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path,
                  void** library_handle);

  /**
    * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
    * functions to enable them in the session:
    *   OrtSessionOptionsAppendExecutionProvider_CPU
    *   OrtSessionOptionsAppendExecutionProvider_CUDA
    *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
    * The order they care called indicates the preference order as well. In other words call this method
    * on your most preferred execution provider first followed by the less preferred ones.
    * If none are called Ort will use its internal CPU execution provider.
    */

  ORT_API2_STATUS(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);

  /**
   * \param out  should be freed by OrtReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param out  should be freed by OrtReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
 * \param out  should be freed by OrtReleaseTypeInfo after use
 */
  ORT_API2_STATUS(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /**
   * \return A pointer to the newly created object. The pointer should be freed by OrtReleaseRunOptions after use
   */
  ORT_API2_STATUS(CreateRunOptions, _Outptr_ OrtRunOptions** out);

  ORT_API2_STATUS(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunTag, _Inout_ OrtRunOptions*, _In_ const char* run_tag);

  ORT_API2_STATUS(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

  // Set a flag so that ALL incomplete OrtRun calls that are using this instance of OrtRunOptions
  // will exit as soon as possible.
  ORT_API2_STATUS(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
  // Unset the terminate flag to enable this OrtRunOptions instance being used in new OrtRun calls.
  ORT_API2_STATUS(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

  /**
   * Create a tensor from an allocator. OrtReleaseValue will also release the buffer inside the output value
   * \param out Should be freed by calling OrtReleaseValue
   * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
   */
  ORT_API2_STATUS(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* shape, size_t shape_len,
                  ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

  /**
   * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
   * p_data is owned by caller. OrtReleaseValue won't release p_data.
   * \param out Should be freed by calling OrtReleaseValue
   */
  ORT_API2_STATUS(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                  size_t p_data_len, _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                  _Outptr_ OrtValue** out);

  /**
   * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
   */
  ORT_API2_STATUS(IsTensor, _In_ const OrtValue* value, _Out_ int* out);

  // This function doesn't work with string tensor
  // this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
  ORT_API2_STATUS(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s each A string array. Each string in this array must be null terminated.
     * \param s_len length of s
     */
  ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param len total data length, not including the trailing '\0' chars.
     */
  ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);

  /**
     * \param s string contents. Each string is NOT null-terminated.
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s_len total data length, get it from OrtGetStringTensorDataLength
     */
  ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                  size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);

  /**
     * Don't free the 'out' value
     */
  ORT_API2_STATUS(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*,
                  _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);

  /**
     * Return OnnxType from OrtTypeInfo
     */
  ORT_API2_STATUS(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);

  /**
     * The 'out' value should be released by calling OrtReleaseTensorTypeAndShapeInfo
     */
  ORT_API2_STATUS(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  ORT_API2_STATUS(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);

  /**
 * \param info Created from CreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
  ORT_API2_STATUS(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  ORT_API2_STATUS(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*,
                  _Out_ enum ONNXTensorElementDataType* out);
  ORT_API2_STATUS(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
  ORT_API2_STATUS(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values,
                  size_t dim_values_length);
  ORT_API2_STATUS(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                  _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);

  /**
 * Return the number of elements specified by the tensor shape.
 * Return a negative value if unknown (i.e., any dimension is negative.)
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
  ORT_API2_STATUS(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

  /**
 * \param out Should be freed by OrtReleaseTensorTypeAndShapeInfo after use
 */
  ORT_API2_STATUS(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  /**
 * Get the type information of an OrtValue
 * \param value
 * \param out The returned value should be freed by OrtReleaseTypeInfo after use
 */
  ORT_API2_STATUS(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);

  ORT_API2_STATUS(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);

  ORT_API2_STATUS(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                  enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);

  /**
 * Convenience function for special case of CreateMemoryInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
 */
  ORT_API2_STATUS(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1,
                  _Outptr_ OrtMemoryInfo** out);

  /**
 * Test if two memory info are equal
 * \Sets 'out' to 0 if equal, -1 if not equal
 */
  ORT_API2_STATUS(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out);

  /**
 * Do not free the returned value
 */
  ORT_API2_STATUS(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
  ORT_API2_STATUS(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

  ORT_API2_STATUS(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
  ORT_API2_STATUS(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
  ORT_API2_STATUS(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out);

  // The returned pointer doesn't have to be freed.
  // Always returns the same instance on every invocation.
  ORT_API2_STATUS(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);

  // Override symbolic dimensions with actual values if known at session initialization time to enable
  // optimizations that can take advantage of fixed values (such as memory planning, etc)
  ORT_API2_STATUS(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* symbolic_dim,
                  _In_ int64_t dim_override);

  /**
   * APIs to support non-tensor types - map and sequence.
   * Currently only the following types are supported
   * Note: the following types should be kept in sync with data_types.h
   * Map types
   * =========
   * std::map<std::string, std::string>
   * std::map<std::string, int64_t>
   * std::map<std::string, float>
   * std::map<std::string, double>
   * std::map<int64_t, std::string>
   * std::map<int64_t, int64_t>
   * std::map<int64_t, float>
   * std::map<int64_t, double>
   * 
   * Sequence types
   * ==============
   * std::vector<std::string>
   * std::vector<int64_t>
   * std::vector<float>
   * std::vector<double>
   * std::vector<std::map<std::string, float>>
   * std::vector<std::map<int64_t, float>
   */

  /**
   * If input OrtValue represents a map, you need to retrieve the keys and values
   * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
   * If input OrtValue represents a sequence, use index to retrieve the index'th element
   * of the sequence.
   */
  ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ OrtValue** out);

  /**
   * Returns 2 for type map and N for sequence where N is the number of elements
   * in the sequence.
   */
  ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);

  /**
   * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
   * representing keys and values.
   * To construct a sequence, use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an arrary of N OrtValues.
   * \value_type should be either map or sequence.
   */
  ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                  enum ONNXType value_type, _Outptr_ OrtValue** out);

  /**
     * Construct OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization. OrtValue in this case would contain
     * an internal representation of the Opaque type. Opaque types are distinguished between
     * each other by two strings 1) domain and 2) type name. The combination of the two
     * must be unique, so the type representation is properly identified internally. The combination
     * must be properly registered from within ORT at both compile/run time or by another API.
     *
     * To construct the OrtValue pass domain and type names, also a pointer to a data container
     * the type of which must be know to both ORT and the client program. That data container may or may
     * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
     * verification purposes.
     *
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */
  ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                  _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);

  /**
     * Fetch data from an OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization.
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */

  ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in,
                  _Out_ void* data_container, size_t data_container_size);

  ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ float* out);
  ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ int64_t* out);
  ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out,
                  _Inout_ size_t* size);

  ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index,
                  _Out_ const OrtValue** out);
  ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index,
                  _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);

  ORT_CLASS_RELEASE(Env);
  ORT_CLASS_RELEASE(Status);  // nullptr for Status* indicates success
  ORT_CLASS_RELEASE(MemoryInfo);
  ORT_CLASS_RELEASE(Session);  //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
  ORT_CLASS_RELEASE(Value);
  ORT_CLASS_RELEASE(RunOptions);
  ORT_CLASS_RELEASE(TypeInfo);
  ORT_CLASS_RELEASE(TensorTypeAndShapeInfo);
  ORT_CLASS_RELEASE(SessionOptions);
  ORT_CLASS_RELEASE(CustomOpDomain);

  // End of Version 1 - DO NOT MODIFY ABOVE

  // Shipped as version 2 - DO NOT MODIFY

  /**
    * GetDenotationFromTypeInfo
     * This api augments OrtTypeInfo to return denotations on the type.
     * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
    */
  ORT_API2_STATUS(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation,
                  _Out_ size_t* len);

  // OrtTypeInfo Casting methods

  /**
    * CastTypeInfoToMapTypeInfo
     * This api augments OrtTypeInfo to return an OrtMapTypeInfo when the type is a map.
     * The OrtMapTypeInfo has additional information about the map's key type and value type.
     * This is used by WinML to support model reflection APIs.
     * This is used by WinML to support model reflection APIs.
     *
     * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtMapTypeInfo** out);

  /**
    * CastTypeInfoToSequenceTypeInfo
     * This api augments OrtTypeInfo to return an OrtSequenceTypeInfo when the type is a sequence.
     * The OrtSequenceTypeInfo has additional information about the sequence's element type.
    * This is used by WinML to support model reflection APIs.
     *
     * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

  // OrtMapTypeInfo Accessors

  /**
    * GetMapKeyType
     * This api augments get the key type of a map. Key types are restricted to being scalar types and use ONNXTensorElementDataType.
     * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);

  /**
    * GetMapValueType
     * This api augments get the value type of a map.
    */
  ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

  // OrtSequenceTypeInfo Accessors

  /**
    * GetSequenceElementType
     * This api augments get the element type of a sequence.
     * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info,
                  _Outptr_ OrtTypeInfo** type_info);

  ORT_CLASS_RELEASE(MapTypeInfo);
  ORT_CLASS_RELEASE(SequenceTypeInfo);

  /**
   * \param out is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * Profiling is turned ON automatically if enabled for the particular session by invoking EnableProfiling() 
   * on the SessionOptions instance used to create the session.  
   */
  ORT_API2_STATUS(SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator, _Outptr_ char** out);

  /**
   * \param out is a pointer to the newly created object. The pointer should be freed by calling ReleaseModelMetadata after use.
   */
  ORT_API2_STATUS(SessionGetModelMetadata, _In_ const OrtSession* sess, _Outptr_ OrtModelMetadata** out);

  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * 'value' will be a nullptr if the given key is not found in the custom metadata map.
   */
  ORT_API2_STATUS(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

  ORT_API2_STATUS(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata, _Out_ int64_t* value);

  ORT_CLASS_RELEASE(ModelMetadata);

  // End of Version 2 - DO NOT MODIFY ABOVE

  // Version 3 - In development, feel free to add/remove/rearrange here

  /*
  * Creates an environment with global threadpools that will be shared across sessions.
  * Use this in conjunction with DisablePerSessionThreads API or else the session will use
  * its own thread pools.
  */
  ORT_API2_STATUS(CreateEnvWithGlobalThreadPools, OrtLoggingLevel default_logging_level, _In_ const char* logid,
                  _In_ const OrtThreadingOptions* t_options, _Outptr_ OrtEnv** out);

  /* TODO: Should there be a version of CreateEnvWithGlobalThreadPools with custom logging function? */

  /*
  * Calling this API will make the session use the global threadpools shared across sessions.
  * This API should be used in conjunction with CreateEnvWithGlobalThreadPools API.
  */
  ORT_API2_STATUS(DisablePerSessionThreads, _Inout_ OrtSessionOptions* options);

  ORT_API2_STATUS(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);

  ORT_CLASS_RELEASE(ThreadingOptions);

  /**
   * \param num_keys contains the number of keys in the custom metadata map
   * \param keys is an array of null terminated strings (array count = num_keys) allocated using 'allocator'. 
   * The caller is responsible for freeing each string and the pointer array.
   * 'keys' will be a nullptr if custom metadata map is empty.
   */
  ORT_API2_STATUS(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);
  // End of Version 2 - DO NOT MODIFY ABOVE
};

static constexpr OrtApi1to2 ort_api_1_to_2 = {
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
    &OrtApis::GetTensorShapeElementCountV1,
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

    // Shipped as version 2 - DO NOT MODIFY (see above text for more information)
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
    &OrtApis::ReleaseModelMetadata
    // End of Version 2 - DO NOT MODIFY ABOVE (see above text for more information)

};

static constexpr OrtApi ort_api_3 = {
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
    &OrtApis::GetTensorShapeElementCountV2,
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

    // Shipped as version 2 - DO NOT MODIFY (see above text for more information)
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

    // Version 3 - In development, feel free to add/remove/rearrange here
    &OrtApis::CreateEnvWithGlobalThreadPools,
    &OrtApis::DisablePerSessionThreads,
    &OrtApis::CreateThreadingOptions,
    &OrtApis::ReleaseThreadingOptions,
    &OrtApis::ModelMetadataGetCustomMetadataMapKeys};

// Assert to do a limited check to ensure Version 1 and version 2 of OrtApi never changes (will detect an addition or deletion but not if they cancel out each other)
// If this assert hits, read the above 'Rules on how to add a new Ort API version'
static_assert(offsetof(OrtApi1to2, ReleaseCustomOpDomain) / sizeof(void*) == 101, "Size of version 1 API cannot change");
static_assert(offsetof(OrtApi1to2, ReleaseModelMetadata) / sizeof(void*) == 118, "Size of version 2 API cannot change");

static_assert(offsetof(OrtApi, ReleaseCustomOpDomain) / sizeof(void*) == 101, "Size of version 1 API cannot change");
static_assert(offsetof(OrtApi, ReleaseModelMetadata) / sizeof(void*) == 118, "Size of version 2 API cannot change");


ORT_API(const OrtApi*, OrtApis::GetApi, uint32_t version) {
  if (version == 1 || version == 2)
    return reinterpret_cast<const OrtApi*>(&ort_api_1_to_2);

  if (version == 3)
    return &ort_api_3;

  return nullptr;  // Unsupported version
}

ORT_API(const char*, OrtApis::GetVersionString) {
  return ORT_VERSION;
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

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif
