// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include <cassert>
#include <cstring>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
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
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"

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

#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status) return _status;      \
  } while (0)

struct OrtEnv {
 public:
  Environment* value;
  LoggingManager* loggingManager;

  OrtEnv(Environment* value1, LoggingManager* loggingManager1) : value(value1), loggingManager(loggingManager1) {
  }
  /**
   * This function will call ::google::protobuf::ShutdownProtobufLibrary
   */
  ~OrtEnv() {
    delete loggingManager;
    delete value;
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEnv);
};

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

class LoggingWrapper : public ISink {
 public:
  LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
      : logging_function_(logging_function), logger_param_(logger_param) {
  }

  void SendImpl(const Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                const Capture& message) override {
    std::string s = message.Location().ToString();
    logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                      logger_id.c_str(), s.c_str(), message.Message().c_str());
  }

 private:
  OrtLoggingFunction logging_function_;
  void* logger_param_;
};

ORT_API(const char*, OrtGetVersionString) {
  return ORT_VERSION;
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  std::unique_ptr<ISink> logger = onnxruntime::make_unique<LoggingWrapper>(logging_function, logger_param);
  auto default_logging_manager = onnxruntime::make_unique<LoggingManager>(std::move(logger),
                                                                  static_cast<Severity>(default_warning_level), false,
                                                                  LoggingManager::InstanceType::Default,
                                                                  &name);
  std::unique_ptr<Environment> env;
  Status status = Environment::Create(env);
  if (status.IsOK())
    *out = new OrtEnv(env.release(), default_logging_manager.release());
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnv, OrtLoggingLevel default_warning_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  auto default_logging_manager = onnxruntime::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                                  static_cast<Severity>(default_warning_level), false,
                                                                  LoggingManager::InstanceType::Default,
                                                                  &name);
  std::unique_ptr<Environment> env;
  Status status = Environment::Create(env);
  if (status.IsOK()) {
    *out = new OrtEnv(env.release(), default_logging_manager.release());
    return nullptr;
  }
  *out = nullptr;
  return ToOrtStatus(status);
  API_IMPL_END
}

template <typename T>
OrtStatus* CreateTensorImpl(const int64_t* shape, size_t shape_len, OrtAllocator* allocator,
                            std::unique_ptr<Tensor>* out) {
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    shapes[i] = shape[i];
  }
  std::shared_ptr<IAllocator> alloc_ptr = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  *out = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<T>(), onnxruntime::TensorShape(shapes), alloc_ptr);
  return nullptr;
}

/**
 *
 * this function will create a copy of the allocator info
 */
template <typename T>
OrtStatus* CreateTensorImpl(const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                            void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(sizeof(T), elem_count, &size_to_allocate)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "size overflow");
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<T>(), onnxruntime::TensorShape(shapes), p_data, *info);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<float>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<std::string>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<bool>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<MLFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<BFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<double>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
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
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
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
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<float>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<int64_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<std::string>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<bool>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<MLFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<BFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<double>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(CreateTensorImpl<uint64_t>(shape, shape_len, allocator, &tensor));
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
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
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

ORT_API(void, OrtApis::ReleaseCustomOpDomain, OrtCustomOpDomain* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CustomOpDomain_Add, _In_ OrtCustomOpDomain* custom_op_domain, OrtCustomOp* op) {
  API_IMPL_BEGIN
  custom_op_domain->custom_ops_.emplace_back(op);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddCustomOpDomain, _In_ OrtSessionOptions* options, OrtCustomOpDomain* custom_op_domain) {
  API_IMPL_BEGIN
  options->custom_op_domains_.emplace_back(custom_op_domain);
  return nullptr;
  API_IMPL_END
}

namespace {
template <typename Loader>
OrtStatus* CreateSessionImpl(_In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                             Loader loader, _Outptr_ OrtSession** out) {
  auto sess = onnxruntime::make_unique<::onnxruntime::InferenceSession>(
      options == nullptr ? onnxruntime::SessionOptions() : options->value, env->loggingManager);
  Status status;
  if (options != nullptr) {
    if (!options->custom_op_domains_.empty()) {
      status = sess->AddCustomOpDomains(options->custom_op_domains_);
      if (!status.IsOK())
        return ToOrtStatus(status);
    }
  }

  if (options != nullptr)
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      if (provider)
        sess->RegisterExecutionProvider(std::move(provider));
    }
  status = loader(*sess);
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
  const auto loader = [model_path](InferenceSession& sess) {
    return sess.Load(model_path);
  };
  return CreateSessionImpl(env, options, loader, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  const auto loader = [model_data, model_data_length](InferenceSession& sess) {
    return sess.Load(model_data, static_cast<int>(model_data_length));
  };
  return CreateSessionImpl(env, options, loader, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Run, _Inout_ OrtSession* sess,
                    _In_opt_ const OrtRunOptions* run_options,
                    _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
                    _In_ const char* const* output_names1, size_t output_names_len, _Outptr_ OrtValue** output) {
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

ORT_API_STATUS_IMPL(OrtApis::IsTensor, _In_ const OrtValue* value, int* out) {
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

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorContent, _In_ const OrtValue* value,
                    _Out_ void* s, size_t s_len, _Out_ size_t* offsets, size_t offsets_len) {
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

static OrtStatus* GetNodeDefListCountHelper(const OrtSession* sess, GetDefListFn get_fn, size_t* out) {
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

static OrtStatus* GetNodeDefTypeInfoHelper(const OrtSession* sess, GetDefListFn get_fn, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second->size() <= index)
    return OrtApis::CreateStatus(ORT_FAIL, "out of index");
  const ONNX_NAMESPACE::TypeProto* type_proto = (*p.second)[index]->TypeAsProto();
  return OrtTypeInfo::FromDataTypeImpl(type_proto, out);
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

static char* StrDup(const std::string& str, OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static OrtStatus* GetNodeDefNameImpl(_In_ const OrtSession* sess, size_t index,
                                     _Inout_ OrtAllocator* allocator, GetDefListFn get_fn,
                                     _Outptr_ char** output) {
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

///////////////////////////////////////////////////////////////////////////
// Code to handle non-tensor types
// OrtGetValueCount
// OrtGetVaue
// OrtCreateValue
///////////////////////////////////////////////////////////////////////////
const int NUM_MAP_INDICES = 2;

////////////////////
// OrtGetValueCount
template <typename T>
OrtStatus* OrtGetNumSequenceElements(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<T>();
  *out = data.size();
  return nullptr;
}

static OrtStatus* OrtGetValueCountImpl(const OrtValue* value, size_t* out) {
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
    if (type == DataTypeImpl::GetType<VectorString>()) {
      return OrtGetNumSequenceElements<VectorString>(v, out);
    }
    if (type == DataTypeImpl::GetType<VectorInt64>()) {
      return OrtGetNumSequenceElements<VectorInt64>(v, out);
    } else if (type == DataTypeImpl::GetType<VectorFloat>()) {
      return OrtGetNumSequenceElements<VectorFloat>(v, out);
    } else if (type == DataTypeImpl::GetType<VectorDouble>()) {
      return OrtGetNumSequenceElements<VectorDouble>(v, out);
    } else if (type == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
      return OrtGetNumSequenceElements<VectorMapStringToFloat>(v, out);
    } else if (type == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
      return OrtGetNumSequenceElements<VectorMapInt64ToFloat>(v, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
    }
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetValueCount, const OrtValue* value, size_t* out) {
  API_IMPL_BEGIN
  return OrtGetValueCountImpl(value, out);
  API_IMPL_END
}

///////////////////
// OrtGetValue
template <typename T>
static OrtStatus* OrtGetValueImplSeqOfMap(const OrtValue* p_ml_value, int index, OrtValue** out) {
  using TKey = typename T::value_type::key_type;
  using TVal = typename T::value_type::mapped_type;
  using MapType = std::map<TKey, TVal>;
  auto& data_vec = p_ml_value->Get<T>();
  auto& data_elem = data_vec.at(index);
  auto copy_data_elem = onnxruntime::make_unique<MapType>(data_elem);
  auto value = onnxruntime::make_unique<OrtValue>();
  value->Init(copy_data_elem.release(),
              DataTypeImpl::GetType<MapType>(),
              DataTypeImpl::GetType<MapType>()->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename T>
ONNXTensorElementDataType GetONNXTensorElementDataType() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

template <>
ONNXTensorElementDataType GetONNXTensorElementDataType<std::string>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}

template <>
ONNXTensorElementDataType GetONNXTensorElementDataType<float>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

template <>
ONNXTensorElementDataType GetONNXTensorElementDataType<double>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

template <>
ONNXTensorElementDataType GetONNXTensorElementDataType<int64_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

template <typename T>
OrtStatus* PopulateTensorWithData(OrtValue* oval, const T* data_elem, size_t num_elems) {
  void* raw_data = nullptr;
  auto st = OrtApis::GetTensorMutableData(oval, &raw_data);
  if (st) {
    return st;
  }
  memcpy(raw_data, data_elem, sizeof(T) * num_elems);
  return nullptr;
}

template <>
OrtStatus* PopulateTensorWithData<std::string>(OrtValue* oval, const std::string* data_elem,
                                               size_t num_elems) {
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

template <typename T>
OrtStatus* OrtGetValueImplSeqOfPrimitives(const OrtValue* p_ml_value, int index, OrtAllocator* allocator,
                                          OrtValue** out) {
  using ElemType = typename T::value_type;
  auto& data = p_ml_value->Get<T>();
  auto& data_elem = data.at(index);
  std::vector<int64_t> dims = {1};
  OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                  GetONNXTensorElementDataType<ElemType>(), out);
  return st ? st : PopulateTensorWithData<ElemType>(*out, &data_elem, 1);
}

static OrtStatus* OrtGetValueImplSeq(const OrtValue* value, int index, OrtAllocator* allocator,
                                     OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  if (type == DataTypeImpl::GetType<VectorString>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorString>(p_ml_value, index, allocator, out);
  }
  if (type == DataTypeImpl::GetType<VectorInt64>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorInt64>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<VectorFloat>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorFloat>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<VectorDouble>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorDouble>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
    return OrtGetValueImplSeqOfMap<VectorMapStringToFloat>(p_ml_value, index, out);
  } else if (type == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
    return OrtGetValueImplSeqOfMap<VectorMapInt64ToFloat>(p_ml_value, index, out);
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
  }
}

template <typename T>
static OrtStatus* OrtGetValueImplMapHelper(const OrtValue* p_ml_value, int index, OrtAllocator* allocator,
                                           OrtValue** out) {
  using TKey = typename T::key_type;
  using TVal = typename T::mapped_type;
  auto& data = p_ml_value->Get<T>();
  int64_t num_kv_pairs = data.size();
  switch (index) {
    case 0: {  // user is requesting keys
      std::vector<TKey> vec;
      vec.reserve(num_kv_pairs);
      for (const auto& kv : data) {
        vec.push_back(kv.first);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TKey>(), out);
      return st ? st : PopulateTensorWithData<TKey>(*out, vec.data(), num_kv_pairs);
    }
    case 1: {  // user is requesting values
      std::vector<TVal> vec;
      vec.reserve(num_kv_pairs);
      for (const auto& kv : data) {
        vec.push_back(kv.second);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TVal>(), out);
      return st ? st : PopulateTensorWithData<TVal>(*out, vec.data(), num_kv_pairs);
    }
    default:
      return OrtApis::CreateStatus(ORT_FAIL, "Invalid index requested for map type.");
  }
}

static OrtStatus* OrtGetValueImplMap(const OrtValue* value, int index, OrtAllocator* allocator,
                                     OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  if (type == DataTypeImpl::GetType<MapStringToString>()) {
    return OrtGetValueImplMapHelper<MapStringToString>(p_ml_value, index, allocator, out);
  }
  if (type == DataTypeImpl::GetType<MapStringToInt64>()) {
    return OrtGetValueImplMapHelper<MapStringToInt64>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapStringToFloat>()) {
    return OrtGetValueImplMapHelper<MapStringToFloat>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapStringToDouble>()) {
    return OrtGetValueImplMapHelper<MapStringToDouble>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToString>()) {
    return OrtGetValueImplMapHelper<MapInt64ToString>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToInt64>()) {
    return OrtGetValueImplMapHelper<MapInt64ToInt64>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToFloat>()) {
    return OrtGetValueImplMapHelper<MapInt64ToFloat>(p_ml_value, index, allocator, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToDouble>()) {
    return OrtGetValueImplMapHelper<MapInt64ToDouble>(p_ml_value, index, allocator, out);
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
  }
}

static OrtStatus* OrtGetValueImpl(const OrtValue* value, int index, OrtAllocator* allocator,
                                  OrtValue** out) {
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

ORT_API_STATUS_IMPL(OrtApis::GetValue, const OrtValue* value, int index, OrtAllocator* allocator,
                    OrtValue** out) {
  API_IMPL_BEGIN
  return OrtGetValueImpl(value, index, allocator, out);
  API_IMPL_END
}

///////////////////
// OrtCreateValue
template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelperMap(const OrtValue* const* in, size_t num_values, OrtValue** out) {
  using SeqType = std::vector<T>;
  auto vec_ptr = onnxruntime::make_unique<SeqType>();
  vec_ptr->reserve(num_values);
  for (size_t idx = 0; idx < num_values; ++idx) {
    auto& m = reinterpret_cast<const OrtValue*>(in[idx])->Get<T>();
    vec_ptr->push_back(m);
  }
  // create OrtValue with this vector
  auto value = onnxruntime::make_unique<OrtValue>();
  value->Init(vec_ptr.release(),
              DataTypeImpl::GetType<SeqType>(),
              DataTypeImpl::GetType<SeqType>()->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelper(const OrtValue* const* in, size_t num_values, OrtValue** out) {
  using SeqType = std::vector<T>;
  auto vec_ptr = onnxruntime::make_unique<SeqType>();
  vec_ptr->reserve(num_values);
  for (size_t idx = 0; idx < num_values; ++idx) {
    auto& tensor = reinterpret_cast<const OrtValue*>(in[idx])->Get<Tensor>();
    auto data = tensor.Data<T>();
    if (!data) {
      return OrtApis::CreateStatus(ORT_FAIL, "Encountered nullptr.");
    }
    vec_ptr->push_back(*data);
  }
  // create OrtValue with this vector
  auto value = onnxruntime::make_unique<OrtValue>();
  value->Init(vec_ptr.release(),
              DataTypeImpl::GetType<SeqType>(),
              DataTypeImpl::GetType<SeqType>()->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

static OrtStatus* OrtCreateValueImplSeq(const OrtValue* const* in, size_t num_values, OrtValue** out) {
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
    auto vec_type = first_mlvalue->Get<Tensor>().DataType();
    if (vec_type == DataTypeImpl::GetType<std::string>()) {
      return OrtCreateValueImplSeqHelper<std::string>(in, num_values, out);
    }
    if (vec_type == DataTypeImpl::GetType<int64_t>()) {
      return OrtCreateValueImplSeqHelper<int64_t>(in, num_values, out);
    } else if (vec_type == DataTypeImpl::GetType<float>()) {
      return OrtCreateValueImplSeqHelper<float>(in, num_values, out);
    } else if (vec_type == DataTypeImpl::GetType<double>()) {
      return OrtCreateValueImplSeqHelper<double>(in, num_values, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Type not supported.");
    }
  } else if (first_value_type == ONNX_TYPE_MAP) {
    auto map_type = first_mlvalue->Type();
    if (map_type == DataTypeImpl::GetType<MapStringToFloat>()) {
      return OrtCreateValueImplSeqHelperMap<MapStringToFloat>(in, num_values, out);
    }
    if (map_type == DataTypeImpl::GetType<MapInt64ToFloat>()) {
      return OrtCreateValueImplSeqHelperMap<MapInt64ToFloat>(in, num_values, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
    }
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Unsupported input type");
  }
}

template <typename KeyType, typename ValueType>
static OrtStatus* OrtCreateMapMLValue(const Tensor& key_tensor, const Tensor& value_tensor,
                                      OrtValue** out) {
  using MapType = std::map<KeyType, ValueType>;
  auto map_ptr = onnxruntime::make_unique<MapType>();
  // iterate through the key and value tensors and populate map
  auto key_data = key_tensor.Data<KeyType>();
  auto value_data = value_tensor.Data<ValueType>();
  size_t num_kv_pairs = key_tensor.Shape().Size();
  for (size_t n = 0; n < num_kv_pairs; ++n, ++key_data, ++value_data) {
    map_ptr->insert({*key_data, *value_data});
  }
  // create ort_value with this map
  auto value = onnxruntime::make_unique<OrtValue>();
  value->Init(map_ptr.release(),
              DataTypeImpl::GetType<MapType>(),
              DataTypeImpl::GetType<MapType>()->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename KeyType>
static OrtStatus* OrtCreateValueImplMapHelper(const Tensor& key_tensor, const Tensor& value_tensor,
                                              OrtValue** out) {
  auto value_type = value_tensor.DataType();
  if (value_type == DataTypeImpl::GetType<std::string>()) {
    return OrtCreateMapMLValue<KeyType, std::string>(key_tensor, value_tensor, out);
  }
  if (value_type == DataTypeImpl::GetType<int64_t>()) {
    return OrtCreateMapMLValue<KeyType, int64_t>(key_tensor, value_tensor, out);
  } else if (value_type == DataTypeImpl::GetType<float>()) {
    return OrtCreateMapMLValue<KeyType, float>(key_tensor, value_tensor, out);
  } else if (value_type == DataTypeImpl::GetType<double>()) {
    return OrtCreateMapMLValue<KeyType, double>(key_tensor, value_tensor, out);
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Value type is not supported yet.");
  }
}

static OrtStatus* OrtCreateValueImplMap(const OrtValue* const* in, size_t num_values, OrtValue** out) {
  if (num_values != NUM_MAP_INDICES) {
    return OrtApis::CreateStatus(ORT_FAIL, "For map type num_values MUST be 2");
  }

  const OrtValue* ort_keys = in[0];
  auto p_key_ml_value = reinterpret_cast<const OrtValue*>(ort_keys);
  auto& key_tensor = p_key_ml_value->Get<Tensor>();
  auto key_type = key_tensor.DataType();

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

  if (key_type == DataTypeImpl::GetType<std::string>()) {
    return OrtCreateValueImplMapHelper<std::string>(key_tensor, value_tensor, out);
  }
  if (key_type == DataTypeImpl::GetType<int64_t>()) {
    return OrtCreateValueImplMapHelper<int64_t>(key_tensor, value_tensor, out);
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Key type is not supported yet.");
}

static OrtStatus* OrtCreateValueImpl(const OrtValue* const* in, size_t num_values, enum ONNXType value_type, OrtValue** out) {
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

ORT_API_STATUS_IMPL(OrtApis::CreateValue, const OrtValue* const* in, size_t num_values, enum ONNXType value_type, OrtValue** out) {
  API_IMPL_BEGIN
  return OrtCreateValueImpl(in, num_values, value_type, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateOpaqueValue, const char* domain_name, const char* type_name, const void* data_container,
                    size_t data_container_size, OrtValue** out) {
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

ORT_API_STATUS_IMPL(OrtApis::GetOpaqueValue, const char* domain_name, const char* type_name, const OrtValue* in,
                    void* data_container, size_t data_container_size) {
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

static constexpr OrtApi ort_api_1 = {
    &OrtApis::CreateStatus,
    &OrtApis::GetErrorCode,
    &OrtApis::GetErrorMessage,

    &OrtApis::CreateEnv,
    &OrtApis::CreateEnvWithCustomLogger,
    &OrtApis::CreateSession,
    &OrtApis::CreateSessionFromArray,
    &OrtApis::Run,

    &OrtApis::CreateSessionOptions,
    &OrtApis::SetOptimizedModelFilePath,
    &OrtApis::CloneSessionOptions,
    &OrtApis::EnableSequentialExecution,
    &OrtApis::DisableSequentialExecution,
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
    &OrtApis::OrtAddFreeDimensionOverride,
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
};

const OrtApi* ORT_API_CALL OrtGetApi(uint32_t version) NO_EXCEPTION {
  if (version > 1)
    return nullptr;

  return &ort_api_1;
}

DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Env, OrtEnv)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Value, OrtValue)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(RunOptions, OrtRunOptions)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Session, ::onnxruntime::InferenceSession)
