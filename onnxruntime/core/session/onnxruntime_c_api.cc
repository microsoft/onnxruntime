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
#include "core/framework/environment.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"

using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::MLStatus;
using onnxruntime::MLValue;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

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

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                          \
  }                                                           \
  catch (std::exception & ex) {                               \
    return OrtCreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

#define TENSOR_READ_API_BEGIN                                      \
  API_IMPL_BEGIN                                                   \
  auto v = reinterpret_cast<const ::onnxruntime::MLValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN                           \
  API_IMPL_BEGIN                                             \
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value); \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

class LoggingWrapper : public ISink {
 public:
  LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
      : logging_function_{logging_function}, logger_param_{logger_param} {
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

ORT_API_STATUS_IMPL(OrtCreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid,
                    _Out_ OrtEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  std::unique_ptr<ISink> logger = std::make_unique<LoggingWrapper>(logging_function, logger_param);
  auto default_logging_manager = std::make_unique<LoggingManager>(std::move(logger),
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

ORT_API_STATUS_IMPL(OrtCreateEnv, OrtLoggingLevel default_warning_level,
                    _In_ const char* logid, _Out_ OrtEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  auto default_logging_manager = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
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

ORT_API_STATUS_IMPL(OrtGetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* out) {
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
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtFillStringTensor, _In_ OrtValue* value, _In_ const char* const* s, size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len < len) {
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i != len; ++i) {
    //allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

template <typename T>
OrtStatus* CreateTensorImpl(const size_t* shape, size_t shape_len, OrtAllocator* allocator,
                            std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(sizeof(T), elem_count, &size_to_allocate)) {
    return OrtCreateStatus(ORT_FAIL, "not enough memory");
  }
  void* p_data = allocator->Alloc(allocator, size_to_allocate);
  if (p_data == nullptr)
    return OrtCreateStatus(ORT_FAIL, "size overflow");
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes),
                                  static_cast<void*>(p_data),
                                  *allocator->Info(allocator),
                                  std::make_shared<onnxruntime::AllocatorWrapper>(allocator));
  return nullptr;
}

/**
 *
 * this function will create a copy of the allocator info
 */
template <typename T>
OrtStatus* CreateTensorImpl(const size_t* shape, size_t shape_len, const OrtAllocatorInfo* info,
                            void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(sizeof(T), elem_count, &size_to_allocate)) {
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "size overflow");
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes),
                                  p_data,
                                  *info,
                                  nullptr);
  return nullptr;
}

/**
 * this function will create a copy of the allocator info
 */
ORT_API_STATUS_IMPL(OrtCreateTensorWithDataAsOrtValue, _In_ const OrtAllocatorInfo* info,
                    _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Out_ OrtValue** out) {
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
      return OrtCreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const size_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Out_ OrtValue** out) {
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
      return OrtCreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
  API_IMPL_END
}

template <typename T>
static OrtStatus* CreateSessionImpl(_In_ OrtEnv* env, _In_ T model_path,
                                    _In_ const OrtSessionOptions* options,
                                    _Out_ OrtSession** out) {
  API_IMPL_BEGIN
  auto sess = std::make_unique<::onnxruntime::InferenceSession>(options == nullptr ? onnxruntime::SessionOptions() : options->value, env->loggingManager);
  Status status;
  if (options != nullptr && !options->custom_op_paths.empty()) {
    status = sess->LoadCustomOps(options->custom_op_paths);
    if (!status.IsOK())
      return ToOrtStatus(status);
  }
  if (options != nullptr)
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      if (provider)
        sess->RegisterExecutionProvider(std::move(provider));
    }
  status = sess->Load(model_path);
  if (!status.IsOK())
    return ToOrtStatus(status);
  status = sess->Initialize();
  if (!status.IsOK())
    return ToOrtStatus(status);
  *out = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
  API_IMPL_END
}

#ifdef _WIN32
ORT_API_STATUS_IMPL(OrtCreateSession, _In_ OrtEnv* env, _In_ const wchar_t* model_path,
                    _In_ const OrtSessionOptions* options, _Out_ OrtSession** out) {
  API_IMPL_BEGIN
  return CreateSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#else
ORT_API_STATUS_IMPL(OrtCreateSession, _In_ OrtEnv* env, _In_ const char* model_path,
                    _In_ const OrtSessionOptions* options, _Out_ OrtSession** out) {
  API_IMPL_BEGIN
  return CreateSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#endif

ORT_API_STATUS_IMPL(OrtRun, _In_ OrtSession* sess,
                    _In_ OrtRunOptions* run_options,
                    _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
                    _In_ const char* const* output_names1, size_t output_names_len, _Out_ OrtValue** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  ::onnxruntime::NameMLValMap in;
  const int queue_id = 0;
  for (size_t i = 0; i != input_len; ++i) {
    auto kvp = in.insert(std::make_pair(std::string(input_names[i]),
                                        *reinterpret_cast<const ::onnxruntime::MLValue*>(input[i])));
    if (!kvp.second) {
      return OrtCreateStatus(ORT_INVALID_ARGUMENT, "duplicated input name");
    }
    ::onnxruntime::MLValue& value = kvp.first->second;
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
  }
  // Create output feed
  std::vector<std::string> output_names(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output_names1[i] == nullptr || output_names1[i][0] == '\0') {
      return OrtCreateStatus(ORT_INVALID_ARGUMENT, "output name cannot be empty");
    }
    output_names[i] = output_names1[i];
  }

  std::vector<MLValue> fetches(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output[i] != nullptr) {
      ::onnxruntime::MLValue& value = *reinterpret_cast<::onnxruntime::MLValue*>(output[i]);
      if (value.Fence())
        value.Fence()->BeforeUsingAsOutput(onnxruntime::kCpuExecutionProvider, queue_id);
      fetches[i] = value;
    }
  }
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions op;
    status = session->Run(op, in, output_names, &fetches);
  } else {
    status = session->Run(*run_options, in, output_names, &fetches);
  }

  if (!status.IsOK())
    return ToOrtStatus(status);
  for (size_t i = 0; i != output_names_len; ++i) {
    ::onnxruntime::MLValue& value = fetches[i];
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (output[i] == nullptr) {
      output[i] = reinterpret_cast<OrtValue*>(new MLValue(value));
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtGetTensorMutableData, _In_ OrtValue* value, _Out_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  //TODO: test if it's a string tensor
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtGetStringTensorContent, _In_ const OrtValue* value,
                    _Out_ void* s, size_t s_len, _Out_ size_t* offsets, size_t offsets_len) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (offsets_len < len) {
    return OrtCreateStatus(ORT_FAIL, "space is not enough");
  }
  {
    size_t ret = 0;
    for (size_t i = 0; i != len; ++i) {
      ret += input[i].size();
    }
    if (s_len < ret) {
      return OrtCreateStatus(ORT_FAIL, "space is not enough");
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

ORT_API_STATUS_IMPL(OrtTensorProtoToOrtValue, _Inout_ OrtAllocator* allocator,
                    const void* input, int input_len, _Out_ OrtValue** out) {
  API_IMPL_BEGIN
  std::shared_ptr<onnxruntime::IAllocator> allocator_ = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  ::ONNX_NAMESPACE::TensorProto proto;
  if (!proto.ParseFromArray(input, input_len)) {
    return OrtCreateStatus(ORT_FAIL, "parse input tensor proto failed");
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  Status st = onnxruntime::utils::TensorProtoToMLValue(proto, allocator_, nullptr, 0, *value);
  if (!st.IsOK())
    return ToOrtStatus(st);
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
  API_IMPL_END
}

#define DEFINE_RELEASE_ORT_OBJECT_FUNCTION(INPUT_TYPE, REAL_TYPE) \
  ORT_API(void, OrtRelease##INPUT_TYPE, Ort##INPUT_TYPE* value) { \
    delete reinterpret_cast<REAL_TYPE*>(value);                   \
  }

#define DEFINE_RELEASE_ORT_OBJECT_FUNCTION_FOR_ARRAY(INPUT_TYPE, REAL_TYPE) \
  ORT_API(void, OrtRelease##INPUT_TYPE, Ort##INPUT_TYPE* value) {           \
    delete[] reinterpret_cast<REAL_TYPE*>(value);                           \
  }

ORT_API_STATUS_IMPL(OrtSessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelInputs();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelOutputs();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Out_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelInputs();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second->size() <= index)
    return OrtCreateStatus(ORT_FAIL, "out of index");
  const ONNX_NAMESPACE::TypeProto* type_proto = (*p.second)[index]->TypeAsProto();
  return OrtTypeInfo::FromDataTypeImpl(type_proto, out);
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtSessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Out_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelOutputs();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second->size() <= index)
    return OrtCreateStatus(ORT_FAIL, "out of index");
  const ONNX_NAMESPACE::TypeProto* type_proto = (*p.second)[index]->TypeAsProto();
  return OrtTypeInfo::FromDataTypeImpl(type_proto, out);
  API_IMPL_END
}

static char* StrDup(const std::string& str, OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static OrtStatus* GetInputOutputNameImpl(_In_ const OrtSession* sess, size_t index,
                                         _Inout_ OrtAllocator* allocator, bool is_input,
                                         _Out_ char** output) {
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = is_input ? session->GetModelInputs() : session->GetModelOutputs();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second == nullptr)
    return OrtCreateStatus(ORT_FAIL, "internal error");
  const InputDefList& defs = *p.second;
  if (index >= defs.size())
    return OrtCreateStatus(ORT_FAIL, "index out of range");
  *output = StrDup(defs[index]->Name(), allocator);
  return nullptr;
}

ORT_API(int, OrtIsTensor, _In_ const OrtValue* value) {
  auto v = reinterpret_cast<const ::onnxruntime::MLValue*>(value);
  return v->IsTensor() ? 1 : 0;
}

ORT_API(void*, OrtAllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size) {
  try {
    return ptr->Alloc(ptr, size);
  } catch (std::exception&) {
    return nullptr;
  }
}

ORT_API(void, OrtAllocatorFree, _Inout_ OrtAllocator* ptr, void* p) {
  try {
    ptr->Free(ptr, p);
  } catch (std::exception&) {
  }
}

ORT_API(const struct OrtAllocatorInfo*, OrtAllocatorGetInfo, _In_ const OrtAllocator* ptr) {
  try {
    return ptr->Info(ptr);
  } catch (std::exception&) {
    return nullptr;
  }
}

ORT_API_STATUS_IMPL(OrtSessionGetInputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Out_ char** output) {
  API_IMPL_BEGIN
  return GetInputOutputNameImpl(sess, index, allocator, true, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionGetOutputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Out_ char** output) {
  API_IMPL_BEGIN
  return GetInputOutputNameImpl(sess, index, allocator, false, output);
  API_IMPL_END
}

///////////////////////////////////////////////////////////////////////////
// Code to handle non-tensor types
// OrtGetNumValues
// OrtGetVaue
// OrtCreateValue
///////////////////////////////////////////////////////////////////////////

////////////////////
// OrtGetNumValues
template <typename T>
OrtStatus* OrtGetNumSequenceElements(const MLValue* p_ml_value, int* out) {
  auto& data = p_ml_value->Get<T>();
  *out = data.size();
  return nullptr;
}

static OrtStatus* OrtGetNumValuesImpl(const OrtValue* value, int* out) {
  auto value_type = OrtGetValueType(value);
  if (value_type == ONNX_TYPE_MAP) {
    *out = 2;
    return nullptr;
  } else if (value_type == ONNX_TYPE_SEQUENCE) {
    auto v = reinterpret_cast<const MLValue*>(value);
    auto type = v->Type();
    // maintenance nightmare: keep these in sync with the registered types in data_types.h
    if (type == DataTypeImpl::GetType<VectorString>()) {
      return OrtGetNumSequenceElements<VectorString>(v, out);
    } else if (type == DataTypeImpl::GetType<VectorInt64>()) {
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
      return OrtCreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
    }
  } else {
    return OrtCreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtGetNumValues, const OrtValue* value, int* out) {
  API_IMPL_BEGIN
  return OrtGetNumValuesImpl(value, out);
  API_IMPL_END
}

///////////////////
// OrtGetValue
template <typename T>
static OrtStatus* OrtGetValueImplSeqOfMap(const MLValue* p_ml_value, int index, OrtValue** out) {
  using TKey = typename T::key_type;
  using TVal = typename T::value_type;
  auto& data = p_ml_value->Get<T>();
  auto data_elem = std::make_unique<std::map<TKey, TVal>>(data.at(index));
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(data_elem.release(),
              DataTypeImpl::GetType<std::map<TKey, TVal>>(),
              DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
}

template <typename T>
static ONNXTensorElementDataType GetONNXTensorElementDataType() {
  if (std::is_same<T, std::string>::value) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  } else if (std::is_same<T, float>::value) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (std::is_same<T, int64_t>::value) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  // TODO add other types here
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

template <typename T>
static OrtStatus* OrtGetValueImplSeqOfPrimitives(const MLValue* p_ml_value, int index, OrtValue** out) {
  using ElemType = typename T::value_type;
  auto& data = p_ml_value->Get<T>();
  auto& data_elem = data.at(index);
  std::vector<int64_t> dims = {1};
  return OrtCreateTensorWithDataAsOrtValue(info, &data_elem, sizeof(ElemType), dims.size(), GetONNXTensorElementDataType<ElemType>(), out);
}

template <>
static OrtStatus* OrtGetValueImplSeqOfPrimitives<VectorString>(const MLValue* p_ml_value, int index, OrtValue** out) {
  auto& data = p_ml_value->Get<VectorString>();
  auto& data_elem = data.at(index);
  std::vector<int64_t> dims = {1};
  return OrtCreateTensorWithDataAsOrtValue(info, data_elem.data(), data_elem.size(), dims.size(), GetONNXTensorElementDataType<std::string>(), out);
}

static OrtStatus* OrtGetValueImplSeq(const OrtValue* value, int index, const OrtAllocatorInfo* info, OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const MLValue*>(value);
  auto type = p_ml_value->Type();
  // maintenance nightmare: keep these in sync with the registered types in data_types.h
  if (type == DataTypeImpl::GetType<VectorString>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorString>(p_ml_value, out);
  } else if (type == DataTypeImpl::GetType<VectorInt64>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorInt64>(p_ml_value, out);
  } else if (type == DataTypeImpl::GetType<VectorFloat>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorFloat>(p_ml_value, out);
  } else if (type == DataTypeImpl::GetType<VectorDouble>()) {
    return OrtGetValueImplSeqOfPrimitives<VectorDouble>(p_ml_value, out);
  } else if (type == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
    return OrtGetValueImplSeqOfMap<VectorMapStringToFloat>(p_ml_value, index, out);
  } else if (type == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
    return OrtGetValueImplSeqOfMap<VectorMapInt64ToFloat>(p_ml_value, index, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
  }
}

template <typename T>
static OrtStatus* OrtGetValueImplMapHelper(const MLValue* p_ml_value, int index, const OrtAllocatorInfo* info, OrtValue** out) {
  using TKey = typename T::key_type;
  using TVal = typename T::mapped_type;
  auto& data = p_ml_value->Get<T>();
  int num_kv_pairs = data.size();
  switch (index) {
    case 0: {  // user is requesting keys
      std::vector<TKey> keys;
      keys.reserve(num_kv_pairs);
      for (const auto& kv : data) {
        keys.push_back(kv.first);
      }
      std::vector<int64_t> dims = {num_kv_pairs};
      return OrtCreateTensorWithDataAsOrtValue(info, keys.data(), keys.size() * sizeof(TKey), dims.size(), GetONNXTensorElementDataType<TKey>(), out);
    }
    case 1: {  // user is requesting values
      std::vector<TKey> vals;
      vals.reserve(num_kv_pairs);
      for (const auto& kv : data) {
        vals.push_back(kv.second);
      }
      std::vector<int64_t> dims = {num_kv_pairs};
      return OrtCreateTensorWithDataAsOrtValue(info, vals.data(), vals.size() * sizeof(TVal), dims.size(), GetONNXTensorElementDataType<TVal>(), out);
    }
    default:
      return OrtCreateStatus(ORT_FAIL, "Invalid index requested for map type.");
  }
}

static OrtStatus* OrtGetValueImplMap(const OrtValue* value, int index, const OrtAllocatorInfo* info, OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const MLValue*>(value);
  auto type = p_ml_value->Type();
  // maintenance nightmare: keep these in sync with the registered types in data_types.h
  if (type == DataTypeImpl::GetType<MapStringToString>()) {
    return OrtGetValueImplMapHelper<MapStringToString>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapStringToInt64>()) {
    return OrtGetValueImplMapHelper<MapStringToInt64>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapStringToFloat>()) {
    return OrtGetValueImplMapHelper<MapStringToFloat>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapStringToDouble>()) {
    return OrtGetValueImplMapHelper<MapStringToDouble>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToString>()) {
    return OrtGetValueImplMapHelper<MapInt64ToString>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToInt64>()) {
    return OrtGetValueImplMapHelper<MapInt64ToInt64>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToFloat>()) {
    return OrtGetValueImplMapHelper<MapInt64ToFloat>(p_ml_value, index, info, out);
  } else if (type == DataTypeImpl::GetType<MapInt64ToDouble>()) {
    return OrtGetValueImplMapHelper<MapInt64ToDouble>(p_ml_value, index, info, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
  }
}

static OrtStatus* OrtGetValueImpl(const OrtValue* value, int index, const OrtAllocatorInfo* info, OrtValue** out) {
  auto value_type = OrtGetValueType(value);
  if (value_type == ONNX_TYPE_MAP) {
    return OrtGetValueImplMap(value, index, info, out);
  } else if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtGetValueImplSeq(value, index, info, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtGetValue, const OrtValue* value, int index, const OrtAllocatorInfo* info, OrtValue** out) {
  API_IMPL_BEGIN
  return OrtGetValueImpl(value, index, info, out);
  API_IMPL_END
}

///////////////////
// OrtCreateValue
template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelperMap(const OrtValue** in, int num_values, const OrtAllocatorInfo* info, OrtValue** out) {
  auto vec_ptr = std::make_unique<std::vector<T>>();
  vec_ptr->reserve(num_values);
  for (int idx = 0; idx < num_values; ++idx) {
    auto& m = reinterpret_cast<const MLValue*>(in[idx])->Get<T>();
    vec_ptr->push_back(m);
  }
  // create MLValue with this vector
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(vec_ptr.release(),
              DataTypeImpl::GetType<std::vector<T>>(),
              DataTypeImpl::GetType < std::vector<T>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
}

template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelper(const OrtValue** in, int num_values, const OrtAllocatorInfo* info, OrtValue** out) {
  auto vec_ptr = std::make_unique<std::vector<T>>();
  vec_ptr->reserve(num_values);
  for (int idx = 0; idx < num_values; ++idx) {
    auto& tensor = reinterpret_cast<const MLValue*>(in[idx])->Get<Tensor>();
    auto data = tensor.Data<VecElemType>();
    // TODO check for data nullptr?
    vec_ptr->push_back(*data);
  }
  // create MLValue with this vector
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(vec_ptr.release(),
              DataTypeImpl::GetType<std::vector<T>>(),
              DataTypeImpl::GetType < std::vector<T>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
}

static OrtStatus* OrtCreateValueImplSeq(const OrtValue** in, int num_values, const OrtAllocatorInfo* info, OrtValue** out) {
  // We only support limited sequence types. For the sake of simplicity the type of the first
  // OrtValue* in OrtValue** will determine the type of the vector used to create the output OrtValue
  // this type should be either a tensor of limited types or map of limited types
  const OrtValue* ovfirst = in[0];
  auto first_value_type = OrtGetValueType(ovfirst);
  // in onnxruntime type registrations we can support only a fixed vector types
  // this check ensures that the input conforms to that
  if (!(first_value_type == ONNX_TYPE_TENSOR || first_value_type == ONNX_TYPE_MAP)) {
    return OrtCreateStatus(ORT_FAIL, "Each element of the sequence should be either tensor or map.");
  }
  // check if all OrtValues in the input array are of the same type
  // this is because even though the ONNX spec and this API spec supports heterogenous sequences,
  // only a fixed types are registered in onnxruntime
  for (int i = 0; i < num_values; ++i) {
    const OrtValue* ov = in[i];
    auto ov_type = OrtGetValueType(ov);
    if (ov_type != first_value_type) {
      return OrtCreateStatus(ORT_FAIL, "At least one element in the sequence is of a type different from others.");
    }
  }

  // finally create the output vector/MLValue
  auto first_mlvalue = reinterpret_cast<const MLValue*>(ovfirst);
  if (first_value_type == ONNX_TYPE_TENSOR) {
    auto vec_type = first_mlvalue->Get<Tensor>().DataType();
    if (type == DataTypeImpl::GetType<std::string>()) {
      return OrtCreateValueImplSeqHelper<std::string>(in, num_values, info, out);
    } else if (type == DataTypeImpl::GetType<int64_t>()) {
      return OrtCreateValueImplSeqHelper<int64_t>(in, num_values, info, out);
    } else if (type == DataTypeImpl::GetType<float>()) {
      return OrtCreateValueImplSeqHelper<float>(in, num_values, info, out);
    } else if (type == DataTypeImpl::GetType<double>()) {
      return OrtCreateValueImplSeqHelper<double>(in, num_values, info, out);
    } else {
      return OrtCreateStatus("Type not supported.")
    }
  } else {  // map type
    // maintenance nightmare: keep these in sync with the registered types in data_types.h
    if (first_value_type == DataTypeImpl::GetType<MapStringToString>()) {
      OrtCreateValueImplSeqHelperMap<MapStringToString>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapStringToInt64>()) {
      OrtCreateValueImplSeqHelperMap<MapStringToInt64>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapStringToFloat>()) {
      OrtCreateValueImplSeqHelperMap<MapStringToFloat>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapStringToDouble>()) {
      OrtCreateValueImplSeqHelperMap<MapStringToDouble>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapInt64ToString>()) {
      OrtCreateValueImplSeqHelperMap<MapInt64ToString>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapInt64ToInt64>()) {
      OrtCreateValueImplSeqHelperMap<MapInt64ToInt64>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapInt64ToFloat>()) {
      OrtCreateValueImplSeqHelperMap<MapInt64ToFloat>(in, num_values, info, out);
    } else if (first_value_type == DataTypeImpl::GetType<MapInt64ToDouble>()) {
      OrtCreateValueImplSeqHelperMap<MapInt64ToDouble>(in, num_values, info, out);
    } else {
      return OrtCreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
    }
  }
}

template <typename KeyType, typename ValueType>
static OrtStatus* OrtCreateMapMLValue(const Tensor& key_tensor, const Tensor& value_tensor, const OrtAllocatorInfo* info, OrtValue** out) {
  auto map_ptr = std::make_unique<std::map<KeyType, ValueType>>();
  // iterate through the key and value tensors and populate map
  auto key_data = key_tensor.Data<KeyType>();
  auto value_data = value_tensor.Data<ValueType>();
  int num_kv_pairs = key_tensor.Shape().Size();
  for (int n = 0; n < num_kv_pairs; ++n, ++key_data, ++value_data) {
    map_ptr->insert({*key_data, *value_data});
  }
  // create mlvalue with this map
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(map_ptr.release(),
              DataTypeImpl::GetType<std::map<KeyType, ValueType>>(),
              DataTypeImpl::GetType<std::map<KeyType, ValueType>>()->GetDeleteFunc());
  *out = reinterpret_cast<OrtValue*>(value.release());
  return nullptr;
}

template <typename KeyType>
static OrtStatus* OrtCreateValueImplMapHelper(const Tensor& key_tensor, const Tensor& value_tensor, const OrtAllocatorInfo* info, OrtValue** out) {
  auto value_type = value_tensor.DataType();
  if (value_type == DataTypeImpl::GetType<std::string>()) {
    return OrtCreateMapMLValue<KeyType, std::string>(key_tensor, value_tensor, info, out);
  } else if (value_type == DataTypeImpl::GetType<int64_t>()) {
    return OrtCreateMapMLValue<KeyType, int64_t>(key_tensor, value_tensor, info, out);
  } else if (value_type == DataTypeImpl::GetType<float>()) {
    return OrtCreateMapMLValue<KeyType, float>(key_tensor, value_tensor, info, out);
  } else if (value_type == DataTypeImpl::GetType<double>()) {
    return OrtCreateMapMLValue<KeyType, double>(key_tensor, value_tensor, info, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Value type is not supported yet.");
  }
}

static OrtStatus* OrtCreateValueImplMap(const OrtValue** in, int num_values, const OrtAllocatorInfo* info, OrtValue** out) {
  if (num_values != 2) {
    return OrtCreateStatus(ORT_FAIL, "For map type num_values MUST be 2");
  }

  const OrtValue* ort_keys = in[0];
  auto p_key_ml_value = reinterpret_cast<const MLValue*>(ort_keys);
  auto& key_tensor = p_key_ml_value->Get<Tensor>();
  auto key_type = key_tensor.DataType();

  const OrtValue* ort_values = in[1];
  auto p_value_ml_value = reinterpret_cast<const MLValue*>(ort_values);
  auto& value_tensor = p_value_ml_value->Get<Tensor>();

  // TODO validate that key and value tensors are rank-1 tensors only
  // TODO validate that the key and value tensors have the same number of elements
  if (key_type == DataTypeImpl::GetType<std::string>()) {
    return OrtCreateValueImplMapHelper<std::string>(key_tensor, value_tensor, info, out);
  } else if (key_type == DataTypeImpl::GetType<int64_t>()) {
    return OrtCreateValueImplMapHelper<int64_t>(key_tensor, value_tensor, info, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Key type is not supported yet.");
  }
}

static OrtStatus* OrtCreateValueImpl(const OrtValue** in, int num_values, enum ONNXType value_type, const OrtAllocatorInfo* info, OrtValue** out) {
  if (num_values <= 0) {
    return OrtCreateStatus(ORT_FAIL, "Number of values should be at least 1.");
  }
  auto value_type = OrtGetValueType(value);
  if (value_type == ONNX_TYPE_MAP) {
    return OrtCreateValueImplMap(value, num_values, info, out);
  } else if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtCreateValueImplSeq(value, num_values, info, out);
  } else {
    return OrtCreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtCreateValue, const OrtValue** in, int num_values, enum ONNXType value_type, const OrtAllocatorInfo* info, OrtValue** out) {
  API_IMPL_BEGIN
  return OrtCreateValueImpl(in, num_values, value_type, info, out);
  API_IMPL_END
}

// End support for non-tensor types

DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Env, OrtEnv)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Value, MLValue)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(RunOptions, OrtRunOptions)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Session, ::onnxruntime::InferenceSession)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION_FOR_ARRAY(Status, char)
