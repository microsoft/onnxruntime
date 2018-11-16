// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"  //TODO: remove this
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
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
#include "core/framework/environment.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/inference_session.h"
#include "core/graph/graph_base.h"
#include "abi_session_options_impl.h"

using namespace onnxruntime::logging;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::MLStatus;
using onnxruntime::MLValue;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToONNXStatus;
using onnxruntime::common::Status;

#define ONNXRUNTIME_API_RETURN_IF_ERROR(expr) \
  do {                                        \
    auto _status = (expr);                    \
    if (_status) return _status;              \
  } while (0)

struct ONNXEnv {
  ONNXEnv(Environment* value1, LoggingManager* loggingManager1) : value(value1), loggingManager(loggingManager1) {
  }
  /**
  * This function will call ::google::protobuf::ShutdownProtobufLibrary
  */
  ~ONNXEnv() {
    delete loggingManager;
    delete value;
  }
  Environment* value;
  LoggingManager* loggingManager;
  ONNXRUNTIME_DISALLOW_COPY_AND_ASSIGNMENT(ONNXEnv);
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                   \
  }                                                                    \
  catch (std::exception & ex) {                                        \
    return CreateONNXStatus(ONNXRUNTIME_RUNTIME_EXCEPTION, ex.what()); \
  }

#define TENSOR_READ_API_BEGIN                                \
  API_IMPL_BEGIN                                             \
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN                           \
  API_IMPL_BEGIN                                             \
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value); \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

class LoggingWrapper : public ISink {
 public:
  LoggingWrapper(ONNXRuntimeLoggingFunction logging_function, void* logger_param)
      : logging_function_{logging_function}, logger_param_{logger_param} {
  }

  void SendImpl(const Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                const Capture& message) override {
    std::string s = message.Location().ToString();
    logging_function_(logger_param_, static_cast<ONNXRuntimeLoggingLevel>(message.Severity()), message.Category(),
                      logger_id.c_str(), s.c_str(), message.Message().c_str());
  }

 private:
  ONNXRuntimeLoggingFunction logging_function_;
  void* logger_param_;
};

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInitializeWithCustomLogger, ONNXRuntimeLoggingFunction logging_function,
                            void* logger_param, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid,
                            _Out_ ONNXEnv** out) {
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
    *out = new ONNXEnv(env.release(), default_logging_manager.release());
  return ToONNXStatus(status);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInitialize, ONNXRuntimeLoggingLevel default_warning_level,
                            _In_ const char* logid, _Out_ ONNXEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  auto default_logging_manager = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                                  static_cast<Severity>(default_warning_level), false,
                                                                  LoggingManager::InstanceType::Default,
                                                                  &name);
  std::unique_ptr<Environment> env;
  Status status = Environment::Create(env);
  if (status.IsOK())
    *out = new ONNXEnv(env.release(), default_logging_manager.release());
  return ToONNXStatus(status);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetStringTensorDataLength, _In_ ONNXValuePtr value, _Out_ size_t* out) {
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
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeFillStringTensor, _In_ ONNXValuePtr value, _In_ const char* s[], size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len < len) {
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i != len; ++i) {
    //allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

template <typename T>
void CreateTensorImpl(const size_t* shape, size_t shape_len, std::shared_ptr<onnxruntime::IAllocator>& allocator,
                      std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate = sizeof(T) * elem_count;
  void* p_data = allocator->Alloc(size_to_allocate);
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes.data(), shape_len),
                                  static_cast<void*>(p_data),
                                  allocator->Info(),
                                  allocator);
}

template <typename T>
ONNXStatusPtr CreateTensorImpl(const size_t* shape, size_t shape_len, const ONNXRuntimeAllocatorInfo* info,
                               void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate = sizeof(T) * elem_count;
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes.data(), shape_len),
                                  p_data,
                                  *info,
                                  nullptr);
  return nullptr;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateTensorWithDataAsONNXValue, _In_ const ONNXRuntimeAllocatorInfo* info,
                            _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len,
                            OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<float>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<std::string>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<bool>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<MLFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<double>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return CreateONNXStatus(ONNXRUNTIME_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateTensorAsONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                            _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type,
                            _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::shared_ptr<onnxruntime::IAllocator> allocator_ = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      CreateTensorImpl<float>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      CreateTensorImpl<uint8_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      CreateTensorImpl<int8_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      CreateTensorImpl<uint16_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      CreateTensorImpl<int16_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      CreateTensorImpl<int32_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      CreateTensorImpl<int64_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      CreateTensorImpl<std::string>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      CreateTensorImpl<bool>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      CreateTensorImpl<MLFloat16>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      CreateTensorImpl<double>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      CreateTensorImpl<uint32_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      CreateTensorImpl<uint64_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return CreateONNXStatus(ONNXRUNTIME_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

template <typename T>
static ONNXStatusPtr CreateInferenceSessionImpl(_In_ ONNXEnv* env, _In_ T model_path,
                                                _In_ const ONNXRuntimeSessionOptions* options,
                                                _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  auto sess = std::make_unique<::onnxruntime::InferenceSession>(options->value, env->loggingManager);
  Status status;
  if (!options->custom_op_paths.empty()) {
    status = sess->LoadCustomOps(options->custom_op_paths);
    if (!status.IsOK())
      return ToONNXStatus(status);
  }
  for (ONNXRuntimeProviderFactoryPtr* p : options->provider_factories) {
    ONNXRuntimeProviderPtr provider;
    ONNXStatusPtr error_code = (*p)->CreateProvider(p, &provider);
    if (error_code)
      return error_code;
    sess->RegisterExecutionProvider(std::unique_ptr<onnxruntime::IExecutionProvider>(
        reinterpret_cast<onnxruntime::IExecutionProvider*>(provider)));
  }
  status = sess->Load(model_path);
  if (!status.IsOK())
    return ToONNXStatus(status);
  status = sess->Initialize();
  if (!status.IsOK())
    return ToONNXStatus(status);
  *out = reinterpret_cast<ONNXSessionPtr>(sess.release());
  return nullptr;
  API_IMPL_END
}

#ifdef _WIN32
ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const wchar_t* model_path,
                            _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  return CreateInferenceSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#else
ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const char* model_path,
                            _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  return CreateInferenceSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#endif

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunInference, _In_ ONNXSessionPtr sess,
                            _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                            _In_ const char* output_names1[], size_t output_names_len, _Out_ ONNXValuePtr* output) {
  API_IMPL_BEGIN
  ONNXRuntimeRunOptions run_options{};
  return ONNXRuntimeRunInferenceWithRunOptions(sess, &run_options, input_names, input, input_len,
                                               output_names1, output_names_len, output);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunInferenceWithRunOptions, _In_ ONNXSessionPtr sess,
                            _In_ ONNXRuntimeRunOptionsPtr run_options,
                            _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                            _In_ const char* output_names1[], size_t output_names_len, _Out_ ONNXValuePtr* output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  ::onnxruntime::NameMLValMap in;
  const int queue_id = 0;
  for (size_t i = 0; i != input_len; ++i) {
    auto kvp = in.insert(std::make_pair(std::string(input_names[i]),
                                        *reinterpret_cast<::onnxruntime::MLValue*>(input[i])));
    if (!kvp.second) {
      return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "duplicated input name");
    }
    ::onnxruntime::MLValue& value = kvp.first->second;
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
  }
  // Create output feed
  std::vector<std::string> output_names(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output_names1[i] == nullptr || output_names1[i][0] == '\0') {
      return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "output name cannot be empty");
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
  auto status = session->Run(*run_options, in, output_names, &fetches);
  if (!status.IsOK())
    return ToONNXStatus(status);
  for (size_t i = 0; i != output_names_len; ++i) {
    ::onnxruntime::MLValue& value = fetches[i];
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (output[i] == nullptr) {
      output[i] = reinterpret_cast<ONNXValuePtr>(new MLValue(value));
    }
  }
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunInferenceAndFetchAll, _In_ ONNXSessionPtr sess,
                            _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                            _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len) {
  API_IMPL_BEGIN
  ONNXRuntimeRunOptions run_options{};
  return ONNXRuntimeRunInferenceAndFetchAllWithRunOptions(sess, &run_options, input_names, input, input_len,
                                                          output, output_len);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunInferenceAndFetchAllWithRunOptions, _In_ ONNXSessionPtr sess,
                            _In_ ONNXRuntimeRunOptionsPtr run_options,
                            _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len,
                            _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  ::onnxruntime::NameMLValMap in;
  for (size_t i = 0; i != input_len; ++i) {
    auto kvp = in.insert(std::make_pair(std::string(input_names[i]),
                                        *reinterpret_cast<::onnxruntime::MLValue*>(input[i])));
    if (!kvp.second) {
      return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "duplicated input name");
    }
  }
  // Create output feed
  std::vector<std::string> output_names;
  for (auto const& outp : *(session->GetModelOutputs().second)) {
    output_names.push_back(outp->Name());
  }
  std::vector<MLValue> fetches;
  auto status = session->Run(*run_options, in, output_names, &fetches);
  if (!status.IsOK())
    return ToONNXStatus(status);
  auto* out = new MLValue[fetches.size()];
  const int queue_id = 0;
  for (size_t i = 0; i != fetches.size(); ++i) {
    if (fetches[i].Fence())
      fetches[i].Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    out[i] = fetches[i];
  }
  *output_len = fetches.size();
  *output = reinterpret_cast<ONNXValueListPtr>(out);
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorMutableData, _In_ ONNXValuePtr value, _Out_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  //TODO: test if it's a string tensor
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

inline OnnxRuntimeTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(
    const onnxruntime::DataTypeImpl* cpp_type) {
  OnnxRuntimeTensorElementDataType type;
  if (cpp_type == onnxruntime::DataTypeImpl::GetType<float>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint8_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int8_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint16_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int16_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int32_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int64_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<std::string>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<bool>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<MLFloat16>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<double>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint32_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint64_t>()) {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  } else {
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_MAX;
  }
  return type;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorShapeAndType, _In_ const ONNXValuePtr value,
                            _Out_ ONNXRuntimeTensorTypeAndShapeInfo** out) {
  TENSOR_READ_API_BEGIN
  OnnxRuntimeTensorElementDataType type = MLDataTypeToOnnxRuntimeTensorElementDataType(tensor.DataType());
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_MAX == type) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "Not implemented");
  }
  const onnxruntime::TensorShape& shape = tensor.Shape();
  ONNXRuntimeTensorTypeAndShapeInfo* ret = ONNXRuntimeCreateTensorTypeAndShapeInfo();
  auto status = ONNXRuntimeSetTensorElementType(ret, type);
  if (status != nullptr) {
    ONNXRuntimeReleaseObject(ret);
    return status;
  }
  status = ONNXRuntimeSetDims(ret, shape.GetDims().data(), shape.GetDims().size());
  if (status != nullptr) {
    ONNXRuntimeReleaseObject(ret);
    return status;
  }
  *out = ret;
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetStringTensorContent, _In_ ONNXValuePtr value,
                            _Out_ void* s, size_t s_len, _Out_ size_t* offsets, size_t offsets_len) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (offsets_len < len) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "space is not enough");
  }
  {
    size_t ret = 0;
    for (size_t i = 0; i != len; ++i) {
      ret += input[i].size();
    }
    if (s_len < ret) {
      return CreateONNXStatus(ONNXRUNTIME_FAIL, "space is not enough");
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

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeTensorProtoToONNXValue, _Inout_ ONNXRuntimeAllocator* allocator,
                            const void* input, int input_len, _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::shared_ptr<onnxruntime::IAllocator> allocator_ = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  ::ONNX_NAMESPACE::TensorProto proto;
  if (!proto.ParseFromArray(input, input_len)) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "parse input tensor proto failed");
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  Status st = onnxruntime::utils::TensorProtoToMLValue(proto, allocator_, nullptr, 0, *value);
  if (!st.IsOK())
    return ToONNXStatus(st);
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API(ONNXValuePtr, ONNXRuntimeONNXValueListGetNthValue, ONNXValueListPtr list, size_t index) {
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(list);
  return reinterpret_cast<ONNXValuePtr>(v + index);
}

#define DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(INPUT_TYPE, REAL_TYPE) \
  ONNXRUNTIME_API(void, Release##INPUT_TYPE, INPUT_TYPE##Ptr value) {      \
    delete reinterpret_cast<REAL_TYPE*>(value);                            \
  }

#define DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(INPUT_TYPE, REAL_TYPE) \
  ONNXRUNTIME_API(void, Release##INPUT_TYPE, INPUT_TYPE##Ptr value) {                \
    delete[] reinterpret_cast<REAL_TYPE*>(value);                                    \
  }

ONNXRUNTIME_API(void, ReleaseONNXEnv, ONNXEnv* env) {
  delete env;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInferenceSessionGetInputCount, _In_ ONNXSessionPtr sess, _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelInputs();
  if (!p.first.IsOK())
    return ToONNXStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInferenceSessionGetOutputCount, _In_ ONNXSessionPtr sess, _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = session->GetModelOutputs();
  if (!p.first.IsOK())
    return ToONNXStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

static char* StrDup(const std::string& str, ONNXRuntimeAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>((*allocator)->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static ONNXStatusPtr GetInputOutputNameImpl(_In_ ONNXSessionPtr sess, size_t index,
                                            _Inout_ ONNXRuntimeAllocator* allocator, bool is_input,
                                            _Out_ char** output) {
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = is_input ? session->GetModelInputs() : session->GetModelOutputs();
  if (!p.first.IsOK())
    return ToONNXStatus(p.first);
  if (p.second == nullptr)
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "internal error");
  const InputDefList& defs = *p.second;
  if (index >= defs.size())
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "index out of range");
  *output = StrDup(defs[index]->Name(), allocator);
  return nullptr;
}

ONNXRUNTIME_API(int, ONNXRuntimeIsTensor, _In_ ONNXValuePtr value) {
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value);
  return v->IsTensor() ? 1 : 0;
}

ONNXRUNTIME_API(void*, ONNXRuntimeAllocatorAlloc, _Inout_ ONNXRuntimeAllocator* ptr, size_t size) {
  try {
    return (*ptr)->Alloc(ptr, size);
  } catch (std::exception&) {
    return nullptr;
  }
}

ONNXRUNTIME_API(void, ONNXRuntimeAllocatorFree, _Inout_ ONNXRuntimeAllocator* ptr, void* p) {
  try {
    (*ptr)->Free(ptr, p);
  } catch (std::exception&) {
  }
}

ONNXRUNTIME_API(const struct ONNXRuntimeAllocatorInfo*, ONNXRuntimeAllocatorGetInfo, _In_ const ONNXRuntimeAllocator* ptr) {
  try {
    return (*ptr)->Info(ptr);
  } catch (std::exception&) {
    return nullptr;
  }
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInferenceSessionGetInputName, _In_ ONNXSessionPtr sess, size_t index,
                            _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** output) {
  API_IMPL_BEGIN
  return GetInputOutputNameImpl(sess, index, allocator, true, output);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeInferenceSessionGetOutputName, _In_ ONNXSessionPtr sess, size_t index,
                            _Inout_ ONNXRuntimeAllocator* allocator, _Out_ char** output) {
  API_IMPL_BEGIN
  return GetInputOutputNameImpl(sess, index, allocator, false, output);
  API_IMPL_END
}

DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(ONNXValue, MLValue)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(ONNXSession, ::onnxruntime::InferenceSession)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(ONNXValueList, ::onnxruntime::MLValue)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(ONNXStatus, char)
