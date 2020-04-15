#include <core/session/onnxruntime_cxx_api.h>

#include "inference_session_wrap.h"
#include "napi_utils.h"
#include "session_options_helper.h"
#include "tensor_helper.h"

Napi::FunctionReference InferenceSessionWrap::constructor;
Ort::Env *InferenceSessionWrap::ortEnv_;


Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
  // initialize ORT
  ortEnv_ = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "onnxruntime-node"};

  // initialize binding
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "InferenceSession",
      {InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel), InstanceMethod("run", &InferenceSessionWrap::Run),
       InstanceAccessor("inputNames", &InferenceSessionWrap::GetInputNames, nullptr, napi_default, nullptr),
       InstanceAccessor("outputNames", &InferenceSessionWrap::GetOutputNames, nullptr, napi_default, nullptr)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("InferenceSession", func);
  return exports;
}

InferenceSessionWrap::InferenceSessionWrap(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<InferenceSessionWrap>(info), initialized_(false), session_(nullptr),
      defaultRunOptions_(new Ort::RunOptions{}) {
  // parse session options
  if (info.Length() >= 1 && info[0].IsObject()) {
    sessionOptions_.reset(new Ort::SessionOptions());
    ParseSessionOptions(info[0].As<Napi::Object>(), *sessionOptions_.get());
  }
}

Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (this->initialized_) {
    throw Napi::Error::New(env, "Model already loaded. Cannot load model multiple times.");
  }

  size_t length = info.Length();
  if (length == 0) {
    throw Napi::TypeError::New(env, "Expect argument: model file path or buffer");
  }

  if (length == 1 && info[0].IsString()) {
    Napi::String value = info[0].As<Napi::String>();

    try {
      std::unique_ptr<Ort::SessionOptions> sessionOptions(std::move(sessionOptions_));
      if (sessionOptions) {
        this->session_.reset(new Ort::Session(OrtEnv(),
#ifdef _WIN32
                                              reinterpret_cast<const wchar_t *>(value.Utf16Value().c_str()),
#else
                                              value.Utf8Value().c_str(),
#endif
                                              *sessionOptions.get()));
      } else {
        this->session_.reset(new Ort::Session(OrtEnv(),
#ifdef _WIN32
                                              reinterpret_cast<const wchar_t *>(value.Utf16Value().c_str()),
#else
                                              value.Utf8Value().c_str(),
#endif
                                              Ort::SessionOptions{}));
      }
    } catch (std::exception const &e) {
      throw Napi::Error::New(env, e.what());
    }
  } else if (length == 3 && info[0].IsArrayBuffer() && info[1].IsNumber() && info[2].IsNumber()) {
    void *buffer = info[0].As<Napi::ArrayBuffer>().Data();
    int64_t offset = info[1].As<Napi::Number>().Int64Value();
    int64_t length = info[2].As<Napi::Number>().Int64Value();

    try {
      std::unique_ptr<Ort::SessionOptions> sessionOptions(std::move(sessionOptions_));
      if (sessionOptions) {
        this->session_.reset(
            new Ort::Session(OrtEnv(), reinterpret_cast<char *>(buffer) + offset, length, *sessionOptions.get()));
      } else {
        this->session_.reset(
            new Ort::Session(OrtEnv(), reinterpret_cast<char *>(buffer) + offset, length, Ort::SessionOptions{}));
      }
    } catch (std::exception const &e) {
      throw Napi::Error::New(env, e.what());
    }
  } else {
    throw Napi::TypeError::New(env,
                               "Invalid argument: has to be either model file path (string) or buffer (Uint8Array)");
  }

  // cache input/output names and types
  Ort::AllocatorWithDefaultOptions allocator;

  size_t count = session_->GetInputCount();
  inputNames_.reserve(count);
  for (size_t i = 0; i < count; i++) {
    inputNames_.emplace_back(session_->GetInputName(i, allocator));
    auto typeInfo = session_->GetInputTypeInfo(i);
    auto onnxType = typeInfo.GetONNXType();
    inputTypes_.emplace_back(onnxType);
    inputTensorElementDataTypes_.emplace_back(onnxType == ONNX_TYPE_TENSOR
                                                  ? typeInfo.GetTensorTypeAndShapeInfo().GetElementType()
                                                  : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
  }

  count = session_->GetOutputCount();
  outputNames_.reserve(count);
  for (size_t i = 0; i < count; i++) {
    outputNames_.emplace_back(session_->GetOutputName(i, allocator));
    auto typeInfo = session_->GetOutputTypeInfo(i);
    auto onnxType = typeInfo.GetONNXType();
    outputTypes_.emplace_back(onnxType);
    outputTensorElementDataTypes_.emplace_back(onnxType == ONNX_TYPE_TENSOR
                                                   ? typeInfo.GetTensorTypeAndShapeInfo().GetElementType()
                                                   : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
  }

  this->initialized_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::GetInputNames(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }

  return scope.Escape(CreateNapiArrayFrom(env, inputNames_));
}

Napi::Value InferenceSessionWrap::GetOutputNames(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }

  return scope.Escape(CreateNapiArrayFrom(env, outputNames_));
}

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);

  if (!this->initialized_) {
    throw Napi::Error::New(env, "Session not initialized.");
  }
  if (info.Length() < 2) {
    throw Napi::TypeError::New(env, "Expect argument: inputs(feed) and outputs(fetch)");
  }
  if (!info[0].IsObject() || !info[1].IsObject()) {
    throw Napi::TypeError::New(env, "Expect inputs(feed) and outputs(fetch) to be objects");
  }
  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();

  std::vector<const char *> inputNames_cstr;
  std::vector<Ort::Value> inputValues;
  std::vector<const char *> outputNames_cstr;
  std::vector<Ort::Value> outputValues;
  std::vector<bool> reuseOutput;

  size_t inputIndex = 0;
  for (auto &name : inputNames_) {
    if (feed.Has(name)) {
      inputIndex++;
      inputNames_cstr.push_back(name.c_str());
      auto value = feed.Get(name);
      inputValues.push_back(NapiValueToOrtValue(env, value));
    }
  }
  size_t outputIndex = 0;
  for (auto &name : outputNames_) {
    if (fetch.Has(name)) {
      outputIndex++;
      outputNames_cstr.push_back(name.c_str());
      auto value = fetch.Get(name);
      reuseOutput.push_back(!value.IsNull());
      outputValues.emplace_back(value.IsNull() ? Ort::Value{nullptr} : NapiValueToOrtValue(env, value));
    }
  }

  try {
    session_->Run(*defaultRunOptions_.get(), &inputNames_cstr[0], &inputValues[0], inputValues.size(),
                  &outputNames_cstr[0], &outputValues[0], outputValues.size());
  } catch (std::exception const &e) {
    throw Napi::Error::New(env, e.what());
  }

  Napi::Object result = Napi::Object::New(env);

  for (size_t i = 0; i < outputIndex; i++) {
    result.Set(outputNames_[i], OrtValueToNapiValue(env, outputValues[i]));
  }

  return scope.Escape(result);
}
