// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "directml_load_helper.h"
#include "inference_session_wrap.h"
#include "run_options_helper.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#include <string>

Napi::FunctionReference InferenceSessionWrap::constructor;

Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
#if defined(USE_DML) && defined(_WIN32)
  LoadDirectMLDll(env);
#endif
  // create ONNX runtime env
  Ort::InitApi();
  ORT_NAPI_THROW_ERROR_IF(
      Ort::Global<void>::api_ == nullptr, env,
      "Failed to initialize ONNX Runtime API. It could happen when this nodejs binding was built with a higher version "
      "ONNX Runtime but now runs with a lower version ONNX Runtime DLL(or shared library).");
  auto ortEnv = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "onnxruntime-node"};
  env.SetInstanceData(ortEnv);
  // initialize binding
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "InferenceSession",
      {InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel), InstanceMethod("run", &InferenceSessionWrap::Run),
       InstanceMethod("dispose", &InferenceSessionWrap::Dispose),
       InstanceAccessor("inputNames", &InferenceSessionWrap::GetInputNames, nullptr, napi_default, nullptr),
       InstanceAccessor("outputNames", &InferenceSessionWrap::GetOutputNames, nullptr, napi_default, nullptr)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("InferenceSession", func);

  Napi::Function listSupportedBackends = Napi::Function::New(env, InferenceSessionWrap::ListSupportedBackends);
  exports.Set("listSupportedBackends", listSupportedBackends);

  return exports;
}

InferenceSessionWrap::InferenceSessionWrap(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<InferenceSessionWrap>(info), initialized_(false), disposed_(false), session_(nullptr),
      defaultRunOptions_(nullptr) {}

Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  ORT_NAPI_THROW_ERROR_IF(this->initialized_, env, "Model already loaded. Cannot load model multiple times.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  size_t argsLength = info.Length();
  ORT_NAPI_THROW_TYPEERROR_IF(argsLength == 0, env, "Expect argument: model file path or buffer.");

  try {
    defaultRunOptions_.reset(new Ort::RunOptions{});
    Ort::SessionOptions sessionOptions;

    if (argsLength == 2 && info[0].IsString() && info[1].IsObject()) {
      Napi::String value = info[0].As<Napi::String>();

      ParseSessionOptions(info[1].As<Napi::Object>(), sessionOptions);
      this->session_.reset(new Ort::Session(*env.GetInstanceData<Ort::Env>(),
#ifdef _WIN32
                                            reinterpret_cast<const wchar_t *>(value.Utf16Value().c_str()),
#else
                                            value.Utf8Value().c_str(),
#endif
                                            sessionOptions));

    } else if (argsLength == 4 && info[0].IsArrayBuffer() && info[1].IsNumber() && info[2].IsNumber() &&
               info[3].IsObject()) {
      void *buffer = info[0].As<Napi::ArrayBuffer>().Data();
      int64_t bytesOffset = info[1].As<Napi::Number>().Int64Value();
      int64_t bytesLength = info[2].As<Napi::Number>().Int64Value();

      ParseSessionOptions(info[3].As<Napi::Object>(), sessionOptions);
      this->session_.reset(new Ort::Session(*env.GetInstanceData<Ort::Env>(),
                                            reinterpret_cast<char *>(buffer) + bytesOffset, bytesLength,
                                            sessionOptions));
    } else {
      ORT_NAPI_THROW_TYPEERROR(
          env,
          "Invalid argument: args has to be either (modelPath, options) or (buffer, byteOffset, byteLength, options).");
    }

    // cache input/output names and types
    Ort::AllocatorWithDefaultOptions allocator;

    size_t count = session_->GetInputCount();
    inputNames_.reserve(count);
    for (size_t i = 0; i < count; i++) {
      auto inp_name = session_->GetInputNameAllocated(i, allocator);
      inputNames_.emplace_back(inp_name.get());
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
      auto out_name = session_->GetOutputNameAllocated(i, allocator);
      outputNames_.emplace_back(out_name.get());
      auto typeInfo = session_->GetOutputTypeInfo(i);
      auto onnxType = typeInfo.GetONNXType();
      outputTypes_.emplace_back(onnxType);
      outputTensorElementDataTypes_.emplace_back(onnxType == ONNX_TYPE_TENSOR
                                                     ? typeInfo.GetTensorTypeAndShapeInfo().GetElementType()
                                                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    }
  } catch (Napi::Error const &e) {
    throw e;
  } catch (std::exception const &e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
  this->initialized_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::GetInputNames(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);
  return scope.Escape(CreateNapiArrayFrom(env, inputNames_));
}

Napi::Value InferenceSessionWrap::GetOutputNames(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);
  return scope.Escape(CreateNapiArrayFrom(env, outputNames_));
}

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() < 2, env, "Expect argument: inputs(feed) and outputs(fetch).");
  ORT_NAPI_THROW_TYPEERROR_IF(!info[0].IsObject() || !info[1].IsObject(), env,
                              "Expect inputs(feed) and outputs(fetch) to be objects.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() > 2 && (!info[2].IsObject() || info[2].IsNull()), env,
                              "'runOptions' must be an object.");

  Napi::EscapableHandleScope scope(env);

  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();

  std::vector<const char *> inputNames_cstr;
  std::vector<Ort::Value> inputValues;
  std::vector<const char *> outputNames_cstr;
  std::vector<Ort::Value> outputValues;
  std::vector<bool> reuseOutput;
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  OrtMemoryInfo *memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault).release();

  try {
    for (auto &name : inputNames_) {
      if (feed.Has(name)) {
        inputIndex++;
        inputNames_cstr.push_back(name.c_str());
        auto value = feed.Get(name);
        inputValues.push_back(NapiValueToOrtValue(env, value, memory_info));
      }
    }
    for (auto &name : outputNames_) {
      if (fetch.Has(name)) {
        outputIndex++;
        outputNames_cstr.push_back(name.c_str());
        auto value = fetch.Get(name);
        reuseOutput.push_back(!value.IsNull());
        outputValues.emplace_back(value.IsNull() ? Ort::Value{nullptr} : NapiValueToOrtValue(env, value, memory_info));
      }
    }

    Ort::RunOptions runOptions{nullptr};
    if (info.Length() > 2) {
      runOptions = Ort::RunOptions{};
      ParseRunOptions(info[2].As<Napi::Object>(), runOptions);
    }

    session_->Run(runOptions == nullptr ? *defaultRunOptions_.get() : runOptions,
                  inputIndex == 0 ? nullptr : &inputNames_cstr[0], inputIndex == 0 ? nullptr : &inputValues[0],
                  inputIndex, outputIndex == 0 ? nullptr : &outputNames_cstr[0],
                  outputIndex == 0 ? nullptr : &outputValues[0], outputIndex);

    Napi::Object result = Napi::Object::New(env);

    for (size_t i = 0; i < outputIndex; i++) {
      result.Set(outputNames_[i], OrtValueToNapiValue(env, outputValues[i]));
    }

    return scope.Escape(result);
  } catch (Napi::Error const &e) {
    throw e;
  } catch (std::exception const &e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  this->defaultRunOptions_.reset(nullptr);
  this->session_.reset(nullptr);

  this->disposed_ = true;
}

Napi::Value InferenceSessionWrap::ListSupportedBackends(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  Napi::Array result = Napi::Array::New(env);

  auto createObject = [&env](const std::string &name, const bool bundled) -> Napi::Object {
    Napi::Object result = Napi::Object::New(env);
    result.Set("name", name);
    result.Set("bundled", bundled);
    return result;
  };

  result.Set(uint32_t(0), createObject("cpu", true));

#ifdef USE_DML
  result.Set(result.Length(), createObject("dml", true));
#endif
#ifdef USE_CUDA
  result.Set(result.Length(), createObject("cuda", false));
#endif
#ifdef USE_TENSORRT
  result.Set(result.Length(), createObject("tensorrt", false));
#endif
#ifdef USE_COREML
  result.Set(result.Length(), createObject("coreml", true));
#endif

  return scope.Escape(result);
}
