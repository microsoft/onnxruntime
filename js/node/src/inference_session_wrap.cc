// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "inference_session_wrap.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "run_options_helper.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#include <optional>
#include <string>

namespace {

struct RunAsyncContext {
  napi_async_work work;
  napi_env env;       // raw handle — Napi::Env has no default constructor
  napi_deferred deferred;  // raw handle — Napi::Promise::Deferred has no default constructor
  Napi::ObjectReference sessionRef;
  std::vector<Napi::Reference<Napi::Value>> inputValueRefs;

  Ort::Session* session;
  Ort::IoBinding* ioBinding;
  bool useIoBinding;
  std::vector<int> preferredOutputLocations;

  std::vector<const char*> inputNames_cstr;
  std::vector<Ort::Value> inputValues;
  std::vector<const char*> outputNames_cstr;
  std::vector<Ort::Value> outputValues;
  size_t inputIndex;
  size_t outputIndex;

  std::optional<Ort::RunOptions> runOptions;  // empty = use singleton default
  Ort::MemoryInfo cpuMemoryInfo{nullptr};       // overwritten in Run() before dispatch
  Ort::MemoryInfo gpuBufferMemoryInfo{nullptr}; // overwritten in Run() before dispatch

  std::vector<Ort::Value> asyncOutputs;
  std::string errorMessage;
  bool hasError;
};

void RunWorkCallback(napi_env /*env*/, void* data) {
  auto* ctx = static_cast<RunAsyncContext*>(data);
  try {
    auto* ortObjects = OrtSingletonData::GetOrtObjects();
    if (!ortObjects) {
      ctx->hasError = true;
      ctx->errorMessage = "ORT runtime has been destroyed.";
      return;
    }
    Ort::RunOptions& opts = ctx->runOptions.has_value() ? *ctx->runOptions : ortObjects->default_run_options;
    if (!ctx->useIoBinding) {
      ctx->session->Run(opts,
                        ctx->inputIndex == 0 ? nullptr : ctx->inputNames_cstr.data(),
                        ctx->inputIndex == 0 ? nullptr : ctx->inputValues.data(),
                        ctx->inputIndex,
                        ctx->outputIndex == 0 ? nullptr : ctx->outputNames_cstr.data(),
                        ctx->outputIndex == 0 ? nullptr : ctx->outputValues.data(),
                        ctx->outputIndex);
    } else {
      ctx->session->Run(opts, *ctx->ioBinding);
      ctx->asyncOutputs = ctx->ioBinding->GetOutputValues();
    }
  } catch (std::exception const& e) {
    ctx->hasError = true;
    ctx->errorMessage = e.what();
  }
}

void RunAfterWorkCallback(napi_env /*env*/, napi_status status, void* data) {
  auto* ctx = static_cast<RunAsyncContext*>(data);
  Napi::Env env(ctx->env);
  Napi::HandleScope scope(env);

  if (ctx->hasError) {
    napi_reject_deferred(ctx->env, ctx->deferred, Napi::Error::New(env, ctx->errorMessage).Value());
  } else if (status == napi_cancelled) {
    napi_reject_deferred(ctx->env, ctx->deferred, Napi::Error::New(env, "Async inference was cancelled.").Value());
  } else {
    try {
      Napi::Object result = Napi::Object::New(env);
      if (!ctx->useIoBinding) {
        for (size_t i = 0; i < ctx->outputIndex; i++) {
          result.Set(ctx->outputNames_cstr[i], OrtValueToNapiValue(env, std::move(ctx->outputValues[i])));
        }
      } else {
        for (size_t i = 0; i < ctx->outputIndex; i++) {
          result.Set(ctx->outputNames_cstr[i], OrtValueToNapiValue(env, std::move(ctx->asyncOutputs[i])));
        }
      }
      napi_resolve_deferred(ctx->env, ctx->deferred, result);
    } catch (std::exception const& e) {
      napi_reject_deferred(ctx->env, ctx->deferred, Napi::Error::New(env, e.what()).Value());
    }
  }

  ctx->sessionRef.Reset();
  for (auto& ref : ctx->inputValueRefs) {
    ref.Reset();
  }
  napi_delete_async_work(ctx->env, ctx->work);
  delete ctx;
}

}  // namespace

Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
  // create ONNX runtime env
  Ort::InitApi();

  // initialize binding
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "InferenceSession",
      {InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel),
       InstanceMethod("run", &InferenceSessionWrap::Run),
       InstanceMethod("runSync", &InferenceSessionWrap::RunSync),
       InstanceMethod("dispose", &InferenceSessionWrap::Dispose),
       InstanceMethod("endProfiling", &InferenceSessionWrap::EndProfiling),
       InstanceAccessor("inputMetadata", &InferenceSessionWrap::GetMetadata, nullptr, napi_default, reinterpret_cast<void*>(true)),
       InstanceAccessor("outputMetadata", &InferenceSessionWrap::GetMetadata, nullptr, napi_default, reinterpret_cast<void*>(false))});

  OrtInstanceData::Create(env, func);

  exports.Set("InferenceSession", func);

  Napi::Function listSupportedBackends = Napi::Function::New(env, InferenceSessionWrap::ListSupportedBackends);
  exports.Set("listSupportedBackends", listSupportedBackends);

  Napi::Function initOrtOnce = Napi::Function::New(env, InferenceSessionWrap::InitOrtOnce);
  exports.Set("initOrtOnce", initOrtOnce);

  return exports;
}

Napi::Value InferenceSessionWrap::InitOrtOnce(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  int log_level = info[0].As<Napi::Number>().Int32Value();
  Napi::Function tensorConstructor = info[1].As<Napi::Function>();
  bool is_main_thread = info[2].As<Napi::Boolean>().Value();

  OrtInstanceData::InitOrt(env, log_level, tensorConstructor, is_main_thread);

  return env.Undefined();
}

InferenceSessionWrap::InferenceSessionWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InferenceSessionWrap>(info), initialized_(false), disposed_(false), session_(nullptr) {}

InferenceSessionWrap::~InferenceSessionWrap() {
  // If the ORT singleton has already been destroyed (e.g. during process shutdown when the
  // cleanup hook fires before N-API finalizers run), we must not call into ORT to
  // release owned ORT objects — doing so would crash. Intentionally leak in that case.
  if (!OrtSingletonData::GetOrtObjects()) {
    for (auto& type_info : inputTypes_) {
      (void)type_info.release();
    }
    for (auto& type_info : outputTypes_) {
      (void)type_info.release();
    }
    (void)ioBinding_.release();
    (void)session_.release();
  }
}

Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  ORT_NAPI_THROW_ERROR_IF(this->initialized_, env, "Model already loaded. Cannot load model multiple times.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  size_t argsLength = info.Length();
  ORT_NAPI_THROW_TYPEERROR_IF(argsLength == 0, env, "Expect argument: model file path or buffer.");

  try {
    Ort::SessionOptions sessionOptions;

    if (argsLength == 2 && info[0].IsString() && info[1].IsObject()) {
      Napi::String value = info[0].As<Napi::String>();

      ParseSessionOptions(info[1].As<Napi::Object>(), sessionOptions);
      this->session_.reset(new Ort::Session(OrtSingletonData::GetOrtObjects()->env,
#ifdef _WIN32
                                            reinterpret_cast<const wchar_t*>(value.Utf16Value().c_str()),
#else
                                            value.Utf8Value().c_str(),
#endif
                                            sessionOptions));

    } else if (argsLength == 4 && info[0].IsArrayBuffer() && info[1].IsNumber() && info[2].IsNumber() &&
               info[3].IsObject()) {
      void* buffer = info[0].As<Napi::ArrayBuffer>().Data();
      int64_t bytesOffset = info[1].As<Napi::Number>().Int64Value();
      int64_t bytesLength = info[2].As<Napi::Number>().Int64Value();

      ParseSessionOptions(info[3].As<Napi::Object>(), sessionOptions);
      this->session_.reset(new Ort::Session(OrtSingletonData::GetOrtObjects()->env,
                                            reinterpret_cast<char*>(buffer) + bytesOffset, bytesLength,
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
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      inputNames_.emplace_back(input_name.get());
      inputTypes_.push_back(session_->GetInputTypeInfo(i));
    }

    count = session_->GetOutputCount();
    outputNames_.reserve(count);
    for (size_t i = 0; i < count; i++) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      outputNames_.emplace_back(output_name.get());
      outputTypes_.push_back(session_->GetOutputTypeInfo(i));
    }

    // cache preferred output locations
    ParsePreferredOutputLocations(info[argsLength - 1].As<Napi::Object>(), outputNames_, preferredOutputLocations_);
    if (preferredOutputLocations_.size() > 0) {
      ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);
    }
  } catch (Napi::Error const& e) {
    throw e;
  } catch (std::exception const& e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
  this->initialized_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::GetMetadata(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);
  auto& names = info.Data() != nullptr ? inputNames_ : outputNames_;
  auto& types = info.Data() != nullptr ? inputTypes_ : outputTypes_;
  auto array = Napi::Array::New(env, types.size());
  for (uint32_t i = 0; i < types.size(); i++) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("name", names[i]);
    auto& typeInfo = types[i];
    if (typeInfo.GetONNXType() == ONNX_TYPE_TENSOR) {
      obj.Set("isTensor", true);

      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      obj.Set("type", static_cast<std::underlying_type_t<ONNXTensorElementDataType>>(tensorInfo.GetElementType()));
      obj.Set("symbolicDimensions", CreateNapiArrayFrom(env, tensorInfo.GetSymbolicDimensions()));
      obj.Set("shape", CreateNapiArrayFrom(env, tensorInfo.GetShape()));
    } else {
      obj.Set("isTensor", false);
    }
    array.Set(i, Napi::Value::From(env, obj));
  }
  return scope.Escape(array);
}

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() < 2, env, "Expect argument: inputs(feed) and outputs(fetch).");
  ORT_NAPI_THROW_TYPEERROR_IF(!info[0].IsObject() || !info[1].IsObject(), env,
                              "Expect inputs(feed) and outputs(fetch) to be objects.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() > 2 && (!info[2].IsObject() || info[2].IsNull()), env,
                              "'runOptions' must be an object.");

  napi_value promise_value;
  auto* ctx = new RunAsyncContext{};
  ctx->env = static_cast<napi_env>(env);
  napi_create_promise(env, &ctx->deferred, &promise_value);
  ctx->sessionRef = Napi::Persistent(info.This().As<Napi::Object>());
  ctx->session = session_.get();
  ctx->ioBinding = ioBinding_.get();
  ctx->useIoBinding = (preferredOutputLocations_.size() > 0);
  ctx->preferredOutputLocations = preferredOutputLocations_;
  ctx->inputIndex = 0;
  ctx->outputIndex = 0;
  ctx->cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  ctx->gpuBufferMemoryInfo = Ort::MemoryInfo{"WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault};
  ctx->hasError = false;

  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();

  try {
    for (auto& name : inputNames_) {
      if (feed.Has(name)) {
        ctx->inputIndex++;
        ctx->inputNames_cstr.push_back(name.c_str());
        auto value = feed.Get(name);
        // Keep a persistent reference so the JS ArrayBuffer backing CPU tensors
        // (which are zero-copy borrowed by NapiValueToOrtValue) stays alive
        // for the duration of the async thread-pool work.
        ctx->inputValueRefs.push_back(Napi::Persistent(value));
        ctx->inputValues.push_back(NapiValueToOrtValue(env, value, ctx->cpuMemoryInfo, ctx->gpuBufferMemoryInfo));
      }
    }
    for (auto& name : outputNames_) {
      if (fetch.Has(name)) {
        ctx->outputIndex++;
        ctx->outputNames_cstr.push_back(name.c_str());
        auto value = fetch.Get(name);
        ctx->outputValues.emplace_back(value.IsNull() ? Ort::Value{nullptr}
                                                      : NapiValueToOrtValue(env, value, ctx->cpuMemoryInfo, ctx->gpuBufferMemoryInfo));
      }
    }

    if (info.Length() > 2) {
      ctx->runOptions.emplace();
      ParseRunOptions(info[2].As<Napi::Object>(), *ctx->runOptions);
    }

    if (ctx->useIoBinding) {
      ORT_NAPI_THROW_ERROR_IF(preferredOutputLocations_.size() != outputNames_.size(), env,
                              "Preferred output locations must have the same size as output names.");
      for (size_t i = 0; i < ctx->inputIndex; i++) {
        ctx->ioBinding->BindInput(ctx->inputNames_cstr[i], ctx->inputValues[i]);
      }
      for (size_t i = 0; i < ctx->outputIndex; i++) {
        // TODO: support preallocated output tensor (ctx->outputValues[i])
        if (ctx->preferredOutputLocations[i] == DATA_LOCATION_GPU_BUFFER) {
          ctx->ioBinding->BindOutput(ctx->outputNames_cstr[i], ctx->gpuBufferMemoryInfo);
        } else {
          ctx->ioBinding->BindOutput(ctx->outputNames_cstr[i], ctx->cpuMemoryInfo);
        }
      }
    }
  } catch (...) {
    ctx->sessionRef.Reset();
    for (auto& ref : ctx->inputValueRefs) {
      ref.Reset();
    }
    delete ctx;
    throw;
  }

  napi_value async_resource_name;
  napi_create_string_utf8(env, "OnnxRuntimeRun", NAPI_AUTO_LENGTH, &async_resource_name);
  napi_create_async_work(env, nullptr, async_resource_name, RunWorkCallback, RunAfterWorkCallback, ctx, &ctx->work);
  napi_queue_async_work(env, ctx->work);

  return Napi::Value(env, promise_value);
}

Napi::Value InferenceSessionWrap::RunSync(const Napi::CallbackInfo& info) {
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

  std::vector<const char*> inputNames_cstr;
  std::vector<Ort::Value> inputValues;
  std::vector<const char*> outputNames_cstr;
  std::vector<Ort::Value> outputValues;
  std::vector<bool> reuseOutput;
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::MemoryInfo gpuBufferMemoryInfo{"WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault};

  try {
    for (auto& name : inputNames_) {
      if (feed.Has(name)) {
        inputIndex++;
        inputNames_cstr.push_back(name.c_str());
        auto value = feed.Get(name);
        inputValues.push_back(NapiValueToOrtValue(env, value, cpuMemoryInfo, gpuBufferMemoryInfo));
      }
    }
    for (auto& name : outputNames_) {
      if (fetch.Has(name)) {
        outputIndex++;
        outputNames_cstr.push_back(name.c_str());
        auto value = fetch.Get(name);
        reuseOutput.push_back(!value.IsNull());
        outputValues.emplace_back(value.IsNull() ? Ort::Value{nullptr} : NapiValueToOrtValue(env, value, cpuMemoryInfo, gpuBufferMemoryInfo));
      }
    }

    Ort::RunOptions runOptions{nullptr};
    if (info.Length() > 2) {
      runOptions = Ort::RunOptions{};
      ParseRunOptions(info[2].As<Napi::Object>(), runOptions);
    }
    if (preferredOutputLocations_.size() == 0) {
      session_->Run(runOptions == nullptr ? OrtSingletonData::GetOrtObjects()->default_run_options : runOptions,
                    inputIndex == 0 ? nullptr : &inputNames_cstr[0], inputIndex == 0 ? nullptr : &inputValues[0],
                    inputIndex, outputIndex == 0 ? nullptr : &outputNames_cstr[0],
                    outputIndex == 0 ? nullptr : &outputValues[0], outputIndex);

      Napi::Object result = Napi::Object::New(env);

      for (size_t i = 0; i < outputIndex; i++) {
        result.Set(outputNames_cstr[i], OrtValueToNapiValue(env, std::move(outputValues[i])));
      }
      return scope.Escape(result);
    } else {
      // IO binding
      ORT_NAPI_THROW_ERROR_IF(preferredOutputLocations_.size() != outputNames_.size(), env,
                              "Preferred output locations must have the same size as output names.");

      for (size_t i = 0; i < inputIndex; i++) {
        ioBinding_->BindInput(inputNames_cstr[i], inputValues[i]);
      }
      for (size_t i = 0; i < outputIndex; i++) {
        // TODO: support preallocated output tensor (outputValues[i])

        if (preferredOutputLocations_[i] == DATA_LOCATION_GPU_BUFFER) {
          ioBinding_->BindOutput(outputNames_cstr[i], gpuBufferMemoryInfo);
        } else {
          ioBinding_->BindOutput(outputNames_cstr[i], cpuMemoryInfo);
        }
      }

      session_->Run(runOptions == nullptr ? OrtSingletonData::GetOrtObjects()->default_run_options : runOptions, *ioBinding_);

      auto outputs = ioBinding_->GetOutputValues();
      ORT_NAPI_THROW_ERROR_IF(outputs.size() != outputIndex, env, "Output count mismatch.");

      Napi::Object result = Napi::Object::New(env);
      for (size_t i = 0; i < outputIndex; i++) {
        result.Set(outputNames_cstr[i], OrtValueToNapiValue(env, std::move(outputs[i])));
      }
      return scope.Escape(result);
    }
  } catch (Napi::Error const& e) {
    throw e;
  } catch (std::exception const& e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  this->inputTypes_.clear();
  this->outputTypes_.clear();

  this->ioBinding_.reset(nullptr);
  this->session_.reset(nullptr);

  this->disposed_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::EndProfiling(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);

  Ort::AllocatorWithDefaultOptions allocator;

  auto filename = session_->EndProfilingAllocated(allocator);
  Napi::String filenameValue = Napi::String::From(env, filename.get());
  return scope.Escape(filenameValue);
}

Napi::Value InferenceSessionWrap::ListSupportedBackends(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  Napi::Array result = Napi::Array::New(env);

  auto createObject = [&env](const std::string& name, const bool bundled) -> Napi::Object {
    Napi::Object result = Napi::Object::New(env);
    result.Set("name", name);
    result.Set("bundled", bundled);
    return result;
  };

  result.Set(uint32_t(0), createObject("cpu", true));

#ifdef USE_DML
  result.Set(result.Length(), createObject("dml", true));
#endif
#ifdef USE_WEBGPU
  result.Set(result.Length(), createObject("webgpu", true));
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
#ifdef USE_QNN
  result.Set(result.Length(), createObject("qnn", true));
#endif

  return scope.Escape(result);
}
