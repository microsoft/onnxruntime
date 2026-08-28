// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "ep_context_data_read_helper.h"
#include "inference_session_wrap.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "run_options_helper.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * LoadModelWorker constructs the native Ort::Session on a libuv worker thread.
 *
 * Constructing the session on a worker thread keeps the JavaScript event loop available, which is a
 * requirement for `sessionOptions.epContextDataRead`: ONNX Runtime calls that callback while the session
 * is being created and the call has to be marshalled back to the JavaScript thread.
 *
 * The worker owns everything the native constructor needs, including references to the JavaScript values
 * that back the model path/buffer and the session options, so that they stay alive until it completes.
 */
class LoadModelWorker : public Napi::AsyncWorker {
 public:
  LoadModelWorker(Napi::Env env, InferenceSessionWrap& wrap, const Napi::Object self, const Napi::Object options,
                  Ort::SessionOptions&& sessionOptions,
                  std::shared_ptr<EpContextDataReadState> epContextDataReadState,
                  std::vector<std::vector<char>>&& externalDataBuffers)
      : Napi::AsyncWorker{env, "onnxruntime.InferenceSession.loadModel"},
        wrap_{wrap},
        selfRef_{Napi::Persistent(self)},
        optionsRef_{Napi::Persistent(options)},
        externalDataBuffers_{std::move(externalDataBuffers)},
        sessionOptions_{std::move(sessionOptions)},
        epContextDataReadState_{std::move(epContextDataReadState)},
        deferred_{Napi::Promise::Deferred::New(env)} {
    // The ORT singleton must not be destroyed by an environment cleanup hook while the worker thread
    // is still using it.
    OrtSingletonData::RetainOrtObjects();
  }

  ~LoadModelWorker() override {
    // Give up the ORT objects before the reference that keeps the ORT singleton alive.
    ReleaseNativeSession();
    OrtSingletonData::ReleaseOrtObjects();
  }

  // Keep the model path alive for the duration of the native call.
  void SetModelPath(std::basic_string<ORTCHAR_T> modelPath) { modelPath_ = std::move(modelPath); }

  // Copy the model before yielding to JavaScript. A persistent reference does not pin an ArrayBuffer's
  // backing store, which can be detached or transferred while the worker is running.
  void SetModelBuffer(Napi::ArrayBuffer buffer, int64_t byteOffset, int64_t byteLength) {
    const size_t bufferLength = buffer.ByteLength();
    if (byteOffset < 0 || byteLength < 0 || static_cast<uint64_t>(byteOffset) > bufferLength ||
        static_cast<uint64_t>(byteLength) > bufferLength - static_cast<size_t>(byteOffset)) {
      throw Napi::RangeError::New(buffer.Env(), "Model buffer offset or length is out of range.");
    }

    modelDataLength_ = static_cast<size_t>(byteLength);
    if (modelDataLength_ != 0) {
      const auto* data = static_cast<const uint8_t*>(buffer.Data()) + static_cast<size_t>(byteOffset);
      modelDataStorage_.assign(data, data + modelDataLength_);
      modelData_ = modelDataStorage_.data();
    }
    useModelBuffer_ = true;
  }

  Napi::Promise Promise() const { return deferred_.Promise(); }

 protected:
  void Execute() override {
    try {
      auto* ortObjects = OrtSingletonData::GetOrtObjects();
      if (ortObjects == nullptr) {
        SetError("ONNX Runtime is not initialized.");
        return;
      }

      if (useModelBuffer_) {
        session_ = std::make_unique<Ort::Session>(ortObjects->env, modelData_, modelDataLength_, sessionOptions_);
      } else {
        session_ = std::make_unique<Ort::Session>(ortObjects->env, modelPath_.c_str(), sessionOptions_);
      }

      // cache input/output names and types
      Ort::AllocatorWithDefaultOptions allocator;

      size_t count = session_->GetInputCount();
      inputNames_.reserve(count);
      inputTypes_.reserve(count);
      for (size_t i = 0; i < count; i++) {
        auto inputName = session_->GetInputNameAllocated(i, allocator);
        inputNames_.emplace_back(inputName.get());
        inputTypes_.push_back(session_->GetInputTypeInfo(i));
      }

      count = session_->GetOutputCount();
      outputNames_.reserve(count);
      outputTypes_.reserve(count);
      for (size_t i = 0; i < count; i++) {
        auto outputName = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.emplace_back(outputName.get());
        outputTypes_.push_back(session_->GetOutputTypeInfo(i));
      }
    } catch (std::exception const& e) {
      ReleaseNativeSession();
      SetError(e.what());
    } catch (...) {
      ReleaseNativeSession();
      SetError("Failed to create the inference session.");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::HandleScope scope{env};

    try {
      wrap_.AdoptLoadedSession(std::move(session_), std::move(inputNames_), std::move(inputTypes_),
                               std::move(outputNames_), std::move(outputTypes_), optionsRef_.Value());
    } catch (Napi::Error const& e) {
      ReleaseNativeSession();
      wrap_.ResetAfterFailedLoad();
      deferred_.Reject(e.Value());
      return;
    } catch (std::exception const& e) {
      ReleaseNativeSession();
      wrap_.ResetAfterFailedLoad();
      deferred_.Reject(Napi::Error::New(env, e.what()).Value());
      return;
    }

    deferred_.Resolve(env.Undefined());
  }

  void OnError(const Napi::Error& e) override {
    Napi::Env env = Env();
    Napi::HandleScope scope{env};

    // The native session is released before the callback state that it can call into.
    ReleaseNativeSession();
    wrap_.ResetAfterFailedLoad();
    deferred_.Reject(e.Value());
  }

 private:
  void ReleaseNativeSession() noexcept {
    inputTypes_.clear();
    outputTypes_.clear();
    session_.reset(nullptr);
  }

  InferenceSessionWrap& wrap_;

  // Keeps the wrapper object and options object alive until the worker completes.
  Napi::ObjectReference selfRef_;
  Napi::ObjectReference optionsRef_;

  // Owns every external initializer backing store referenced by sessionOptions_.
  std::vector<std::vector<char>> externalDataBuffers_;
  Ort::SessionOptions sessionOptions_;

  // Strong snapshot of the callback state: it must outlive the native session under construction.
  // Declared before `session_` so that the session is destroyed first.
  std::shared_ptr<EpContextDataReadState> epContextDataReadState_;

  std::basic_string<ORTCHAR_T> modelPath_;
  bool useModelBuffer_ = false;
  void* modelData_ = nullptr;
  size_t modelDataLength_ = 0;
  std::vector<uint8_t> modelDataStorage_;

  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> inputNames_;
  std::vector<Ort::TypeInfo> inputTypes_;
  std::vector<std::string> outputNames_;
  std::vector<Ort::TypeInfo> outputTypes_;

  Napi::Promise::Deferred deferred_;
};

Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
  // create ONNX runtime env
  Ort::InitApi();

  // initialize binding
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "InferenceSession",
      {InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel),
       InstanceMethod("run", &InferenceSessionWrap::Run),
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
    : Napi::ObjectWrap<InferenceSessionWrap>(info),
      initialized_(false),
      loading_(false),
      disposed_(false),
      session_(nullptr) {}

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
  } else {
    ioBinding_.reset(nullptr);
    session_.reset(nullptr);
  }

  // The native session is gone, so the callback state can no longer be reached from ONNX Runtime.
  ReleaseEpContextDataReadState();
}

void InferenceSessionWrap::AdoptLoadedSession(std::unique_ptr<Ort::Session> session,
                                              std::vector<std::string> inputNames,
                                              std::vector<Ort::TypeInfo> inputTypes,
                                              std::vector<std::string> outputNames,
                                              std::vector<Ort::TypeInfo> outputTypes, const Napi::Object options) {
  this->session_ = std::move(session);
  this->inputNames_ = std::move(inputNames);
  this->inputTypes_ = std::move(inputTypes);
  this->outputNames_ = std::move(outputNames);
  this->outputTypes_ = std::move(outputTypes);

  // cache preferred output locations
  ParsePreferredOutputLocations(options, outputNames_, preferredOutputLocations_);
  if (preferredOutputLocations_.size() > 0) {
    ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);
  }

  this->loading_ = false;
  this->initialized_ = true;
}

void InferenceSessionWrap::ResetAfterFailedLoad() noexcept {
  this->loading_ = false;
  this->initialized_ = false;

  this->inputNames_.clear();
  this->outputNames_.clear();
  this->preferredOutputLocations_.clear();

  if (OrtSingletonData::GetOrtObjects()) {
    this->inputTypes_.clear();
    this->outputTypes_.clear();
    this->ioBinding_.reset(nullptr);
    this->session_.reset(nullptr);
  }

  // Deterministic teardown: the callback state is released only after the native session is gone.
  ReleaseEpContextDataReadState();
}

void InferenceSessionWrap::ReleaseEpContextDataReadState() noexcept {
  if (this->epContextDataReadState_) {
    this->epContextDataReadState_->Release();
    this->epContextDataReadState_.reset();
  }
}

Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);

  ORT_NAPI_THROW_ERROR_IF(this->initialized_, env, "Model already loaded. Cannot load model multiple times.");
  ORT_NAPI_THROW_ERROR_IF(this->loading_, env, "Model is being loaded. Cannot load model multiple times.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  size_t argsLength = info.Length();
  ORT_NAPI_THROW_TYPEERROR_IF(argsLength == 0, env, "Expect argument: model file path or buffer.");

  const bool isModelPath = argsLength == 2 && info[0].IsString() && info[1].IsObject();
  const bool isModelBuffer = argsLength == 4 && info[0].IsArrayBuffer() && info[1].IsNumber() &&
                             info[2].IsNumber() && info[3].IsObject();
  ORT_NAPI_THROW_TYPEERROR_IF(
      !isModelPath && !isModelBuffer, env,
      "Invalid argument: args has to be either (modelPath, options) or (buffer, byteOffset, byteLength, options).");

  auto options = info[argsLength - 1].As<Napi::Object>();

  std::unique_ptr<LoadModelWorker> worker;
  try {
    Ort::SessionOptions sessionOptions;
    std::vector<std::vector<char>> externalDataBuffers;
    ParseSessionOptions(options, sessionOptions, externalDataBuffers);
    ParseEpContextDataReadOptions(options, sessionOptions, this->epContextDataReadState_);

    worker = std::make_unique<LoadModelWorker>(env, *this, info.This().As<Napi::Object>(), options,
                                               std::move(sessionOptions), this->epContextDataReadState_,
                                               std::move(externalDataBuffers));

    if (isModelPath) {
      auto value = info[0].As<Napi::String>();
#ifdef _WIN32
      auto modelPath = value.Utf16Value();
      worker->SetModelPath(std::wstring{modelPath.begin(), modelPath.end()});
#else
      worker->SetModelPath(value.Utf8Value());
#endif
    } else {
      worker->SetModelBuffer(info[0].As<Napi::ArrayBuffer>(), info[1].As<Napi::Number>().Int64Value(),
                             info[2].As<Napi::Number>().Int64Value());
    }
  } catch (Napi::Error const& e) {
    worker.reset(nullptr);
    ReleaseEpContextDataReadState();
    throw e;
  } catch (std::exception const& e) {
    worker.reset(nullptr);
    ReleaseEpContextDataReadState();
    ORT_NAPI_THROW_ERROR(env, e.what());
  }

  this->loading_ = true;
  auto promise = worker->Promise();
  // Napi::AsyncWorker deletes itself once it completed.
  try {
    worker->Queue();
    worker.release();
  } catch (...) {
    this->loading_ = false;
    ReleaseEpContextDataReadState();
    throw;
  }
  return scope.Escape(promise);
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

  // Deterministic teardown: the native session is released before the state it can call into.
  this->ReleaseEpContextDataReadState();

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
