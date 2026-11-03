// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "device_lock.h"
#include "ep_context_data_read_helper.h"
#include "inference_session_wrap.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "run_options_helper.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ModelBufferView {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

ModelBufferView GetModelBufferView(Napi::ArrayBuffer buffer, int64_t byteOffset, int64_t byteLength) {
  const size_t bufferLength = buffer.ByteLength();
  if (byteOffset < 0 || byteLength < 0 || static_cast<uint64_t>(byteOffset) > bufferLength ||
      static_cast<uint64_t>(byteLength) > bufferLength - static_cast<size_t>(byteOffset)) {
    throw Napi::RangeError::New(buffer.Env(), "Model buffer offset or length is out of range.");
  }

  const size_t length = static_cast<size_t>(byteLength);
  const auto* data =
      length == 0 ? nullptr
                  : static_cast<const uint8_t*>(buffer.Data()) + static_cast<size_t>(byteOffset);
  return {data, length};
}

}  // namespace

std::shared_ptr<SessionCore> SessionCore::Create(bool requiresDeviceSerialization,
                                                 std::shared_ptr<EpContextDataReadState> epContextDataReadState) {
  auto core = std::shared_ptr<SessionCore>(new SessionCore{}, [](SessionCore* core) {
    if (core->requiresDeviceSerialization) {
      // Destroying the session tears down its execution provider, which is device work.
      ReleaseUnderDeviceLock([core] { delete core; });
    } else {
      delete core;
    }
  });
  core->requiresDeviceSerialization = requiresDeviceSerialization;
  core->epContextDataReadState = std::move(epContextDataReadState);
  return core;
}

void SessionCore::ReadSessionMetadata() {
  Ort::AllocatorWithDefaultOptions allocator;

  size_t count = session.GetInputCount();
  inputNames.reserve(count);
  inputTypes.reserve(count);
  for (size_t i = 0; i < count; i++) {
    auto inputName = session.GetInputNameAllocated(i, allocator);
    inputNames.emplace_back(inputName.get());
    inputTypes.push_back(session.GetInputTypeInfo(i));
  }

  count = session.GetOutputCount();
  outputNames.reserve(count);
  outputTypes.reserve(count);
  for (size_t i = 0; i < count; i++) {
    auto outputName = session.GetOutputNameAllocated(i, allocator);
    outputNames.emplace_back(outputName.get());
    outputTypes.push_back(session.GetOutputTypeInfo(i));
  }

  cpuAllocator = Ort::Allocator(session, Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault));
}

/**
 * LoadModelWorker constructs the native Ort::Session on a libuv worker thread.
 *
 * Constructing the session on a worker thread keeps the JavaScript event loop available, which is a
 * requirement for `sessionOptions.epContextDataRead`: ONNX Runtime calls that callback while the session
 * is being created and the call has to be marshalled back to the JavaScript thread.
 *
 * The worker owns everything the native constructor needs, including copied model/external-data buffers
 * and the callback state, so that they remain valid until construction completes.
 */
class LoadModelWorker : public Napi::AsyncWorker {
 public:
  LoadModelWorker(Napi::Env env, InferenceSessionWrap& wrap, const Napi::Object self, const Napi::Object options,
                  Ort::SessionOptions&& sessionOptions, bool requiresDeviceSerialization,
                  std::shared_ptr<EpContextDataReadState> epContextDataReadState,
                  std::vector<std::vector<char>>&& externalDataBuffers)
      : Napi::AsyncWorker{env, "onnxruntime.InferenceSession.loadModel"},
        wrap_{wrap},
        requiresDeviceSerialization_{requiresDeviceSerialization},
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
    const auto view = GetModelBufferView(buffer, byteOffset, byteLength);
    modelDataLength_ = view.size;
    if (modelDataLength_ != 0) {
      modelDataStorage_.assign(view.data, view.data + modelDataLength_);
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

      core_ = SessionCore::Create(requiresDeviceSerialization_, epContextDataReadState_);
      DeviceLock deviceLock{requiresDeviceSerialization_};
      if (useModelBuffer_) {
        core_->session = Ort::Session(ortObjects->env, modelData_, modelDataLength_, sessionOptions_);
      } else {
        core_->session = Ort::Session(ortObjects->env, modelPath_.c_str(), sessionOptions_);
      }

      core_->ReadSessionMetadata();
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
      wrap_.AdoptLoadedSession(std::move(core_), optionsRef_.Value());
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
  void ReleaseNativeSession() noexcept { core_.reset(); }

  InferenceSessionWrap& wrap_;
  const bool requiresDeviceSerialization_;

  // Keeps the wrapper object and options object alive until the worker completes.
  Napi::ObjectReference selfRef_;
  Napi::ObjectReference optionsRef_;

  // Owns every external initializer backing store referenced by sessionOptions_.
  std::vector<std::vector<char>> externalDataBuffers_;
  Ort::SessionOptions sessionOptions_;

  // Strong snapshot of the callback state: it must outlive the native session under construction.
  std::shared_ptr<EpContextDataReadState> epContextDataReadState_;

  std::basic_string<ORTCHAR_T> modelPath_;
  bool useModelBuffer_ = false;
  void* modelData_ = nullptr;
  size_t modelDataLength_ = 0;
  std::vector<uint8_t> modelDataStorage_;

  std::shared_ptr<SessionCore> core_;

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
      disposed_(false) {}

InferenceSessionWrap::~InferenceSessionWrap() {
  core_.reset();
  ReleaseEpContextDataReadState();
}

void InferenceSessionWrap::AdoptLoadedSession(std::shared_ptr<SessionCore> core, const Napi::Object options) {
  // Still unpublished, so it can be completed here.
  ParsePreferredOutputLocations(options, core->outputNames, core->preferredOutputLocations);
  this->core_ = std::move(core);

  this->loading_ = false;
  this->initialized_ = true;
}

void InferenceSessionWrap::ResetAfterFailedLoad() noexcept {
  this->loading_ = false;
  this->initialized_ = false;
  this->core_.reset();
  ReleaseEpContextDataReadState();
}

void InferenceSessionWrap::ReleaseEpContextDataReadState() noexcept {
  if (this->epContextDataReadState_) {
    this->epContextDataReadState_->Release();
    this->epContextDataReadState_.reset();
  }
}

Napi::Value InferenceSessionWrap::LoadModelSynchronously(const Napi::CallbackInfo& info, bool isModelPath,
                                                         const Napi::Object& options,
                                                         Ort::SessionOptions&& sessionOptions,
                                                         bool requiresDeviceSerialization) {
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  this->loading_ = true;

  try {
    auto* ortObjects = OrtSingletonData::GetOrtObjects();
    ORT_NAPI_THROW_ERROR_IF(ortObjects == nullptr, env, "ONNX Runtime is not initialized.");

    auto core = SessionCore::Create(requiresDeviceSerialization, nullptr);
    {
      // Blocks this thread while another session on the device runs, as session creation does in any case.
      DeviceLock deviceLock{requiresDeviceSerialization};
      if (isModelPath) {
        auto value = info[0].As<Napi::String>();
#ifdef _WIN32
        auto modelPath = value.Utf16Value();
        std::wstring wideModelPath{modelPath.begin(), modelPath.end()};
        core->session = Ort::Session(ortObjects->env, wideModelPath.c_str(), sessionOptions);
#else
        auto modelPath = value.Utf8Value();
        core->session = Ort::Session(ortObjects->env, modelPath.c_str(), sessionOptions);
#endif
      } else {
        const auto view = GetModelBufferView(info[0].As<Napi::ArrayBuffer>(),
                                             info[1].As<Napi::Number>().Int64Value(),
                                             info[2].As<Napi::Number>().Int64Value());
        core->session = Ort::Session(ortObjects->env, view.data, view.size, sessionOptions);
      }
    }

    core->ReadSessionMetadata();
    AdoptLoadedSession(std::move(core), options);
    deferred.Resolve(env.Undefined());
  } catch (Napi::Error const& e) {
    ResetAfterFailedLoad();
    deferred.Reject(e.Value());
  } catch (std::exception const& e) {
    ResetAfterFailedLoad();
    deferred.Reject(Napi::Error::New(env, e.what()).Value());
  } catch (...) {
    ResetAfterFailedLoad();
    deferred.Reject(Napi::Error::New(env, "Failed to create the inference session.").Value());
  }

  return deferred.Promise();
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
    ParseEpContextDataReadOptions(options, sessionOptions, this->epContextDataReadState_);

    bool requiresDeviceSerialization = false;
    if (!this->epContextDataReadState_) {
      ParseSessionOptions(options, sessionOptions, requiresDeviceSerialization);
      return scope.Escape(LoadModelSynchronously(info, isModelPath, options, std::move(sessionOptions),
                                                 requiresDeviceSerialization));
    }

    std::vector<std::vector<char>> externalDataBuffers;
    ParseSessionOptions(options, sessionOptions, requiresDeviceSerialization, &externalDataBuffers);

    worker = std::make_unique<LoadModelWorker>(env, *this, info.This().As<Napi::Object>(), options,
                                               std::move(sessionOptions), requiresDeviceSerialization,
                                               this->epContextDataReadState_, std::move(externalDataBuffers));

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
  auto& names = info.Data() != nullptr ? core_->inputNames : core_->outputNames;
  auto& types = info.Data() != nullptr ? core_->inputTypes : core_->outputTypes;
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

namespace {
/**
 * RunWorker runs one inference on a libuv worker thread.
 *
 * It holds its own reference to the SessionCore, so the session outlives the run even if dispose() is called or
 * the wrapper is collected meanwhile. Everything ORT reads during the run is owned by the worker: CPU inputs are
 * copies, gpu-buffer inputs and outputs are held by their DeviceValue, and preallocated CPU outputs are written
 * only once the run has completed.
 */
class RunWorker : public Napi::AsyncWorker {
 public:
  RunWorker(Napi::Env env, std::shared_ptr<SessionCore> core)
      : Napi::AsyncWorker{env, "onnxruntime.InferenceSession.run"},
        core_{std::move(core)},
        cpuMemoryInfo_{Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)},
        gpuBufferMemoryInfo_{"WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault} {}

  ~RunWorker() override {
    // Output values left over after an error may be device-backed.
    if (core_->requiresDeviceSerialization && !outputValues_.empty()) {
      auto values = std::make_shared<std::vector<Ort::Value>>(std::move(outputValues_));
      ReleaseUnderDeviceLock([values, core = core_] { values->clear(); });
    }
  }

  // Read the run arguments. Runs on the JavaScript thread.
  // @throw Napi::Error if an argument is invalid.
  void Prepare(const Napi::Object feed, const Napi::Object fetch, const Napi::Object options) {
    Napi::Env env = Env();
    for (const auto& name : core_->inputNames) {
      if (feed.Has(name)) {
        auto input = NapiValueToOrtValue(env, feed.Get(name), false, cpuMemoryInfo_, core_->cpuAllocator,
                                         gpuBufferMemoryInfo_);
        inputNames_.push_back(name.c_str());
        inputValues_.push_back(std::move(input.value));
        if (input.device_value) {
          deviceValues_.push_back(std::move(input.device_value));
        }
      }
    }

    for (size_t i = 0; i < core_->outputNames.size(); i++) {
      const auto& name = core_->outputNames[i];
      if (!fetch.Has(name)) {
        continue;
      }
      auto value = fetch.Get(name);
      Output output;
      output.index = i;
      if (value.IsNull()) {
        outputValues_.emplace_back(nullptr);
      } else {
        auto preallocated = NapiValueToOrtValue(env, value, true, cpuMemoryInfo_, core_->cpuAllocator,
                                                gpuBufferMemoryInfo_);
        if (preallocated.device_value) {
          // Device memory is only written in place through IoBinding: as a preallocated fetch, ORT may replace the
          // value instead of filling it, and there is no copy back to fall back on.
          ORT_NAPI_THROW_ERROR_IF(core_->preferredOutputLocations.empty(), env, "Preallocated output '", name,
                                  "' is on a device, which requires the session to be created with "
                                  "'preferredOutputLocation'.");
          deviceValues_.push_back(std::move(preallocated.device_value));
        } else {
          output.data = Napi::Persistent(preallocated.data);
          output.type = preallocated.type;
          output.dims = std::move(preallocated.dims);
        }
        output.tensor = Napi::Persistent(value);
        outputValues_.push_back(std::move(preallocated.value));
      }
      outputNames_.push_back(name.c_str());
      outputs_.push_back(std::move(output));
    }

    ParseRunOptions(options, runOptions_);
  }

  // Queue the run. From here on the worker deletes itself once it has completed.
  Napi::Promise Start() {
    deferred_.emplace(Napi::Promise::Deferred::New(Env()));
    Queue();
    return deferred_->Promise();
  }

 protected:
  void Execute() override {
    {
      DeviceLock deviceLock{core_->requiresDeviceSerialization};
      try {
        if (core_->preferredOutputLocations.empty()) {
          core_->session.Run(runOptions_, inputNames_.data(), inputValues_.data(), inputValues_.size(),
                             outputNames_.data(), outputValues_.data(), outputValues_.size());
        } else {
          Ort::IoBinding ioBinding{core_->session};
          for (size_t i = 0; i < inputNames_.size(); i++) {
            ioBinding.BindInput(inputNames_[i], inputValues_[i]);
          }
          for (size_t i = 0; i < outputs_.size(); i++) {
            if (static_cast<OrtValue*>(outputValues_[i]) != nullptr) {
              ioBinding.BindOutput(outputNames_[i], outputValues_[i]);
            } else if (outputs_[i].tensor.IsEmpty() &&
                       core_->preferredOutputLocations[outputs_[i].index] == DATA_LOCATION_GPU_BUFFER) {
              ioBinding.BindOutput(outputNames_[i], gpuBufferMemoryInfo_);
            } else {
              ioBinding.BindOutput(outputNames_[i], cpuMemoryInfo_);
            }
          }
          core_->session.Run(runOptions_, ioBinding);
          outputValues_ = ioBinding.GetOutputValues();
        }
      } catch (const std::exception& e) {
        SetError(e.what());
        outputValues_.clear();
      }

      // Release what may refer to device memory while still holding the lock.
      inputValues_.clear();
      deviceValues_.clear();
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::HandleScope scope{env};
    try {
      deferred_->Resolve(CreateResult(env));
    } catch (const Napi::Error& e) {
      deferred_->Reject(e.Value());
    } catch (const std::exception& e) {
      deferred_->Reject(Napi::Error::New(env, e.what()).Value());
    }
  }

  void OnError(const Napi::Error& e) override { deferred_->Reject(e.Value()); }

 private:
  struct Output {
    // index into SessionCore::outputNames
    size_t index{0};
    // the caller's Tensor, for a preallocated output; it is what the result holds for this output
    Napi::Reference<Napi::Value> tensor;
    // for a preallocated CPU output: its typed array, and the type and shape the result must have
    Napi::Reference<Napi::Value> data;
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::vector<int64_t> dims;
  };

  Napi::Object CreateResult(Napi::Env env) {
    ORT_NAPI_THROW_ERROR_IF(outputValues_.size() != outputs_.size(), env, "Output count mismatch.");

    // Creating a Tensor runs Javascript, which could detach a preallocated buffer. Create them all before any buffer
    // is checked and written, and check every buffer before writing any, so a rejected run leaves them untouched.
    std::vector<Napi::Value> values(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (outputs_[i].tensor.IsEmpty()) {
        values[i] = OrtValueToNapiValue(env, std::move(outputValues_[i]), core_);
      }
    }
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (!outputs_[i].data.IsEmpty()) {
        ValidatePreallocatedOutput(env, outputValues_[i], outputs_[i].data.Value(), outputs_[i].type,
                                   outputs_[i].dims);
      }
    }
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (!outputs_[i].data.IsEmpty()) {
        CopyOrtValueToNapiTypedArray(outputValues_[i], outputs_[i].data.Value());
      }
      if (!outputs_[i].tensor.IsEmpty()) {
        values[i] = outputs_[i].tensor.Value();
      }
    }

    auto result = Napi::Object::New(env);
    for (size_t i = 0; i < outputs_.size(); i++) {
      result.Set(outputNames_[i], values[i]);
    }
    return result;
  }

  // Declared first so that it is released last: every value below may come from the session's allocators.
  std::shared_ptr<SessionCore> core_;
  Ort::MemoryInfo cpuMemoryInfo_;
  Ort::MemoryInfo gpuBufferMemoryInfo_;
  Ort::RunOptions runOptions_;

  // The names point into core_, which does not change while it is alive.
  std::vector<const char*> inputNames_;
  std::vector<Ort::Value> inputValues_;
  std::vector<DeviceValuePtr> deviceValues_;
  std::vector<const char*> outputNames_;
  std::vector<Ort::Value> outputValues_;
  std::vector<Output> outputs_;

  std::optional<Napi::Promise::Deferred> deferred_;
};
}  // namespace

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

  // The worker takes its reference to the session before any argument is read: a getter on a feed may call
  // dispose(), and the run has to keep working with the session it started on.
  auto worker = std::make_unique<RunWorker>(env, core_);
  try {
    worker->Prepare(info[0].As<Napi::Object>(), info[1].As<Napi::Object>(),
                    info.Length() > 2 ? info[2].As<Napi::Object>() : Napi::Object::New(env));
  } catch (Napi::Error const& e) {
    throw e;
  } catch (std::exception const& e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }

  auto promise = worker->Start();
  worker.release();
  return scope.Escape(promise);
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  // Runs still in flight hold their own reference and release the session when the last one completes.
  this->core_.reset();

  // A session kept alive by a run still holds the callback state through SessionCore; once released here, a late
  // callback fails with a status instead of reaching JavaScript.
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

  auto filename = core_->session.EndProfilingAllocated(allocator);
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
