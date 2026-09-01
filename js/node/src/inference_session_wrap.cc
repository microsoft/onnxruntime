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
#include <string>

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

class InferenceSessionWrap::RunAsyncWorker : public Napi::AsyncWorker {
 public:
  using OutputBufferLease = OrtInstanceData::OutputBufferLease;

  RunAsyncWorker(InferenceSessionWrap& session, const Napi::Object& feed, const Napi::Object& fetch,
                 const Napi::Object& options, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(session.Value().Env(), "InferenceSession.run", session.Value()),
        env_(session.Value().Env()),
        session_(&session),
        deferred_(deferred),
        session_reference_(Napi::Persistent(session.Value())),
        keep_alive_reference_(Napi::Persistent(Napi::Array::New(env_))),
        cpu_memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        gpu_buffer_memory_info_("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault) {
    auto keep_alive = keep_alive_reference_.Value().As<Napi::Array>();

    for (const auto& name : session.inputNames_) {
      if (feed.Has(name)) {
        auto value = feed.Get(name);
        input_names_.push_back(name);
        input_values_.push_back(
            NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_, NapiValueUsage::kInput,
                                &gpu_value_owners_));
        KeepTensorAndDataAlive(keep_alive, value);
      }
    }

    for (const auto& name : session.outputNames_) {
      if (fetch.Has(name)) {
        auto value = fetch.Get(name);
        output_names_.push_back(name);
        reuse_output_.push_back(!value.IsNull());
        if (value.IsNull()) {
          output_values_.emplace_back(nullptr);
          output_expected_.emplace_back();
          output_js_value_indices_.push_back(0);
          output_js_data_indices_.push_back(0);
          output_buffer_key_indices_.push_back(0);
          copy_output_to_js_.push_back(false);
        } else {
          PreallocatedOutputInfo expected;
          auto output_value = NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_,
                                                  NapiValueUsage::kPreallocatedOutput, &gpu_value_owners_, &expected);
          // CPU outputs come back empty: ORT allocates them and OnOK() validates the result against
          // 'expected' before copying it into the caller's buffer. Device outputs carry the caller's
          // memory and are bound directly, so ORT writes into it and no copy is needed.
          copy_output_to_js_.push_back(static_cast<OrtValue*>(output_value) == nullptr);
          output_values_.push_back(std::move(output_value));
          output_expected_.push_back(std::move(expected));
          uint32_t data_index = 0;
          uint32_t buffer_key_index = 0;
          output_js_value_indices_.push_back(
              KeepTensorAndDataAlive(keep_alive, value, &data_index, &buffer_key_index));
          output_js_data_indices_.push_back(data_index);
          output_buffer_key_indices_.push_back(buffer_key_index);
        }
      }
    }

    preferred_output_locations_ = session.preferredOutputLocations_;
    ParseRunOptions(options, run_options_);
  }

  ~RunAsyncWorker() override {
    ReleaseOutputBufferLeases();
  }

  void AcquireOutputBufferLeases() {
    auto keep_alive = keep_alive_reference_.Value().As<Napi::Array>();
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (reuse_output_[i]) {
        output_buffer_leases_.push_back(
            OrtInstanceData::AcquireOutputBufferLease(
                keep_alive.Get(output_buffer_key_indices_[i]).As<Napi::Object>()));
      }
    }
  }

  void Execute() override {
    try {
      std::vector<const char*> input_names_cstr;
      input_names_cstr.reserve(input_names_.size());
      for (const auto& name : input_names_) {
        input_names_cstr.push_back(name.c_str());
      }

      std::vector<const char*> output_names_cstr;
      output_names_cstr.reserve(output_names_.size());
      for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
      }

      if (preferred_output_locations_.empty()) {
        session_->session_->Run(run_options_, input_names_cstr.data(), input_values_.data(), input_values_.size(),
                                output_names_cstr.data(), output_values_.data(), output_values_.size());
        return;
      }

      if (preferred_output_locations_.size() != session_->outputNames_.size()) {
        SetError("Preferred output locations must have the same size as output names.");
        return;
      }

      io_binding_ = std::make_unique<Ort::IoBinding>(*session_->session_);
      for (size_t i = 0; i < input_names_.size(); ++i) {
        io_binding_->BindInput(input_names_cstr[i], input_values_[i]);
      }

      for (size_t i = 0; i < output_names_.size(); ++i) {
        if (reuse_output_[i]) {
          if (static_cast<OrtValue*>(output_values_[i]) != nullptr) {
            // Device output: ORT writes straight into the caller's buffer.
            io_binding_->BindOutput(output_names_cstr[i], output_values_[i]);
          } else {
            // Preallocated CPU output: ORT allocates and OnOK() copies into the caller's buffer.
            io_binding_->BindOutput(output_names_cstr[i], cpu_memory_info_);
          }
        } else if (preferred_output_locations_[i] == DATA_LOCATION_GPU_BUFFER) {
          io_binding_->BindOutput(output_names_cstr[i], gpu_buffer_memory_info_);
        } else {
          io_binding_->BindOutput(output_names_cstr[i], cpu_memory_info_);
        }
      }

      session_->session_->Run(run_options_, *io_binding_);
      output_values_ = io_binding_->GetOutputValues();
      if (output_values_.size() != output_names_.size()) {
        SetError("Output count mismatch.");
      }
    } catch (const std::exception& e) {
      SetError(e.what());
    } catch (...) {
      SetError("Unknown error while running the model.");
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(env_);
    try {
      auto result = Napi::Object::New(env_);
      auto keep_alive = keep_alive_reference_.Value().As<Napi::Array>();
      for (size_t i = 0; i < output_values_.size(); ++i) {
        if (copy_output_to_js_[i]) {
          CopyOrtValueToNapiTypedArray(env_, output_values_[i], keep_alive.Get(output_js_data_indices_[i]),
                                       output_expected_[i]);
        }
        result.Set(output_names_[i], reuse_output_[i] ? keep_alive.Get(output_js_value_indices_[i])
                                                      : OrtValueToNapiValue(env_, std::move(output_values_[i])));
      }
      Complete();
      deferred_.Resolve(result);
    } catch (const Napi::Error& e) {
      Complete();
      deferred_.Reject(e.Value());
    } catch (const std::exception& e) {
      Complete();
      deferred_.Reject(Napi::Error::New(env_, e.what()).Value());
    } catch (...) {
      Complete();
      deferred_.Reject(Napi::Error::New(env_, "Unknown error while converting model outputs.").Value());
    }
  }

  void OnError(const Napi::Error& error) override {
    Complete();
    deferred_.Reject(error.Value());
  }

 private:
  uint32_t KeepTensorAndDataAlive(Napi::Array& keep_alive, const Napi::Value& value,
                                  uint32_t* data_index = nullptr, uint32_t* buffer_key_index = nullptr) {
    const auto tensor_index = keep_alive.Length();
    keep_alive.Set(tensor_index, value);
    if (data_index != nullptr) {
      *data_index = tensor_index;
    }
    if (buffer_key_index != nullptr) {
      *buffer_key_index = tensor_index;
    }
    if (!value.IsObject()) {
      return tensor_index;
    }

    auto tensor = value.As<Napi::Object>();
    const auto location = tensor.Get("location");
    if (!location.IsString()) {
      return tensor_index;
    }

    const auto location_string = location.As<Napi::String>().Utf8Value();
    if (location_string == "cpu" || location_string == "cpu-pinned") {
      const auto data = tensor.Get("data");
      const auto index = keep_alive.Length();
      keep_alive.Set(index, data);
      if (data_index != nullptr) {
        *data_index = index;
      }
      if (buffer_key_index != nullptr) {
        auto buffer = data.As<Napi::TypedArray>().ArrayBuffer();
        const auto buffer_index = keep_alive.Length();
        keep_alive.Set(buffer_index, buffer);
        *buffer_key_index = buffer_index;
      }
    } else if (location_string == "gpu-buffer") {
      const auto index = keep_alive.Length();
      keep_alive.Set(index, tensor.Get("gpuBuffer"));
      if (buffer_key_index != nullptr) {
        *buffer_key_index = index;
      }
    }
    return tensor_index;
  }

  void Complete() {
    if (completed_) {
      return;
    }
    completed_ = true;

    ReleaseOutputBufferLeases();

    session_->active_runs_.fetch_sub(1, std::memory_order_release);
  }

  void ReleaseOutputBufferLeases() {
    for (const auto& lease : output_buffer_leases_) {
      OrtInstanceData::ReleaseOutputBufferLease(env_, lease);
    }
    output_buffer_leases_.clear();
  }

  Napi::Env env_;
  InferenceSessionWrap* session_;
  Napi::Promise::Deferred deferred_;
  Napi::ObjectReference session_reference_;
  Napi::Reference<Napi::Array> keep_alive_reference_;
  Ort::MemoryInfo cpu_memory_info_;
  Ort::MemoryInfo gpu_buffer_memory_info_;
  Ort::RunOptions run_options_;
  std::vector<OrtValueOwner> gpu_value_owners_;
  std::vector<std::string> input_names_;
  std::vector<Ort::Value> input_values_;
  std::vector<std::string> output_names_;
  std::vector<Ort::Value> output_values_;
  std::vector<PreallocatedOutputInfo> output_expected_;
  std::vector<bool> reuse_output_;
  std::vector<uint32_t> output_js_value_indices_;
  std::vector<uint32_t> output_js_data_indices_;
  std::vector<uint32_t> output_buffer_key_indices_;
  std::vector<bool> copy_output_to_js_;
  std::vector<OutputBufferLease> output_buffer_leases_;
  std::vector<int> preferred_output_locations_;
  std::unique_ptr<Ort::IoBinding> io_binding_;
  bool completed_{false};
};

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() < 2, env, "Expect argument: inputs(feed) and outputs(fetch).");
  ORT_NAPI_THROW_TYPEERROR_IF(!info[0].IsObject() || !info[1].IsObject(), env,
                              "Expect inputs(feed) and outputs(fetch) to be objects.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() > 2 && (!info[2].IsObject() || info[2].IsNull()), env,
                              "'runOptions' must be an object.");

  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();
  auto options = info.Length() > 2 ? info[2].As<Napi::Object>() : Napi::Object::New(env);
  auto deferred = Napi::Promise::Deferred::New(env);

  std::unique_ptr<RunAsyncWorker> worker;
  const auto unregister_active_run = [this]() { active_runs_.fetch_sub(1, std::memory_order_release); };

  // Register the run before reading 'feed' and 'fetch': those reads can re-enter JS through
  // getters or Proxy traps, and a reentrant dispose() must not be allowed to tear the session
  // down while this run is still being prepared.
  active_runs_.fetch_add(1, std::memory_order_acquire);
  try {
    worker = std::make_unique<RunAsyncWorker>(*this, feed, fetch, options, deferred);
    worker->AcquireOutputBufferLeases();
    worker->Queue();
  } catch (const Napi::Error&) {
    unregister_active_run();
    throw;
  } catch (const std::exception& e) {
    unregister_active_run();
    ORT_NAPI_THROW_ERROR(env, e.what());
  } catch (...) {
    unregister_active_run();
    ORT_NAPI_THROW_ERROR(env, "Unknown error while preparing inference.");
  }

  worker.release();
  return deferred.Promise();
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_ERROR_IF(this->active_runs_.load(std::memory_order_acquire) != 0, env,
                          "Cannot dispose session while inference is running.");

  this->inputTypes_.clear();
  this->outputTypes_.clear();

  this->session_.reset(nullptr);

  this->disposed_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::EndProfiling(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_ERROR_IF(this->active_runs_.load(std::memory_order_acquire) != 0, env,
                          "Cannot end profiling while inference is running.");

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
