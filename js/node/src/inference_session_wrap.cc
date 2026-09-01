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
        PinTensorAndKeepDataAlive(keep_alive, value);
        input_names_.push_back(name);
        input_values_.push_back(NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_));
      }
    }

    for (const auto& name : session.outputNames_) {
      if (fetch.Has(name)) {
        auto value = fetch.Get(name);
        output_names_.push_back(name);
        if (value.IsNull()) {
          output_values_.emplace_back(nullptr);
        } else {
          PinTensorAndKeepDataAlive(keep_alive, value);
          output_values_.push_back(NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_));
        }
      }
    }

    preferred_output_locations_ = session.preferredOutputLocations_;
    ParseRunOptions(options, run_options_);
  }

  ~RunAsyncWorker() override {
    try {
      UnpinTensors();
    } catch (...) {
      // Do not let a user-mutated Tensor object abort worker cleanup.
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
        if (preferred_output_locations_[i] == DATA_LOCATION_GPU_BUFFER) {
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
      for (size_t i = 0; i < output_values_.size(); ++i) {
        result.Set(output_names_[i], OrtValueToNapiValue(env_, std::move(output_values_[i])));
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
  void PinTensorAndKeepDataAlive(Napi::Array& keep_alive, const Napi::Value& value) {
    // Keep both the Tensor and its backing resource alive, and prevent JS from disposing or accessing it
    // while ORT may still be using the underlying buffer.
    keep_alive.Set(keep_alive.Length(), value);

    if (!value.IsObject()) {
      return;
    }

    auto tensor = value.As<Napi::Object>();
    const auto dispose = tensor.Get("dispose");
    if (dispose.IsFunction()) {
      for (const auto record_index : pinned_tensor_indices_) {
        const auto record = keep_alive.Get(record_index).As<Napi::Array>();
        if (record.Get(uint32_t(0)).StrictEquals(tensor)) {
          return;
        }
      }

      const auto record_index = keep_alive.Length();
      const auto get_data = tensor.Get("getData");
      auto record = Napi::Array::New(env_, 5);
      record.Set(uint32_t(0), tensor);
      record.Set(uint32_t(1), dispose);
      record.Set(uint32_t(2), tensor.HasOwnProperty("dispose"));
      record.Set(uint32_t(3), get_data);
      record.Set(uint32_t(4), tensor.HasOwnProperty("getData"));

      keep_alive.Set(record_index, record);
      pinned_tensor_indices_.push_back(record_index);

      const auto guard = CreateTensorUseGuard(env_);
      tensor.Set("dispose", guard);
      if (get_data.IsFunction()) {
        tensor.Set("getData", guard);
      }
    }

    const auto location = tensor.Get("location");
    if (!location.IsString()) {
      return;
    }

    const auto location_string = location.As<Napi::String>().Utf8Value();
    if (location_string == "cpu" || location_string == "cpu-pinned") {
      keep_alive.Set(keep_alive.Length(), tensor.Get("data"));
    } else if (location_string == "gpu-buffer") {
      keep_alive.Set(keep_alive.Length(), tensor.Get("gpuBuffer"));
    }
  }

  void UnpinTensors() {
    if (pinned_tensor_indices_.empty()) {
      return;
    }

    auto keep_alive = keep_alive_reference_.Value().As<Napi::Array>();
    for (const auto record_index : pinned_tensor_indices_) {
      const auto record = keep_alive.Get(record_index).As<Napi::Array>();
      auto tensor = record.Get(uint32_t(0)).As<Napi::Object>();

      if (record.Get(uint32_t(2)).As<Napi::Boolean>().Value()) {
        tensor.Set("dispose", record.Get(uint32_t(1)));
      } else {
        tensor.Delete("dispose");
      }

      if (record.Get(uint32_t(4)).As<Napi::Boolean>().Value()) {
        tensor.Set("getData", record.Get(uint32_t(3)));
      } else if (record.Get(uint32_t(3)).IsFunction()) {
        tensor.Delete("getData");
      }
    }
    pinned_tensor_indices_.clear();
  }

  static Napi::Function CreateTensorUseGuard(Napi::Env env) {
    return Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
      ORT_NAPI_THROW_ERROR(info.Env(), "Tensor is being used by an asynchronous inference.");
    });
  }

  void Complete() {
    if (completed_) {
      return;
    }
    completed_ = true;

    try {
      UnpinTensors();
    } catch (...) {
      // The run must still release the session even if Tensor restoration fails.
    }
    session_->active_runs_.fetch_sub(1, std::memory_order_release);
  }

  Napi::Env env_;
  InferenceSessionWrap* session_;
  Napi::Promise::Deferred deferred_;
  Napi::ObjectReference session_reference_;
  Napi::Reference<Napi::Array> keep_alive_reference_;
  Ort::MemoryInfo cpu_memory_info_;
  Ort::MemoryInfo gpu_buffer_memory_info_;
  Ort::RunOptions run_options_;
  std::vector<std::string> input_names_;
  std::vector<Ort::Value> input_values_;
  std::vector<std::string> output_names_;
  std::vector<Ort::Value> output_values_;
  std::vector<int> preferred_output_locations_;
  std::vector<uint32_t> pinned_tensor_indices_;
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
  bool active_run_registered = false;
  try {
    worker = std::make_unique<RunAsyncWorker>(*this, feed, fetch, options, deferred);
    active_runs_.fetch_add(1, std::memory_order_acquire);
    active_run_registered = true;
    worker->Queue();
  } catch (...) {
    if (active_run_registered) {
      active_runs_.fetch_sub(1, std::memory_order_release);
    }
    throw;
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
