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
#include <mutex>
#include <string>

namespace {
// Owns run values handed to OrtInstanceData::ReleaseDeviceObject(). If the environment cleanup hook
// destroyed the ORT singleton before these are drained -- at process exit, say -- releasing them
// would call into an unloaded library, so leak instead, as ~InferenceSessionWrap does.
struct DeviceValues {
  DeviceValues(std::vector<Ort::Value>&& moved, std::shared_ptr<Ort::Session> owning_session)
      : values(std::move(moved)), session(std::move(owning_session)) {}

  std::vector<Ort::Value> values;
  // These may be device-backed, in which case their buffers belong to an allocator owned by the
  // session's execution provider. Draining can happen long after the run, so hold the session.
  std::shared_ptr<Ort::Session> session;

  ~DeviceValues() {
    if (!OrtSingletonData::GetOrtObjects()) {
      for (auto& value : values) {
        (void)value.release();
      }
      new std::shared_ptr<Ort::Session>(std::move(session));
    }
  }
};
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
    // Intentionally leak the session as well: a device-backed value may still reference it.
    new std::shared_ptr<Ort::Session>(std::move(session_));
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

      ParseSessionOptions(info[1].As<Napi::Object>(), sessionOptions, &requires_device_serialization_);
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

      ParseSessionOptions(info[3].As<Napi::Object>(), sessionOptions, &requires_device_serialization_);
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

// Inference is dispatched with Napi::AsyncWorker, so it runs on the libuv thread pool. That pool is
// shared with fs, dns, zlib and crypto and defaults to four threads, so a run occupies one slot for
// its duration; unrelated work only queues once concurrent runs reach the pool size, and
// UV_THREADPOOL_SIZE raises that ceiling.
class InferenceSessionWrap::RunAsyncWorker : public Napi::AsyncWorker {
 public:
  using OutputBufferLease = OrtInstanceData::OutputBufferLease;

  // Everything the completion callback needs to know about one fetched output. Holding it in a
  // single structure keeps a null fetch to one default-constructed element instead of placeholder
  // entries pushed into several vectors that must stay in lockstep.
  struct OutputBinding {
    std::string name;
    // The caller supplied a tensor for this output rather than asking ORT to allocate one.
    bool reuse{false};
    // The caller's tensor is CPU-backed, so ORT allocates and OnOK() copies into it.
    bool copy_to_js{false};
    // Type and shape the caller's tensor declares, checked against what the model produced.
    PreallocatedOutputInfo declared;
    // The caller's Tensor, and the storage behind it: the typed array for a CPU output (which is
    // also the copy destination) or the gpu-buffer External for a device one. The two never coexist,
    // and the lease key is derived from the storage, so no third reference is needed.
    //
    // Held as persistent references rather than slots in a Javascript array: array element
    // assignment can be intercepted by an inherited setter (Array.prototype[0]), leaving the value
    // unpinned and letting a matching getter return a different object on every read, so validation
    // and the copy could see different buffers.
    Napi::Reference<Napi::Value> js_value;
    Napi::Reference<Napi::Value> js_storage;
    size_t lease_byte_offset{0};
    size_t lease_byte_length{0};
    // Device outputs lease the whole External; they have no addressable sub-range.
    bool lease_whole_resource{false};
  };

  RunAsyncWorker(InferenceSessionWrap& session, const Napi::Object& feed, const Napi::Object& fetch,
                 const Napi::Object& options, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(session.Value().Env(), "InferenceSession.run", session.Value()),
        env_(session.Value().Env()),
        session_(&session),
        deferred_(deferred),
        session_reference_(Napi::Persistent(session.Value())),
        cpu_memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        gpu_buffer_memory_info_("WebGPU_Buf", OrtDeviceAllocator, 0, OrtMemTypeDefault) {
    // Disarmed once construction succeeds; until then it hands back anything device-backed that has
    // already been built, since a throwing constructor never runs its own destructor.
    struct FailureGuard {
      RunAsyncWorker* worker;
      const InferenceSessionWrap* owner;
      bool armed{true};
      ~FailureGuard() {
        if (armed) {
          worker->ReleaseDeviceValuesOnFailure(owner->session_);
        }
      }
    } failure_guard{this, &session};

    for (const auto& name : session.inputNames_) {
      if (feed.Has(name)) {
        auto value = feed.Get(name);
        input_names_.push_back(name);
        // No conversion detail is needed for inputs: cpu data is copied into ORT-owned storage and
        // gpu-buffer storage is held for the run by gpu_value_owners_, so there is nothing to pin.
        input_values_.push_back(NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_,
                                                    NapiValueUsage::kInput, &gpu_value_owners_));
      }
    }

    for (const auto& name : session.outputNames_) {
      if (fetch.Has(name)) {
        auto value = fetch.Get(name);
        OutputBinding output;
        output.name = name;
        output.reuse = !value.IsNull();

        if (!output.reuse) {
          output_values_.emplace_back(nullptr);
        } else {
          NapiTensorConversion conversion;
          auto output_value = NapiValueToOrtValue(env_, value, cpu_memory_info_, gpu_buffer_memory_info_,
                                                  NapiValueUsage::kPreallocatedOutput, &gpu_value_owners_,
                                                  &conversion);
          // CPU outputs come back empty: ORT allocates them and OnOK() validates the result against
          // the declared type and shape before copying it into the caller's buffer. Device outputs
          // carry the caller's memory and are bound directly, so ORT writes into it and no copy is
          // needed.
          output.copy_to_js = static_cast<OrtValue*>(output_value) == nullptr;

          // A device output is only ever written by binding it. On the plain Run() path it would go
          // to ORT as a preallocated fetch, which ORT is free to replace rather than fill -- the
          // caller's buffer would then keep its previous contents and the promise would still
          // resolve. There is no copy-back for device memory to fall back on, so refuse instead.
          ORT_NAPI_THROW_ERROR_IF(!output.copy_to_js && session.preferredOutputLocations_.empty(), env_,
                                  "Preallocated output '", name,
                                  "' is on a device, which requires the session to be created with "
                                  "'preferredOutputLocation'.");

          output_values_.push_back(std::move(output_value));

          output.js_value = Napi::Persistent(value);
          if (!conversion.data.IsEmpty()) {
            output.js_storage = Napi::Persistent(conversion.data);
          } else if (!conversion.gpuBuffer.IsEmpty()) {
            output.js_storage = Napi::Persistent(conversion.gpuBuffer);
          } else {
            output.js_storage = Napi::Persistent(value);
          }
          output.declared = std::move(conversion.declared);
          output.lease_byte_offset = conversion.dataByteOffset;
          output.lease_byte_length = conversion.dataByteLength;
          output.lease_whole_resource = conversion.dataArrayBuffer.IsEmpty();
        }
        outputs_.push_back(std::move(output));
      }
    }

    preferred_output_locations_ = session.preferredOutputLocations_;
    ParseRunOptions(options, run_options_);
    failure_guard.armed = false;
  }

  // Hand over anything device-backed that was built before a failure. A constructor that throws
  // never runs its own destructor, so its members would otherwise be destroyed inline on the
  // Javascript thread, outside the device lock every other release path goes through.
  void ReleaseDeviceValuesOnFailure(const std::shared_ptr<Ort::Session>& owning_session) {
    OrtInstanceData::ReleaseDeviceObject(std::make_shared<DeviceValues>(std::move(output_values_), owning_session));
    OrtInstanceData::ReleaseDeviceObject(std::make_shared<DeviceValues>(std::move(input_values_), owning_session));
    OrtInstanceData::ReleaseDeviceObject(std::make_shared<std::vector<OrtValueOwner>>(std::move(gpu_value_owners_)));
  }

  ~RunAsyncWorker() override {
    // AsyncWorker::OnWorkComplete() skips OnOK() and OnError() entirely when the work was cancelled
    // and still destroys the worker, so drain the run count here and settle the promise rather than
    // leaving it pending forever.
    Complete();
    if (!settled_) {
      settled_ = true;
      try {
        deferred_.Reject(Napi::Error::New(env_, "Inference was abandoned before it completed.").Value());
      } catch (...) {
        // The environment is going away; there is nobody left to observe the rejection.
      }
    }
  }

  // Settle the promise for a run that failed before it was queued. Routed through the worker so the
  // deferred is settled exactly once: settling it twice frees the napi_deferred twice.
  void Fail(Napi::Value error) {
    Complete();
    Settle([&] { deferred_.Reject(error); });
  }


  void AcquireOutputBufferLeases() {
    for (const auto& output : outputs_) {
      if (output.reuse) {
        // A CPU output leases a range of the ArrayBuffer behind its typed array. Reading that goes
        // through N-API's internal slots, so unlike a property read it cannot be intercepted.
        auto storage = output.js_storage.Value();
        auto resource = output.lease_whole_resource ? storage.As<Napi::Object>()
                                                    : storage.As<Napi::TypedArray>().ArrayBuffer();
        output_buffer_leases_.push_back(OrtInstanceData::AcquireOutputBufferLease(
            resource, output.lease_byte_offset, output.lease_byte_length, output.lease_whole_resource));
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
      output_names_cstr.reserve(outputs_.size());
      for (const auto& output : outputs_) {
        output_names_cstr.push_back(output.name.c_str());
      }

      // A provider with global device state cannot have two runs in flight anywhere in the process,
      // and binding does device work before Run() that ORT's own guard does not cover either.
      // Sessions without such a provider never take the lock and stay fully concurrent.
      // Destroyed after device_lock releases, so anything queued in the window between the final
      // drain and the unlock gets another chance rather than waiting for the next device run.
      struct PostUnlockDrain {
        ~PostUnlockDrain() { OrtInstanceData::TryDrainDeviceReleases(); }
      } post_unlock_drain;

      std::unique_lock<std::mutex> device_lock;
      if (session_->requires_device_serialization_) {
        device_lock = std::unique_lock<std::mutex>(OrtInstanceData::DeviceMutex());
        OrtInstanceData::DrainDeviceReleasesLocked();
      }

      // Whatever happens below, drop everything bound to the device before leaving the lock. Doing
      // it from Complete() instead would run on the Javascript thread while another session binds.
      struct DeviceCleanup {
        RunAsyncWorker* worker;
        std::unique_lock<std::mutex>* lock;
        ~DeviceCleanup() {
          worker->io_binding_.reset();
          worker->input_values_.clear();
          worker->gpu_value_owners_.clear();
          if (lock->owns_lock()) {
            OrtInstanceData::DrainDeviceReleasesLocked();
          }
        }
      } device_cleanup{this, &device_lock};

      if (preferred_output_locations_.empty()) {
        session_->session_->Run(run_options_, input_names_cstr.data(), input_values_.data(), input_values_.size(),
                                output_names_cstr.data(), output_values_.data(), output_values_.size());
        return;
      }

      io_binding_ = std::make_unique<Ort::IoBinding>(*session_->session_);
      for (size_t i = 0; i < input_names_.size(); ++i) {
        io_binding_->BindInput(input_names_cstr[i], input_values_[i]);
      }

      for (size_t i = 0; i < outputs_.size(); ++i) {
        if (outputs_[i].reuse) {
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
      if (output_values_.size() != outputs_.size()) {
        SetError("Output count mismatch.");
      }
    } catch (const std::exception& e) {
      SetError(e.what());
    } catch (...) {
      SetError("Unknown error while running the model.");
    }
  }

  void OnError(const Napi::Error& error) override {
    Complete();
    Settle([&] { deferred_.Reject(error.Value()); });
  }

  void OnOK() override {
    Napi::HandleScope scope(env_);
    try {
      // Build the result first. OrtValueToNapiValue() runs the Javascript Tensor constructor and can
      // throw, and no caller-owned buffer may be written before the last step that can fail or hand
      // control back to Javascript.
      auto result = Napi::Object::New(env_);
      for (size_t i = 0; i < output_values_.size(); ++i) {
        const auto& output = outputs_[i];
        auto value = output.reuse ? output.js_value.Value()
                                  : OrtValueToNapiValue(env_, std::move(output_values_[i]), session_->session_);
        // Define the property rather than assigning it: assignment would run an inherited setter for
        // the output name (Object.prototype.<name>), which is user Javascript that could both
        // swallow the output and detach a preallocated buffer.
        result.DefineProperty(Napi::PropertyDescriptor::Value(
            output.name, value,
            static_cast<napi_property_attributes>(napi_writable | napi_enumerable | napi_configurable)));
      }

      // Validate every preallocated destination, then write them. Nothing between these two loops
      // may run Javascript: a buffer detached after its check would be memcpy'd through a dead
      // pointer, and a failure partway through the writes would reject with some caller buffers
      // already holding this run's results.
      for (size_t i = 0; i < output_values_.size(); ++i) {
        const auto& output = outputs_[i];
        if (output.copy_to_js) {
          ValidateOrtValueForNapiTypedArray(env_, output_values_[i], output.js_storage.Value(), output.declared);
        } else if (output.reuse) {
          // A device output is handed straight back to the caller, so nothing would otherwise
          // notice that the model produced a different type or shape than the tensor declares.
          // This only bites on the IO-binding path, where output_values_ has been replaced by
          // GetOutputValues(); on the plain Run() path it still holds the wrapper built from the
          // caller's own declaration, so the comparison there is trivially satisfied.
          ValidateOrtValueMatchesDeclared(env_, output_values_[i], output.declared);
        }
      }

      for (size_t i = 0; i < output_values_.size(); ++i) {
        if (outputs_[i].copy_to_js) {
          CopyOrtValueToNapiTypedArray(env_, output_values_[i], outputs_[i].js_storage.Value(), outputs_[i].declared);
        }
      }
      Complete();
      Settle([&] { deferred_.Resolve(result); });
    } catch (const Napi::Error& e) {
      Complete();
      Settle([&] { deferred_.Reject(e.Value()); });
    } catch (const std::exception& e) {
      Complete();
      Settle([&] { deferred_.Reject(Napi::Error::New(env_, e.what()).Value()); });
    } catch (...) {
      Complete();
      Settle([&] { deferred_.Reject(Napi::Error::New(env_, "Unknown error while converting model outputs.").Value()); });
    }
  }

 private:

  template <typename Fn>
  void Settle(Fn&& settle) {
    if (settled_) {
      return;
    }
    settled_ = true;
    settle();
  }

  void Complete() {
    if (completed_) {
      return;
    }
    completed_ = true;

    ReleaseOutputBufferLeases();

    // Ort::IoBinding references the session, so it must go before EndRun() may tear the session
    // down; Execute() has already dropped it under the device lock on every path that created one.
    io_binding_.reset();

    // The remaining values may be device-backed, and releasing those is device work. Hand them over
    // rather than destroying them here: this runs on the Javascript thread, where taking the device
    // lock would stall the event loop for the length of another session's inference. Their buffers
    // keep their allocators alive through a shared_ptr, so they do not depend on the session.
    OrtInstanceData::ReleaseDeviceObject(
        std::make_shared<DeviceValues>(std::move(output_values_), session_->session_));
    OrtInstanceData::ReleaseDeviceObject(
        std::make_shared<DeviceValues>(std::move(input_values_), session_->session_));
    OrtInstanceData::ReleaseDeviceObject(std::make_shared<std::vector<OrtValueOwner>>(std::move(gpu_value_owners_)));

    session_->EndRun();
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
  Ort::MemoryInfo cpu_memory_info_;
  Ort::MemoryInfo gpu_buffer_memory_info_;
  Ort::RunOptions run_options_;
  std::vector<OrtValueOwner> gpu_value_owners_;
  std::vector<std::string> input_names_;
  std::vector<Ort::Value> input_values_;
  std::vector<OutputBinding> outputs_;
  // Parallel to outputs_. Kept separate because Ort::Session::Run() and Ort::IoBinding both want a
  // contiguous array of Ort::Value, and IoBinding replaces the whole vector with its results.
  std::vector<Ort::Value> output_values_;
  std::vector<OutputBufferLease> output_buffer_leases_;
  std::vector<int> preferred_output_locations_;
  std::unique_ptr<Ort::IoBinding> io_binding_;
  bool completed_{false};
  bool settled_{false};
};

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);

  // run() is declared as returning a Promise, so state and argument problems are reported by
  // rejecting it rather than by raising synchronously.
  const auto reject = [&deferred](Napi::Value error) {
    deferred.Reject(error);
    return deferred.Promise();
  };
  if (!this->initialized_) {
    return reject(Napi::Error::New(env, "Session is not initialized.").Value());
  }
  if (this->disposed_) {
    return reject(Napi::Error::New(env, "Session already disposed.").Value());
  }
  if (info.Length() < 2) {
    return reject(Napi::TypeError::New(env, "Expect argument: inputs(feed) and outputs(fetch).").Value());
  }
  if (!info[0].IsObject() || !info[1].IsObject()) {
    return reject(Napi::TypeError::New(env, "Expect inputs(feed) and outputs(fetch) to be objects.").Value());
  }
  if (info.Length() > 2 && (!info[2].IsObject() || info[2].IsNull())) {
    return reject(Napi::TypeError::New(env, "'runOptions' must be an object.").Value());
  }

  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();
  auto options = info.Length() > 2 ? info[2].As<Napi::Object>() : Napi::Object::New(env);

  std::unique_ptr<RunAsyncWorker> worker;

  // Register the run before reading 'feed' and 'fetch': those reads can re-enter JS through
  // getters or Proxy traps, and a reentrant dispose() must not be allowed to tear the session
  // down while this run is still being prepared.
  BeginRun();
  try {
    worker = std::make_unique<RunAsyncWorker>(*this, feed, fetch, options, deferred);
    worker->AcquireOutputBufferLeases();
    worker->Queue();
  } catch (const Napi::Error& e) {
    // Reject rather than throw: the deferred already exists, and N-API frees it only once it has
    // been settled. Throwing would leak it and would also make a method that is typed as returning
    // a Promise raise synchronously.
    FailRun(worker, deferred, e.Value());
    return deferred.Promise();
  } catch (const std::exception& e) {
    FailRun(worker, deferred, Napi::Error::New(env, e.what()).Value());
    return deferred.Promise();
  } catch (...) {
    FailRun(worker, deferred, Napi::Error::New(env, "Unknown error while preparing inference.").Value());
    return deferred.Promise();
  }

  worker.release();
  return deferred.Promise();
}

void InferenceSessionWrap::BeginRun() {
  ++active_runs_;
}

void InferenceSessionWrap::FailRun(const std::unique_ptr<RunAsyncWorker>& worker,
                                   Napi::Promise::Deferred& deferred, Napi::Value error) {
  // Once the worker exists it owns both the run registration and the promise, so let it settle them;
  // its destructor would otherwise reject a promise this function had already rejected.
  if (worker) {
    worker->Fail(error);
  } else {
    EndRun();
    deferred.Reject(error);
  }
}

void InferenceSessionWrap::EndRun() {
  if (--active_runs_ == 0 && teardown_pending_) {
    teardown_pending_ = false;
    TeardownSession();
  }
}

void InferenceSessionWrap::TeardownSession() {
  inputTypes_.clear();
  outputTypes_.clear();
  session_.reset();
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  // Refuse further calls straight away, but keep the ORT objects alive for runs already in flight:
  // they hold a raw pointer to this session and execute on the libuv threadpool. EndRun() performs
  // the teardown once the last one completes.
  this->disposed_ = true;
  if (this->active_runs_ != 0) {
    this->teardown_pending_ = true;
    return env.Undefined();
  }

  this->TeardownSession();
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::EndProfiling(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  // Not deferrable and not safe to overlap with a run: Profiler::EndTimeAndRecordEvent() calls each
  // execution provider profiler's Stop() outside Profiler::mutex_, while EndProfiling() calls the
  // same object's EndProfiling() under it, and 'enabled_' is a plain bool. Racing them is undefined.
  ORT_NAPI_THROW_ERROR_IF(this->active_runs_ != 0, env, "Cannot end profiling while inference is running.");

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
