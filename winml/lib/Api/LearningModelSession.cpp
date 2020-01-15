// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "LearningModelSession.h"

#include "ImageFeatureDescriptor.h"
#include "LearningModel.h"
#include "LearningModelBinding.h"
#include "LearningModelEvaluationResult.h"
#include "LearningModelDevice.h"
#include "LearningModelSessionOptions.h"
#include "TensorFeatureDescriptor.h"
#include "TelemetryEvent.h"

#include "D3DDeviceCache.h"

static const auto c_enable_debug_output = L"EnableDebugOutput";

namespace guid_details {
// This GUID is to be used for delimiting ML-related categories of capturable work.
// {D113B493-BBA2-4993-8608-D706A73B91CE}
struct __declspec(uuid("D113B493-BBA2-4993-8608-D706A73B91CE")) __declspec(novtable) WINML_PIX_EVAL_CAPTURABLE_WORK_GUID {};
}  // namespace guid_details
static const GUID WINML_PIX_EVAL_CAPTURABLE_WORK_GUID = __uuidof(guid_details::WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);

namespace winrt::Windows::AI::MachineLearning::implementation {

LearningModelSession::LearningModelSession(
  winml::LearningModel const& model) try : LearningModelSession(model,
                                                                make<LearningModelDevice>(LearningModelDeviceKind::Default)) {}
WINML_CATCH_ALL

LearningModelSession::LearningModelSession(
  winml::LearningModel const& model,
  winml::LearningModelDevice const& deviceToRunOn) try : LearningModelSession(model,
                                                                              deviceToRunOn,
                                                                              nullptr) {}
WINML_CATCH_ALL

LearningModelSession::LearningModelSession(
  winml::LearningModel const& model,
  winml::LearningModelDevice const& deviceToRunOn,
  winml::LearningModelSessionOptions const& learningModelSessionOptions) try : model_(model),
                                                                                device_(deviceToRunOn),
                                                                                session_options_(learningModelSessionOptions) {
  Initialize();
}
WINML_CATCH_ALL

winmla::IModelProto*
LearningModelSession::GetOptimizedModel() {
  // Get the model proto

  auto should_close_model =
    session_options_ != nullptr &&
    session_options_.CloseModelOnSessionCreation();

  return GetOptimizedModel(should_close_model);
}

winmla::IModelProto*
LearningModelSession::GetOptimizedModel(bool should_close_model) {
  com_ptr<winmla::IModelProto> model_proto;

  {
    // Lock the model detach/copy since multiple threads can access concurrently
    CWinMLAutoLock lock(&session_creation_lock_);

    // Throw if the model has been disposed and is not capable of creating
    // new sessions.
    auto model = model_.as<winmlp::LearningModel>();
    WINML_THROW_HR_IF_TRUE_MSG(E_INVALIDARG, model->IsDisposed(),
                               "The model has been disposed.");

    model_proto.attach(should_close_model
                      ? model->DetachModelProto()
                      : model->CopyModelProto());
  }

  // Ensure that the model is runnable on the device
  com_ptr<winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
  WINML_THROW_IF_FAILED(adapter->EnsureModelDeviceCompatibility(model_, model_proto.get(), device_.as<winmlp::LearningModelDevice>()->GetD3DDeviceCache()->IsFloat16Supported()));

  return model_proto.detach();
}

void LearningModelSession::Initialize() {
  // Begin recording session creation telemetry
  _winmlt::TelemetryEvent session_creation_event(
    _winmlt::EventCategory::kSessionCreation);
  // Get the optimized model proto from the learning model
  com_ptr<winmla::IModelProto> model_proto; 
  model_proto.attach(GetOptimizedModel());

  // Create the session builder
  auto device_impl = device_.as<winmlp::LearningModelDevice>();

  com_ptr<winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));

  com_ptr<winmla::IOrtSessionBuilder> session_builder;
  WINML_THROW_IF_FAILED(adapter->CreateOrtSessionBuilder(
    device_impl->GetD3DDevice(), 
    device_impl->GetDeviceQueue(),
    session_builder.put()));

  Ort::SessionOptions options(nullptr);
  WINML_THROW_IF_FAILED(session_builder->CreateSessionOptions(options.put()));

  // Make onnxruntime apply the batch size override, if any
  if (session_options_ && session_options_.BatchSizeOverride() != 0)
  {
    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverride(
      options, 
      onnx::DATA_BATCH, 
      session_options_.BatchSizeOverride()));
  }

  com_ptr<winmla::IInferenceSession> session;
  WINML_THROW_IF_FAILED(session_builder->CreateSession(
      options, session.put(), &cached_execution_provider_));

  // Register the custom operator registry
  auto model = model_.as<winmlp::LearningModel>();
  operatorRegistry_.reset(model->GetOperatorRegistry());
  WINML_THROW_IF_FAILED(session->RegisterCustomRegistry(operatorRegistry_.get()));

  // Register only the transformers not already in ORT
  session->RegisterGraphTransformers();

  // Load the model into the session
  WINML_THROW_IF_FAILED(session->LoadModel(model_proto.get()));
  // the session owns the model_proto now, it used detach()
  model_proto = nullptr;

  // Initialize the session
  WINML_THROW_IF_FAILED(session_builder->Initialize(session.get(), cached_execution_provider_));

  // Cache the constructed session
  inference_session_ = session;
}

wfc::IPropertySet
LearningModelSession::EvaluationProperties() try {
  if (evaluation_properties_ == nullptr) {
    evaluation_properties_ = wfc::PropertySet();
  }
  return evaluation_properties_;
}
WINML_CATCH_ALL

winml::LearningModel
LearningModelSession::Model() try {
  return model_;
}
WINML_CATCH_ALL

winml::LearningModelDevice
LearningModelSession::Device() try {
  return device_;
}
WINML_CATCH_ALL

auto CreateBinding(
  LearningModelSession& session,
  wfc::IMap<hstring, wf::IInspectable> const features) {
  auto binding = winrt::make<LearningModelBinding>(session);

  for (auto feature : features.GetView()) {
    binding.Bind(feature.Key(), feature.Value());
  }
  return binding;
}

winml::LearningModelEvaluationResult
LearningModelSession::EvaluateFeatures(
  wfc::IMap<hstring, wf::IInspectable> const features,
  hstring const correlation_id) try {
  auto binding = CreateBinding(*this, features);
  return Evaluate(binding, correlation_id);
}
WINML_CATCH_ALL

wf::IAsyncOperation<winml::LearningModelEvaluationResult>
LearningModelSession::EvaluateFeaturesAsync(
  wfc::IMap<hstring, wf::IInspectable> const features,
  hstring const correlation_id) {
  auto binding = CreateBinding(*this, features);
  return EvaluateAsync(binding, correlation_id);
}

// copied from onnxruntime_cxx_inline.h
inline OrtStatus* OrtRun(
    OrtSession * session, 
    const Ort::RunOptions& run_options, 
    const char* const* input_names, 
    const Ort::Value* input_values, 
    size_t input_count,
    const char* const* output_names, 
    Ort::Value* output_values, 
    size_t output_count) {
  static_assert(sizeof(Ort::Value) == sizeof(OrtValue*), "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<const OrtValue**>(const_cast<Ort::Value*>(input_values));
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  return Ort::GetApi().Run(session, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values);
}

uint64_t
LearningModelSession::Run(
    winrt::com_ptr<winmlp::LearningModelBinding> binding_impl) {
  CheckClosed();
  auto device = device_.as<LearningModelDevice>();
  CWinMLAutoLock lock(!device->IsCpuDevice() ? &evaluate_lock_ : nullptr);
  // TODO : set the run_options
  Ort::RunOptions run_options;
  binding_impl->BindUnboundOutputs();

  std::vector<const char*> inputNames_c;
  for (int i=0; i < binding_impl->GetInputNames().size(); i++)
  {
    inputNames_c.push_back(binding_impl->GetInputNames()[i].c_str());
  }
  std::vector<const char*> outputNames_c;
  for (int i = 0; i < binding_impl->GetOutputNames().size(); i++) {
    outputNames_c.push_back(binding_impl->GetOutputNames()[i].c_str());
  }
  OrtSession* session = nullptr;

  WINML_THROW_IF_FAILED(inference_session_->GetOrtSession(&session));
  // Invoke run on the ORT session.
  Ort::ThrowOnError(OrtRun(
    session, 
    run_options, 
    inputNames_c.data(),
    binding_impl->GetInputs().data(),
    binding_impl->GetInputs().size(),
    outputNames_c.data(),
    binding_impl->GetOutputs().data(),
    binding_impl->GetOutputs().size()));

  if (!device->IsCpuDevice()) {
    // Flush the D3D12 work from the DML execution provider and queue a fence before we release the lock.
    // This allows us to wait without holding onto the lock in GetResults.
    inference_session_->FlushContext(GetExecutionProvider());
    return device->GetD3DDeviceCache()->QueueFenceToD3D12();
  }

  // If it's the cpu then just return zero. fence value will be unused.
  return 0;
}

winml::LearningModelEvaluationResult
LearningModelSession::GetResults(
  winrt::com_ptr<winmlp::LearningModelBinding> binding_impl,
  hstring const& correlation_id,
  uint64_t evaluation_complete_fence) {
  // First wait on the fence value for the expected frame. This is passed in so that
  // the fence value is added to the queue in a thread safe manor.
  auto device = device_.as<winmlp::LearningModelDevice>();
  auto is_gpu_evaluation = !device->IsCpuDevice();

  if (is_gpu_evaluation) {
    device->GetD3DDeviceCache()->WaitForFenceValue(evaluation_complete_fence);
  }

  CWinMLAutoLock lock(is_gpu_evaluation ? &evaluate_lock_ : nullptr);

  if (is_gpu_evaluation) {
    // For DML we aren't using the Sync function because we want to make fencing the
    // completed frame thread safe while not holding the lock while waiting for the gpu.
    inference_session_->ReleaseCompletedReferences(GetExecutionProvider());
  } else {
    // For CPU call the standard Sync function
    GetExecutionProvider()->Sync();
  }

  // This isn't the best we are holding the lock while we wait for detensorize on the GPU.
  // Update output providers
  auto outputs = binding_impl->UpdateProviders();

  // Once the first evaluation following initialization is complete, and therefore the
  // initialization work is also complete, trim the upload heap. This is only done once
  // to avoid requiring the extra allocation during each evaluation.
  if (is_first_evaluate_) {
    if (is_gpu_evaluation) {
      inference_session_->TrimUploadHeap(GetExecutionProvider());
    }
    is_first_evaluate_ = false;
  }

  // Create the return status object
  auto result = winrt::make<LearningModelEvaluationResult>();
  auto result_impl = result.as<winmlp::LearningModelEvaluationResult>();
  result_impl->Succeeded(true);
  result_impl->ErrorStatus(0);
  result_impl->CorrelationId(correlation_id);
  result_impl->SetOutputs(std::move(outputs));

  return result;
}

wf::IAsyncOperation<winml::LearningModelEvaluationResult>
LearningModelSession::EvaluateAsync(
    winml::LearningModelBinding binding,
    hstring const correlation_id) {
  _winmlt::TelemetryEvent kEvaluateModel_event(_winmlt::EventCategory::kEvaluation);
  auto device = device_.as<LearningModelDevice>();

  // Get the ORT binding collection
  auto binding_impl = binding.as<winmlp::LearningModelBinding>();

  ApplyEvaluationProperties();

  // If we're running on the CPU, then return now and process the rest in the background.
  // If we're running on the GPU, then queue up the work first (fast) and wait for the
  // results (slow) in the background.
  bool should_queue_work = (!device->IsCpuDevice());
  if (!should_queue_work) {
    co_await resume_background();
  }

  com_ptr<ID3D12CommandQueue> queue;
  queue.copy_from(device->GetDeviceQueue());
  com_ptr<ID3D12SharingContract> capture_interface = queue.try_as<ID3D12SharingContract>();

  // markers for PIX debugging
  if (capture_interface != nullptr) {
    capture_interface->BeginCapturableWork(WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);
  }

  // call Run synchronously on the calling thread to queue up the work
  uint64_t evaluation_complete_fence = Run(binding_impl);

  // markers for PIX debugging
  if (capture_interface) {
    capture_interface->EndCapturableWork(WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);
  }

  // after the work is queued, return to the caller
  if (should_queue_work) {
    // Queue detensorization
    co_await resume_background();
  }

  // Get the Results on a background thread whenever they're ready
  return GetResults(binding_impl, correlation_id, evaluation_complete_fence);
}

winml::LearningModelEvaluationResult
LearningModelSession::Evaluate(
    winml::LearningModelBinding binding,
    hstring const& correlation_id) try {
  ToggleProfiler();
  _winmlt::TelemetryEvent kEvaluateModel_event(_winmlt::EventCategory::kEvaluation);

  ApplyEvaluationProperties();

  auto device = device_.as<LearningModelDevice>();

  com_ptr<ID3D12CommandQueue> queue;
  queue.copy_from(device->GetDeviceQueue());
  com_ptr<ID3D12SharingContract> capture_interface = queue.try_as<ID3D12SharingContract>();

  // markers for PIX debugging
  if (capture_interface != nullptr) {
    capture_interface->BeginCapturableWork(WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);
  }

  // Get the ORT binding collection
  auto binding_impl = binding.as<implementation::LearningModelBinding>();
  uint64_t evaluation_complete_fence = Run(binding_impl);

  // markers for PIX debugging
  if (capture_interface) {
    capture_interface->EndCapturableWork(WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);
  }

  return GetResults(binding_impl, correlation_id, evaluation_complete_fence);
}
WINML_CATCH_ALL

void LearningModelSession::Close() {
    inference_session_ = nullptr;
}

void LearningModelSession::ApplyEvaluationProperties() try {
  if (evaluation_properties_) {
    auto is_debug_output_enabled = evaluation_properties_.HasKey(c_enable_debug_output);
    if (is_debug_output_enabled) {
      com_ptr<winmla::IWinMLAdapter> adapter;
      WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
      adapter->EnableDebugOutput();
    }
  }
}
WINML_CATCH_ALL

void LearningModelSession::ToggleProfiler() {
  CheckClosed();
  auto is_provider_enabled =
      TraceLoggingProviderEnabled(
          ::winml_trace_logging_provider,
          WINEVENT_LEVEL_VERBOSE,
          WINML_PROVIDER_KEYWORD_LOTUS_PROFILING);

  if (is_provider_enabled) {
    inference_session_->StartProfiling();
  } else {
    inference_session_->EndProfiling();
  }
}

onnxruntime::IExecutionProvider*
LearningModelSession::GetExecutionProvider() {
  return cached_execution_provider_;
}

winmla::IInferenceSession*
LearningModelSession::GetIInferenceSession() {
  return inference_session_.get();
}

void LearningModelSession::CheckClosed() {
  if (!inference_session_) {
    WINML_THROW_HR(RO_E_CLOSED);
  }
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation