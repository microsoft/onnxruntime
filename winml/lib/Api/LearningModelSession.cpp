// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "LearningModelSession.h"

#include "CustomRegistryHelper.h"
#include "ImageFeatureDescriptor.h"
#include "IOrtSessionBuilder.h"
#include "LearningModel.h"
#include "LearningModelBinding.h"
#include "LearningModelEvaluationResult.h"
#include "LearningModelDevice.h"
#include "LearningModelSessionOptions.h"
#include "TensorFeatureDescriptor.h"
#include "TelemetryEvent.h"

#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/customRegistry.h"

#include "D3DDeviceCache.h"

#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"

#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "LotusEnvironment.h"
#include "PheonixSingleton.h"

static const auto c_enable_debug_output = L"EnableDebugOutput";

namespace guid_details {
// This GUID is to be used for delimiting ML-related categories of capturable work.
// {D113B493-BBA2-4993-8608-D706A73B91CE}
struct __declspec(uuid("D113B493-BBA2-4993-8608-D706A73B91CE")) __declspec(novtable) WINML_PIX_EVAL_CAPTURABLE_WORK_GUID {};
}  // namespace guid_details
static const GUID WINML_PIX_EVAL_CAPTURABLE_WORK_GUID = __uuidof(guid_details::WINML_PIX_EVAL_CAPTURABLE_WORK_GUID);

namespace winrt::Windows::AI::MachineLearning::implementation {
// ORT intentionally requires callers derive from their session class to access
// the protected Load method used below.
class InferenceSessionProtectedLoadAccessor : public onnxruntime::InferenceSession {
 public:
  onnxruntime::common::Status
  Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) {
    return onnxruntime::InferenceSession::Load(std::move(p_model_proto));
  }
};

static bool
IsFeatureDescriptorFp16(
    winml::ILearningModelFeatureDescriptor descriptor) {
  if (auto imageFeatureDescriptor = descriptor.try_as<ImageFeatureDescriptor>()) {
    return TensorKind::Float16 == imageFeatureDescriptor->TensorKind();
  }

  if (auto tensorFeatureDescriptor = descriptor.try_as<TensorFeatureDescriptor>()) {
    return TensorKind::Float16 == tensorFeatureDescriptor->TensorKind();
  }

  return false;
}

static void
EnsureModelDeviceCompatibility(
    winml::LearningModel const& model,
    onnx::ModelProto* p_model_proto,
    winml::LearningModelDevice const& device) {
  auto isFloat16Supported = device.as<LearningModelDevice>()->GetD3DDeviceCache()->IsFloat16Supported();
  if (!isFloat16Supported) {
    auto& graph = p_model_proto->graph();

    // The model will not contain fp16 operations if:
    // 1. The model has no fp16 inputs
    // 2. The model has no fp16 initializers
    // 3. The model does not create any fp16 intermediary tensors via the Cast (to float16) operator
    // 4. The model does not have any fp16 outputs

    // 1. Ensure that The model has no fp16 inputs
    for (auto descriptor : model.InputFeatures()) {
      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          IsFeatureDescriptorFp16(descriptor),
          "The model contains a 16-bit input (%ls), but the current device does not support 16-bit float.",
          descriptor.Name().c_str());
    }

    // 2. Ensure that the model has no fp16 initializers
    for (int i = 0; i < graph.node_size(); i++) {
      auto node = graph.node(i);
      if (node.op_type() == "Cast" && node.domain().empty()) {
        for (int attribIndex = 0; attribIndex < node.attribute_size(); attribIndex++) {
          auto attribute = node.attribute(attribIndex);
          if (attribute.name() == "to") {
            WINML_THROW_HR_IF_TRUE_MSG(
                DXGI_ERROR_UNSUPPORTED,
                attribute.i() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
                "The model contains a 16-bit float Cast Op (%s), but the current device does not support 16-bit float.",
                node.name().c_str());
          }
        }
      }
    }

    // 3. Ensure that the model does not create any fp16 intermediary
    //    tensors via the Cast (to float16) operator
    for (int i = 0; i < graph.initializer_size(); i++) {
      auto initializer = graph.initializer(i);

      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          initializer.data_type() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
          "The model contains a 16-bit float initializer (%s), but the current device does not support 16-bit float.",
          initializer.name().c_str());
    }

    // 4. Ensure that the model does not have any fp16 outputs
    for (auto descriptor : model.OutputFeatures()) {
      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          IsFeatureDescriptorFp16(descriptor),
          "The model contains a 16-bit output (%ls), but the current device does not support 16-bit float.",
          descriptor.Name().c_str());
    }
  }
}

static HRESULT
RegisterCustomRegistry(
    onnxruntime::InferenceSession* p_session,
    IMLOperatorRegistry* registry) {
  RETURN_HR_IF(S_OK, registry == nullptr);
  RETURN_HR_IF_NULL(E_POINTER, p_session);

  auto custom_registries = WinML::GetLotusCustomRegistries(registry);

  // Register
  for (auto& custom_registry : custom_registries) {
    WINML_THROW_IF_NOT_OK(p_session->RegisterCustomRegistry(custom_registry));
  }

  return S_OK;
}

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

std::unique_ptr<onnx::ModelProto>
LearningModelSession::GetOptimizedModel() {
  // Get the model proto
  auto should_close_model =
      session_options_ != nullptr &&
      session_options_.CloseModelOnSessionCreation();

  return GetOptimizedModel(should_close_model);
}

std::unique_ptr<onnx::ModelProto>
LearningModelSession::GetOptimizedModel(bool should_close_model) {
  std::unique_ptr<onnx::ModelProto> model_proto;

  {
    // Lock the model detach/copy since multiple threads can access concurrently
    CWinMLAutoLock lock(&session_creation_lock_);

    // Throw if the model has been disposed and is not capable of creating
    // new sessions.
    auto model = model_.as<winmlp::LearningModel>();
    WINML_THROW_HR_IF_TRUE_MSG(E_INVALIDARG, model->IsDisposed(),
                               "The model has been disposed.");

    model_proto = should_close_model
                      ? model->DetachModelProto()
                      : model->CopyModelProto();
  }

  // Ensure that the model is runnable on the device
  EnsureModelDeviceCompatibility(model_, model_proto.get(), device_);

  return model_proto;
}

void LearningModelSession::Initialize() {
  // Begin recording session creation telemetry
  _winmlt::TelemetryEvent session_creation_event(
      _winmlt::EventCategory::kSessionCreation);

  // Get the optimized model proto from the learning model
  auto model_proto = GetOptimizedModel();

  // Create the session builder
  auto session_builder = WinML::CreateOrtSessionBuilder(device_);

  onnxruntime::SessionOptions options = {};
  WINML_THROW_IF_FAILED(session_builder->CreateSessionOptions(&options));
  options.graph_optimization_level = onnxruntime::TransformerLevel::Level3;

  // Make onnxruntime apply the batch size override, if any
  if (session_options_ && session_options_.BatchSizeOverride() != 0)
  {
    onnxruntime::FreeDimensionOverride overrideOption = {};
    overrideOption.dimension_denotation = onnx::DATA_BATCH;
    overrideOption.dimension_override = session_options_.BatchSizeOverride();
    options.free_dimension_overrides.emplace_back(overrideOption);
  }
  {
    onnxruntime::FreeDimensionOverride overrideOption = {};
    overrideOption.dimension_denotation = "inputWidth";
    overrideOption.dimension_override = 640;
    options.free_dimension_overrides.emplace_back(overrideOption);
  }
  {
    onnxruntime::FreeDimensionOverride overrideOption = {};
    overrideOption.dimension_denotation = "inputHeight";
    overrideOption.dimension_override = 576;
    options.free_dimension_overrides.emplace_back(overrideOption);
  }

  auto session = std::unique_ptr<onnxruntime::InferenceSession>();
  WINML_THROW_IF_FAILED(session_builder->CreateSession(
      options, &session, &p_cached_execution_provider));

  // Register the custom operator registry
  auto model = model_.as<winmlp::LearningModel>();
  RegisterCustomRegistry(session.get(), model->GetOperatorRegistry());

  // Register only the transformers not already in ORT
  const bool registerLotusTransformers = false;
  GraphTransformerHelpers::RegisterGraphTransformers(session.get(), registerLotusTransformers);

  // Load the model into the session
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(session.get());
  WINML_THROW_IF_NOT_OK(session_protected_load_accessor->Load(std::move(model_proto)));

  // Initialize the session
  session_builder->Initialize(session.get(), p_cached_execution_provider);

  // Cache the constructed session
  inference_session_ = std::move(session);

  auto device_impl = device_.as<winmlp::LearningModelDevice>();
  telemetry_helper.LogSessionCreation(
      WinML::Strings::UTF8FromHString(model_.Name()),
      device_impl->IsCpuDevice(),
      device_impl->GetDeviceLuid());
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

static onnxruntime::IOBinding&
GetIOBinding(
    winrt::com_ptr<winmlp::LearningModelBinding> binding_impl,
    winml::LearningModel& model) {
  // Get the IOBinding Collection, and bound outputs
  auto& io_binding = binding_impl->BindingCollection();
  auto& bound_output_names = io_binding.GetOutputNames();
  std::unordered_set<std::string> bound_output_names_set(
      bound_output_names.begin(),
      bound_output_names.end());

  // Get model output feature names
  auto model_impl = model.as<winmlp::LearningModel>();
  auto output_features = model_impl->OutputFeatures();
  std::vector<ILearningModelFeatureDescriptor> output_descriptors(
      begin(output_features),
      end(output_features));

  // Convert all output features to their feature names
  std::vector<std::string> output_feature_names;
  std::transform(
      std::begin(output_descriptors),
      std::end(output_descriptors),
      std::back_inserter(output_feature_names),
      [&](auto& descriptor) {
        auto descriptor_native = descriptor.as<ILearningModelFeatureDescriptorNative>();
        const wchar_t* p_name;
        uint32_t size;
        WINML_THROW_IF_FAILED(descriptor_native->GetName(&p_name, &size));
        return WinML::Strings::UTF8FromUnicode(p_name, size);
      });

  // Find the set difference to determine if there are any unbound output features
  std::vector<std::string> unbound_output_names;
  std::copy_if(
      std::begin(output_feature_names), std::end(output_feature_names),
      std::inserter(unbound_output_names, std::begin(unbound_output_names)),
      [&](const auto& outputFeatureName) {
        return bound_output_names_set.find(outputFeatureName) == bound_output_names_set.end();
      });

  // Add all unbound outputs to the iobinding collection
  for (const auto& unbound_output : unbound_output_names) {
    OrtValue value = {};
    WINML_THROW_IF_NOT_OK(io_binding.BindOutput(unbound_output, value));
  }

  return io_binding;
}

uint64_t
LearningModelSession::Run(
    winrt::com_ptr<winmlp::LearningModelBinding> binding_impl) {
  CheckClosed();
  auto device = device_.as<LearningModelDevice>();
  CWinMLAutoLock lock(!device->IsCpuDevice() ? &evaluate_lock_ : nullptr);
  // TODO : set the run_options
  onnxruntime::RunOptions run_options;

  auto& io_binding = GetIOBinding(binding_impl, model_);

  // Invoke run on the ORT session.
  WINML_THROW_IF_NOT_OK(inference_session_->Run(run_options, io_binding));

  if (!device->IsCpuDevice()) {
    // Flush the D3D12 work from the DML execution provider and queue a fence before we release the lock.
    // This allows us to wait without holding onto the lock in GetResults.
    Dml::FlushContext(GetExecutionProvider());
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
    Dml::ReleaseCompletedReferences(GetExecutionProvider());
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
      Dml::TrimUploadHeap(GetExecutionProvider());
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
  _winmlt::PerformanceTelemetryEvent kEvaluateModel_event(WinMLRuntimePerf::kEvaluateModel);

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
  _winmlt::PerformanceTelemetryEvent kEvaluateModel_event(WinMLRuntimePerf::kEvaluateModel);

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
  inference_session_.reset();
}

std::unique_ptr<onnxruntime::IOBinding>
LearningModelSession::CreateSessionBinding() {
  CheckClosed();
  std::unique_ptr<onnxruntime::IOBinding> binding;
  WINML_THROW_IF_NOT_OK(inference_session_->NewIOBinding(&binding));
  return binding;
}

void LearningModelSession::ApplyEvaluationProperties() try {
  if (evaluation_properties_) {
    auto is_debug_output_enabled = evaluation_properties_.HasKey(c_enable_debug_output);
    if (is_debug_output_enabled) {
      WinML::CWinMLLogSink::EnableDebugOutput();
    }
  }
}
WINML_CATCH_ALL

void LearningModelSession::ToggleProfiler() {
  CheckClosed();
  auto is_provider_enabled =
      TraceLoggingProviderEnabled(
          winml_trace_logging_provider,
          WINEVENT_LEVEL_VERBOSE,
          WINML_PROVIDER_KEYWORD_LOTUS_PROFILING);

  if (is_provider_enabled) {
    inference_session_->StartProfiling(PheonixSingleton<WinML::LotusEnvironment>()->GetDefaultLogger());
  } else {
    inference_session_->EndProfiling();
  }
}

onnxruntime::IExecutionProvider*
LearningModelSession::GetExecutionProvider() {
  return p_cached_execution_provider;
}

void LearningModelSession::CheckClosed() {
  if (!inference_session_) {
    WINML_THROW_HR(RO_E_CLOSED);
  }
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation