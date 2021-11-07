// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Ort/pch.h"

#include "OnnxruntimeEngine.h"
#include "OnnxruntimeEngineBuilder.h"
#include "OnnxruntimeCpuSessionBuilder.h"
#include "OnnxruntimeErrors.h"

#ifdef USE_DML
#include "OnnxruntimeDmlSessionBuilder.h"
#endif

#ifdef USE_OPENVINO
#include "OnnxruntimeOpenVinoSessionBuilder.h"
struct __declspec(uuid("f292bc6f-0dd3-423c-bb72-3575222285e1")) OpenVinoProviderOptions;
#endif

#ifdef USE_TENSORRT
#include "OnnxruntimeTensorRTSessionBuilder.h"
struct __declspec(uuid("55411b0d-06d1-426d-94a1-cdd70802f1bd")) TensorRTProviderOptions;
#endif

#ifdef USE_CUDA
#include "OnnxruntimeCUDASessionBuilder.h"
struct __declspec(uuid("34001d56-7780-4aa9-9e37-da05d2ead6d2")) CUDAProviderOptions;
#endif

using namespace _winml;


HRESULT OnnxruntimeEngineBuilder::RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine_factory) {
  engine_factory_ = engine_factory;
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::CreateEngine(_Outptr_ _winml::IEngine** out) {
  auto ort_api = engine_factory_->UseOrtApi();

  Microsoft::WRL::ComPtr<IOrtSessionBuilder> onnxruntime_session_builder;
  do {
#ifdef USE_DML
    if (device_) {
      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeDmlSessionBuilder>(&onnxruntime_session_builder, engine_factory_.Get(), device_.Get(), queue_.Get(), metacommands_enabled_));
      break;
    }
#endif

#ifdef USE_OPENVINO
    Microsoft::WRL::ComPtr<IUnknown> open_vino_options;
    if (custom_execution_provider_options_ != nullptr && SUCCEEDED(custom_execution_provider_options_->QueryInterface(__uuidof(OpenVinoProviderOptions), &open_vino_options))) {
      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeOpenVinoSessionBuilder>(
          &onnxruntime_session_builder, engine_factory_.Get(), custom_execution_provider_options_.Get()));
      break;
    }
#endif

#ifdef USE_TENSORRT
    Microsoft::WRL::ComPtr<IUnknown> tensorrt_options;
    if (custom_execution_provider_options_ != nullptr && SUCCEEDED(custom_execution_provider_options_->QueryInterface(__uuidof(TensorRTProviderOptions), &tensorrt_options))) {
      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeTensorRTSessionBuilder>(
          &onnxruntime_session_builder, engine_factory_.Get(), custom_execution_provider_options_.Get()));
      break;
    }
#endif

#ifdef USE_CUDA
    Microsoft::WRL::ComPtr<IUnknown> cuda_options;
    if (custom_execution_provider_options_ != nullptr && SUCCEEDED(custom_execution_provider_options_->QueryInterface(__uuidof(CUDAProviderOptions), &cuda_options))) {
      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeCUDASessionBuilder>(
          &onnxruntime_session_builder, engine_factory_.Get(), custom_execution_provider_options_.Get()));
      break;
    }
#endif

    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeCpuSessionBuilder>(&onnxruntime_session_builder, engine_factory_.Get()));
  } while (false);

  OrtSessionOptions* ort_options;
  RETURN_IF_FAILED(onnxruntime_session_builder->CreateSessionOptions(&ort_options));
  auto session_options = UniqueOrtSessionOptions(ort_options, ort_api->ReleaseSessionOptions);

  if (batch_size_override_.has_value()) {
    constexpr const char* DATA_BATCH = "DATA_BATCH";
    RETURN_HR_IF_NOT_OK_MSG(ort_api->AddFreeDimensionOverride(session_options.get(), DATA_BATCH, batch_size_override_.value()),
                            ort_api);
  }
  if (named_dimension_overrides_) {
    for (const auto& override : named_dimension_overrides_) {
      std::string narrow_name = _winml::Strings::UTF8FromHString(override.Key());
      ort_api->AddFreeDimensionOverrideByName(session_options.get(), narrow_name.c_str(), override.Value());
    }
  }

  RETURN_HR_IF_NOT_OK_MSG(ort_api->SetIntraOpNumThreads(session_options.get(), intra_op_num_threads_override_), ort_api);

  if (!allow_thread_spinning_) {
    ort_api->AddSessionConfigEntry(session_options.get(), "session.intra_op.allow_spinning", "0");
  }

  OrtSession* ort_session = nullptr;
  onnxruntime_session_builder->CreateSession(session_options.get(), &ort_session);
  auto session = UniqueOrtSession(ort_session, ort_api->ReleaseSession);

  Microsoft::WRL::ComPtr<OnnxruntimeEngine> onnxruntime_engine;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngine>(&onnxruntime_engine,
                                                                        engine_factory_.Get(), std::move(session), onnxruntime_session_builder.Get()));
  RETURN_IF_FAILED(onnxruntime_engine.CopyTo(out));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::GetD3D12Device(_Outptr_ ID3D12Device** device) {
  *device = device_.Get();
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetD3D12Resources(ID3D12Device* device, ID3D12CommandQueue* queue) {
  device_ = device;
  queue_ = queue;
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetMetacommandsEnabled(int enabled) {
  metacommands_enabled_ = static_cast<bool>(enabled);
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::GetID3D12CommandQueue(_Outptr_ ID3D12CommandQueue** queue) {
  *queue = queue_.Get();
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetExecutionProviderOptions(_In_ _winml::IExecutionProviderOptions* options) {
  custom_execution_provider_options_ = options;
  return S_OK;
}


STDMETHODIMP OnnxruntimeEngineBuilder::SetBatchSizeOverride(uint32_t batch_size_override) {
  batch_size_override_ = batch_size_override;
  return S_OK;
}


STDMETHODIMP OnnxruntimeEngineBuilder::SetNamedDimensionOverrides(wfc::IMapView<winrt::hstring, uint32_t> named_dimension_overrides) {
  named_dimension_overrides_ = std::move(named_dimension_overrides);
  return S_OK;
}
  
STDMETHODIMP OnnxruntimeEngineBuilder::SetIntraOpNumThreadsOverride(uint32_t intra_op_num_threads) {
  intra_op_num_threads_override_ = intra_op_num_threads;
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetIntraOpThreadSpinning(bool allow_spinning) {
  allow_thread_spinning_ = allow_spinning;
  return S_OK;
}