// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Ort/pch.h"

#include "OnnxruntimeEngine.h"
#include "OnnxruntimeEngineBuilder.h"
#include "OnnxruntimeCpuSessionBuilder.h"

#ifdef USE_DML
#include "OnnxruntimeDmlSessionBuilder.h"
#endif

#include "OnnxruntimeErrors.h"
using namespace _winml;

HRESULT OnnxruntimeEngineBuilder::RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine_factory) {
  engine_factory_ = engine_factory;
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::CreateEngine(_Outptr_ _winml::IEngine** out) {
  *out = nullptr;
  auto ort_api = engine_factory_->UseOrtApi();

  Microsoft::WRL::ComPtr<IOrtSessionBuilder> onnxruntime_session_builder;

  if (device_ == nullptr) {
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeCpuSessionBuilder>(&onnxruntime_session_builder, engine_factory_.Get()));
  } else {
#ifdef USE_DML
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeDmlSessionBuilder>(&onnxruntime_session_builder, engine_factory_.Get(), device_.Get(), queue_.Get(), metacommands_enabled_));
#endif
  }

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

  if (intra_op_num_threads_override_.has_value()) {
    RETURN_HR_IF_NOT_OK_MSG(ort_api->SetIntraOpNumThreads(session_options.get(), intra_op_num_threads_override_.value()), ort_api);
  }

  if (!allow_thread_spinning_) {
    RETURN_HR_IF_NOT_OK_MSG(ort_api->AddSessionConfigEntry(session_options.get(), "session.intra_op.allow_spinning", "0"),
                            ort_api);
  }

  std::vector<void*> handles;
  for (const auto& path : custom_ops_lib_paths_) {
    void* handle = nullptr;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->RegisterCustomOpsLibrary(session_options.get(), path.c_str(), &handle),
                            ort_api);
    handles.push_back(handle);
  }

  OrtThreadPool* inter_op_ort_pool = thread_pool_ ? static_cast<OnnxruntimeThreading*>(thread_pool_.Get())->UseInterOpThreadPool() : nullptr;
  OrtThreadPool* intra_op_ort_pool = thread_pool_ ? static_cast<OnnxruntimeThreading*>(thread_pool_.Get())->UseIntraOpThreadPool() : nullptr;

  OrtSession* ort_session = nullptr;
  onnxruntime_session_builder->CreateSession(session_options.get(), inter_op_ort_pool, intra_op_ort_pool, &ort_session);
  auto session = UniqueOrtSession(ort_session, ort_api->ReleaseSession);

  Microsoft::WRL::ComPtr<OnnxruntimeEngine> onnxruntime_engine;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngine>(&onnxruntime_engine,
                                                                        engine_factory_.Get(), std::move(session), onnxruntime_session_builder.Get()));
  RETURN_IF_FAILED(onnxruntime_engine->RegisterCustomOpLibraryHandles(handles));
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

STDMETHODIMP OnnxruntimeEngineBuilder::RegisterCustomOpsLibrary(const char* path) {
  custom_ops_lib_paths_.push_back(path);
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetIntraOpThreadSpinning(bool allow_spinning) {
  allow_thread_spinning_ = allow_spinning;
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineBuilder::SetThreadPool(IThreading* thread_pool) {
  thread_pool_ = thread_pool;
  return S_OK;
}
