// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "iengine.h"

namespace _winml {

class OnnxruntimeEngineBuilder
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IEngineBuilder> {
 public:
  HRESULT RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine);

  STDMETHOD(SetD3D12Resources)
  (ID3D12Device* device, ID3D12CommandQueue* queue);

  STDMETHOD(SetMetacommandsEnabled)
  (int enabled);

  STDMETHOD(GetD3D12Device)
  (_Outptr_ ID3D12Device** device);

  STDMETHOD(GetID3D12CommandQueue)
  (_Outptr_ ID3D12CommandQueue** queue);

  STDMETHOD(SetBatchSizeOverride)
  (uint32_t batch_size_override);

  STDMETHOD(SetNamedDimensionOverrides)
  (wfc::IMapView<winrt::hstring, uint32_t> named_dimension_overrides);

  STDMETHOD(SetIntraOpNumThreadsOverride)
  (uint32_t intra_op_num_threads);

  STDMETHOD(SetIntraOpThreadSpinning)
  (bool allow_spinning);

  STDMETHOD(SetThreadPool)
  (IThreading* thread_pool);

  STDMETHOD(RegisterCustomOpsLibrary)
  (const char* path);

  STDMETHOD(CreateEngine)
  (_Outptr_ IEngine** out);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<ID3D12Device> device_ = nullptr;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_ = nullptr;
  Microsoft::WRL::ComPtr<IThreading> thread_pool_ = nullptr;
  bool metacommands_enabled_ = true;
  std::optional<uint32_t> batch_size_override_;
  wfc::IMapView<winrt::hstring, uint32_t> named_dimension_overrides_;
  std::optional<uint32_t> intra_op_num_threads_override_;
  bool allow_thread_spinning_ = true;
  std::vector<std::string> custom_ops_lib_paths_;
};

}  // namespace _winml
