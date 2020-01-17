#include "iengine.h"

namespace Windows::AI::MachineLearning {

class OnnxruntimeEngineBuilder : public Microsoft::WRL::RuntimeClass<
                                     Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                     IEngineBuilder> {
 public:
  HRESULT RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine);

  STDMETHOD(GetD3D12Device)(_Outptr_ ID3D12Device** device);
  STDMETHOD(SetD3D12Device)(_In_ ID3D12Device* device);
  STDMETHOD(GetID3D12CommandQueue)(_Outptr_ ID3D12CommandQueue** queue);
  STDMETHOD(SetID3D12CommandQueue)(_In_ ID3D12CommandQueue* queue);
  STDMETHOD(SetBatchSizeOverride)(uint32_t batch_size_override);
  STDMETHOD(CreateEngine)(_Outptr_ IEngine** out);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<ID3D12Device> device_ = nullptr;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_ = nullptr;
  std::optional<uint32_t> batch_size_override_;
};

}  // namespace Windows::AI::MachineLearning