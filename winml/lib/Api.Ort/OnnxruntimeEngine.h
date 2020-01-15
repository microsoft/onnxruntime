#include "iengine.h"

#include "adapter/winml_adapter_c_api.h"

#include <memory>


namespace Windows::AI::MachineLearning {

using UniqueOrtSessionOptions = std::unique_ptr<OrtSessionOptions, void (*)(OrtSessionOptions*)>;
using UniqueOrtSession = std::unique_ptr<OrtSession, void (*)(OrtSession*)>;
using UniqueOrtExecutionProvider = std::unique_ptr<OrtExecutionProvider, void (*)(OrtExecutionProvider*)>;

class OnnxruntimeEngineBuilder;
class OnnxruntimeEngineFactory;
class OnnxruntimeEnvironment;
class OnnxruntimeModel;

struct IOrtSessionBuilder;

class OnnxruntimeEngine : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngine> {
 public:
  OnnxruntimeEngine();
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, UniqueOrtSession&& session, IOrtSessionBuilder* session_builder);
  
  STDMETHOD(LoadModel)(_In_ IModel* model);
  STDMETHOD(Initialize)();
  STDMETHOD(RegisterGraphTransformers)();
  STDMETHOD(RegisterCustomRegistry)(IMLOperatorRegistry * registry);
  STDMETHOD(EndProfiling)();
  STDMETHOD(StartProfiling)();
  STDMETHOD(FlushContext)();
  STDMETHOD(TrimUploadHeap)();
  STDMETHOD(ReleaseCompletedReferences)();
  STDMETHOD(CopyOneInputAcrossDevices)(const char* input_name, const IValue* src, IValue** dest);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<IOrtSessionBuilder> session_builder_;
  UniqueOrtSession session_;
  UniqueOrtExecutionProvider provider_;
};
  
class OnnxruntimeEngineFactory : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngineFactory> {
 public:
  HRESULT RuntimeClassInitialize();
  STDMETHOD(CreateModel)(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out);
  STDMETHOD(CreateModel)(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out);
  STDMETHOD(CreateEngineBuilder)(IEngineBuilder** engine_builder);
  const OrtApi* UseOrtApi();
  const WinmlAdapterApi* UseWinmlAdapterApi();
  HRESULT GetOrtEnvironment(_Out_ OrtEnv** ort_env);

 private:
  const OrtApi* ort_api_ = nullptr;
  const WinmlAdapterApi* winml_adapter_api_ = nullptr;
  std::shared_ptr<OnnxruntimeEnvironment> environment_;
};


}  // namespace Windows::AI::MachineLearning