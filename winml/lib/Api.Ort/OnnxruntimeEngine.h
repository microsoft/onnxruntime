#include "iengine.h"

#include "adapter/winml_adapter_c_api.h"

#include <memory>


namespace Windows::AI::MachineLearning {

using UniqueOrtSessionOptions = std::unique_ptr<OrtSessionOptions, void (*)(OrtSessionOptions*)>;
using UniqueOrtSession = std::unique_ptr<OrtSession, void (*)(OrtSession*)>;
using UniqueOrtExecutionProvider = std::unique_ptr<OrtExecutionProvider, void (*)(OrtExecutionProvider*)>;
using UniqueOrtValue = std::unique_ptr<OrtValue, void (*)(OrtValue*)>;
using UniqueOrtMemoryInfo = std::unique_ptr<OrtMemoryInfo, void (*)(OrtMemoryInfo*)>;
using UniqueOrtTypeInfo = std::unique_ptr<OrtTypeInfo, void (*)(OrtTypeInfo*)>;
using UniqueOrtTensorTypeAndShapeInfo = std::unique_ptr<OrtTensorTypeAndShapeInfo, void (*)(OrtTensorTypeAndShapeInfo*)>;
using UniqueOrtAllocator = std::unique_ptr<OrtAllocator, OrtStatus* (*)(OrtAllocator*)>;

class OnnxruntimeEngineBuilder;
class OnnxruntimeEngineFactory;
class OnnxruntimeEnvironment;
class OnnxruntimeModel;
class OnnxruntimeEngine;

struct IOrtSessionBuilder;

class OnnxruntimeValue : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IValue> {
 public:
  OnnxruntimeValue();
  ~OnnxruntimeValue();

  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, OnnxruntimeEngine* engine, UniqueOrtValue&& value, UniqueOrtAllocator&& allocator);

  STDMETHOD(IsCpu)(bool* out) override;
  STDMETHOD(GetResource)(void** resource) override;
  STDMETHOD(IsTensor)(bool* out) override;
  STDMETHOD(IsOfTensorType)(winml::TensorKind kind, bool* out) override;
  STDMETHOD(GetTensorShape)(std::vector<int64_t>& shape_vector) override;
  STDMETHOD(IsOfMapType)(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) override;
  STDMETHOD(IsOfVectorMapType)(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) override;

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<OnnxruntimeEngine> engine_;
  UniqueOrtValue value_;
  UniqueOrtAllocator allocator_; // Do we really need to cache this?
};

class OnnxruntimeEngine : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngine> {
 public:
  OnnxruntimeEngine();
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, UniqueOrtSession&& session, IOrtSessionBuilder* session_builder);
  
  STDMETHOD(LoadModel)(_In_ IModel* model) override;
  STDMETHOD(Initialize)() override;
  STDMETHOD(RegisterGraphTransformers)() override;
  STDMETHOD(RegisterCustomRegistry)(IMLOperatorRegistry * registry) override;
  STDMETHOD(EndProfiling)() override;
  STDMETHOD(StartProfiling)() override;
  STDMETHOD(FlushContext)() override;
  STDMETHOD(TrimUploadHeap)() override;
  STDMETHOD(ReleaseCompletedReferences)() override;
  STDMETHOD(CopyOneInputAcrossDevices)(const char* input_name, const IValue* src, IValue** dest) override;
  STDMETHOD(Sync)() override;
  STDMETHOD(CreateTensorValue)(int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) override;
  STDMETHOD(CopyOneInputAcrossDevices)(const char* name, IValue* src, IValue** out) override;

  OrtSession* UseOrtSession();
  bool IsDmlSession();

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<IOrtSessionBuilder> session_builder_;
  UniqueOrtSession session_;
};
  
class OnnxruntimeEngineFactory : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngineFactory> {
 public:
  HRESULT RuntimeClassInitialize();
  STDMETHOD(CreateModel)(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) override;
  STDMETHOD(CreateModel)(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) override;
  STDMETHOD(CreateEngineBuilder)(IEngineBuilder** engine_builder) override;
  STDMETHOD(EnableDebugOutput)(bool is_enabled) override;
  STDMETHOD(CreateCustomRegistry)(_Out_ IMLOperatorRegistry** registry) override;

  const OrtApi* UseOrtApi();
  const WinmlAdapterApi* UseWinmlAdapterApi();
  HRESULT GetOrtEnvironment(_Out_ OrtEnv** ort_env);

 private:
  const OrtApi* ort_api_ = nullptr;
  const WinmlAdapterApi* winml_adapter_api_ = nullptr;
  std::shared_ptr<OnnxruntimeEnvironment> environment_;
};


}  // namespace Windows::AI::MachineLearning