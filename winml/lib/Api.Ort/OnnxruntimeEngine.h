#include "iengine.h"

#include "core/session/winml_adapter_c_api.h"

#include <memory>

namespace Windows::AI::MachineLearning {

using UniqueOrtModel = std::unique_ptr<OrtModel, void (*)(OrtModel*)>;

class OnnxruntimeEngineFactory;

class ModelInfo : public Microsoft::WRL::RuntimeClass<
                      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                      IModelInfo> {
 public:
  HRESULT RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine, _In_ OrtModel* ort_model);

  STDMETHOD(GetAuthor)(const char** out, size_t* len);
  STDMETHOD(GetName)(const char** out, size_t* len);
  STDMETHOD(GetDomain)(const char** out, size_t* len);
  STDMETHOD(GetDescription)(const char** out, size_t* len);
  STDMETHOD(GetVersion)(int64_t* out);
  STDMETHOD(GetModelMetadata)(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata);
  STDMETHOD(GetInputFeatures)(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);
  STDMETHOD(GetOutputFeatures)(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);

 private:
  std::string author_;
  std::string name_;
  std::string domain_;
  std::string description_;
  int64_t version_;
  std::unordered_map<std::string, std::string> model_metadata_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> input_features_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> output_features_;
};

class OnnruntimeModel : public Microsoft::WRL::RuntimeClass<
                            Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                            IModel> {
 public:
  OnnruntimeModel();

  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine, UniqueOrtModel&& ort_model);

  STDMETHOD(GetModelInfo)(IModelInfo** info);
  STDMETHOD(CloneModel)(IModel** copy);
  
 private:
  UniqueOrtModel ort_model_;

  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<ModelInfo> info_;

  std::optional<std::unordered_map<std::string, std::string>> metadata_cache_;
};

class OnnxruntimeEngine : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngine> {
 public:
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
};


class OnnxruntimeEngineFactory : public Microsoft::WRL::RuntimeClass<
                              Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                              IEngineFactory> {
 public:
  HRESULT RuntimeClassInitialize();
  STDMETHOD(CreateModel)(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out);
  STDMETHOD(CreateModel)(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out);
  STDMETHOD(CreateEngine)(_Outptr_ IEngine** out);

  const OrtApi* UseOrtApi();
  const WinmlAdapterApi* UseWinmlAdapterApi();

 private:
  const OrtApi* ort_api_ = nullptr;
  const WinmlAdapterApi* winml_adapter_api_ = nullptr;
};


}  // namespace Windows::AI::MachineLearning