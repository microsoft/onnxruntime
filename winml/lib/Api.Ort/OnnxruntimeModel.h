// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "iengine.h"

namespace _winml {

class OnnxruntimeEngineFactory;

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
MIDL_INTERFACE("92679cbf-7a9d-48bb-b97f-ef9fb447ce8e")
IOnnxruntimeModel : IUnknown {
  virtual HRESULT STDMETHODCALLTYPE DetachOrtModel(OrtModel * *model) PURE;
};

class ModelInfo : public Microsoft::WRL::RuntimeClass<
                      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                      IModelInfo> {
 public:
  HRESULT RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine, _In_ OrtModel* ort_model);

  STDMETHOD(GetAuthor)
  (const char** out, size_t* len);
  STDMETHOD(GetName)
  (const char** out, size_t* len);
  STDMETHOD(GetDomain)
  (const char** out, size_t* len);
  STDMETHOD(GetDescription)
  (const char** out, size_t* len);
  STDMETHOD(GetVersion)
  (int64_t* out);
  STDMETHOD(GetModelMetadata)
  (ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata);
  STDMETHOD(GetInputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);
  STDMETHOD(GetOutputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);

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
                            IModel,
                            IOnnxruntimeModel> {
 public:
  OnnruntimeModel();

  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine, UniqueOrtModel&& ort_model);

  STDMETHOD(GetModelInfo)
  (IModelInfo** info);
  STDMETHOD(ModelEnsureNoFloat16)
  ();
  STDMETHOD(CloneModel)
  (IModel** copy);
  STDMETHOD(DetachOrtModel)
  (OrtModel** model);

 private:
  UniqueOrtModel ort_model_;

  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<ModelInfo> info_;

  std::optional<std::unordered_map<std::string, std::string>> metadata_cache_;
};

}  // namespace _winml