// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml {

MIDL_INTERFACE("eaae30b5-7381-432d-9730-322136b02371")
IModelInfo : IUnknown {
  STDMETHOD(GetAuthor)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetName)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetDomain)
  (const char** out, size_t* len) PURE;


  STDMETHOD(GetDescription)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetVersion)
  (int64_t * out) PURE;

  STDMETHOD(GetModelMetadata)
  (ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING> * *metadata) PURE;

  STDMETHOD(GetInputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> * *features) PURE;

  STDMETHOD(GetOutputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> * *features) PURE;
};

MIDL_INTERFACE("1b198b76-5c44-480d-837c-8433ca6eaf99")
IModel : IUnknown {
  STDMETHOD(GetModelInfo)
  (IModelInfo * *info) PURE;

  STDMETHOD(ModelEnsureNoFloat16)
  () PURE;

  STDMETHOD(CloneModel)
  (IModel * *copy) PURE;
};

using Resource = std::unique_ptr<void, std::function<void(void*)>>;
MIDL_INTERFACE("31f39226-cfe8-4758-af38-3d01b2a33ee1")
IValue : IUnknown {
  STDMETHOD(IsEmpty)
  (bool* out) PURE;

  STDMETHOD(IsCpu)
  (bool* out) PURE;

  STDMETHOD(GetResource)
  (_winml::Resource & resource) PURE;

  STDMETHOD(IsTensor)
  (bool* out) PURE;

  STDMETHOD(IsOfTensorType)
  (winml::TensorKind kind, bool* out) PURE;

  STDMETHOD(GetTensorShape)
  (std::vector<int64_t> & shape_vector) PURE;

  STDMETHOD(IsOfMapType)
  (winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) PURE;

  STDMETHOD(IsOfVectorMapType)
  (winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) PURE;

  STDMETHOD(IsOfVectorTensorType)
  (winml::TensorKind kind, bool* out) PURE;
};

MIDL_INTERFACE("30c99886-38d2-41cb-a615-203fe7d7daac")
IEngine : IUnknown {
  STDMETHOD(LoadModel)
  (_In_ IModel*) PURE;

  STDMETHOD(Initialize)
  () PURE;

  STDMETHOD(RegisterGraphTransformers)
  () PURE;

  STDMETHOD(RegisterCustomRegistry)
  (IMLOperatorRegistry * registry) PURE;

  STDMETHOD(EndProfiling)
  () PURE;

  STDMETHOD(StartProfiling)
  () PURE;

  STDMETHOD(FlushContext)
  () PURE;

  STDMETHOD(ReleaseCompletedReferences)
  () PURE;

  STDMETHOD(Sync)
  () PURE;

  STDMETHOD(CreateTensorValue)
  (const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateTensorValueFromExternalD3DResource)
  (ID3D12Resource * resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateTensorValueFromExternalBuffer)
  (void* data, size_t size_in_bytes, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateStringTensorValueFromDataWithCopy)
  (const char* const* data, size_t num_elements, const int64_t* shape, size_t count, _Out_ IValue** out) PURE;

  STDMETHOD(CreateNullValue)
  (_Out_ IValue * *out) PURE;

  STDMETHOD(CreateMapValue)
  (IInspectable * map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue * *out) PURE;

  STDMETHOD(CreateSequenceOfMapsValue)
  (IInspectable * sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue * *out) PURE;

  STDMETHOD(CreateSequenceOfValuesValue)
  (IValue ** values, size_t size, IValue * *out) PURE;

  STDMETHOD(CreateOneInputAcrossDevices)
  (const char* name, IValue* src, IValue** dest) PURE;

  STDMETHOD(CopyValueAcrossDevices)
  (IValue * src, IValue * dest) PURE;

  STDMETHOD(Run)
  (const char** input_names, IValue** inputs, size_t num_inputs, const char** output_names, IValue** outputs, size_t num_outputs) PURE;

  STDMETHOD(FillFromMapValue)
  (IInspectable * map, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue * value) PURE;

  STDMETHOD(FillSequenceOfMapsValue)
  (IInspectable * sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue * value) PURE;

  STDMETHOD(GetSequenceOfTensorValues)
  (_winml::IValue* sequence_value, _Out_ std::vector<winrt::com_ptr<_winml::IValue>>& out_values) PURE;

  STDMETHOD(GetNumberOfIntraOpThreads)
  (uint32_t * num_threads) PURE;
};

MIDL_INTERFACE("8ac0b6b9-4561-492b-b63d-a07bdd8292c6")
IEngineBuilder : IUnknown {
  STDMETHOD(SetD3D12Resources)
  (ID3D12Device * device, ID3D12CommandQueue * queue) PURE;

  STDMETHOD(SetMetacommandsEnabled)
  (int enabled) PURE;

  STDMETHOD(GetD3D12Device)
  (ID3D12Device * *device) PURE;

  STDMETHOD(GetID3D12CommandQueue)
  (ID3D12CommandQueue * *queue) PURE;

  STDMETHOD(SetBatchSizeOverride)
  (uint32_t batch_size_override) PURE;

  STDMETHOD(SetNamedDimensionOverrides)
  (wfc::IMapView<winrt::hstring, uint32_t> named_dimension_overrides) PURE;

  STDMETHOD(SetIntraOpNumThreadsOverride)
  (uint32_t intra_op_num_threads) PURE;

  STDMETHOD(CreateEngine)
  (IEngine * *out) PURE;
};

MIDL_INTERFACE("5eddd25a-70ad-46ef-a445-78fbaf792c2f")
IEngineFactory : IUnknown {
  STDMETHOD(CreateModel)
  (_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) PURE;

  STDMETHOD(CreateModel)
  (_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) PURE;

  STDMETHOD(CreateEngineBuilder)
  (IEngineBuilder * *engine_builder) PURE;

  STDMETHOD(EnableDebugOutput)
  (bool is_enabled) PURE;

  STDMETHOD(CreateCustomRegistry)
  (_Out_ IMLOperatorRegistry * *registry) PURE;
};

}  // namespace _winml
