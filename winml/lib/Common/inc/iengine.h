// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct OrtTypeInfo;

namespace _winml {

interface IEngineFactory;

using Resource = std::unique_ptr<void, std::function<void(void*)>>;

// clang-format off
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

MIDL_INTERFACE("4637dfcb-fc19-45c3-a632-c84942d0cf8e")
IOrtTypeInfoProvider : IUnknown {
  STDMETHOD(GetTypeInfo)
  (OrtTypeInfo** info) PURE;
};

MIDL_INTERFACE("fe94665f-76cb-42a2-ab21-a06ae1c7f1ae")
IDescriptorInfo : IUnknown{

};

MIDL_INTERFACE("e3feaec4-eb09-4b82-973c-781f1c230842")
IDescriptorInfoProvider : IUnknown{
  STDMETHOD(GetDescriptorInfo)
  (IEngineFactory* engine_factory, IDescriptorInfo** info) PURE;
};


MIDL_INTERFACE("eaae30b5-7381-432d-9730-322136b02371")
IModelInfo : IUnknown {
  STDMETHOD(GetAuthor)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetName)
  (const char** out, size_t* len) PURE;

  STDMETHOD(SetName)
  (const char* name) PURE;

  STDMETHOD(GetDomain)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetDescription)
  (const char** out, size_t* len) PURE;

  STDMETHOD(GetVersion)
  (int64_t* out) PURE;

  STDMETHOD(GetModelMetadata)
  (ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) PURE;

  STDMETHOD(GetInputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) PURE;

  STDMETHOD(GetOutputFeatures)
  (ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) PURE;
};

MIDL_INTERFACE("1b198b76-5c44-480d-837c-8433ca6eaf99")
IModel : IUnknown {
  STDMETHOD(GetModelInfo)
  (IModelInfo** info) PURE;

  STDMETHOD(ModelEnsureNoFloat16)
  () PURE;

  STDMETHOD(CloneModel)
  (IModel** copy) PURE;

  STDMETHOD(SaveModel)
  (_In_ const wchar_t* const file_name,
   _In_ unsigned size) PURE;

  STDMETHOD(SetName)
  (const char* name) PURE;

  STDMETHOD(AddOperator)
  (_In_ const char* const op_type, _In_ const char* const op_name, _In_ const char* const op_domain,
   _In_ const char* const* op_input_names, _In_ const char* const* actual_input_names, size_t num_inputs,
   _In_ const char* const* op_output_names, _In_ const char* const* actual_output_names, size_t num_outputs,
   _In_ const char* const* op_attribute_names, _In_ IValue** constant_value, size_t num_attributes) PURE;

  STDMETHOD(AddModelInput)
  (_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider, bool is_constant, IValue* default_value) PURE;

  STDMETHOD(AddModelOutput)
  (_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider) PURE;

  STDMETHOD(JoinModel)
  (_In_ IModel* other_model, _In_ const char* const* output_names, _In_ const char* const* input_names,
   size_t num_linkages, bool promote_unlinked_outputs, _In_ const char* const join_node_prefix) PURE;
};

MIDL_INTERFACE("30c99886-38d2-41cb-a615-203fe7d7daac")
IEngine : IUnknown {
  STDMETHOD(LoadModel)
  (_In_ IModel*)PURE;

  STDMETHOD(Initialize)
  () PURE;

  STDMETHOD(RegisterGraphTransformers)
  () PURE;

  STDMETHOD(RegisterCustomRegistry)
  (IMLOperatorRegistry* registry) PURE;

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
  (ID3D12Resource* resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateTensorValueFromExternalBuffer)
  (void* data, size_t size_in_bytes, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateStringTensorValueFromDataWithCopy)
  (const char* const* data, size_t num_elements, const int64_t* shape, size_t count, _Out_ IValue** out) PURE;

  STDMETHOD(CreateNullValue)
  (_Out_ IValue** out) PURE;

  STDMETHOD(CreateMapValue)
  (IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateSequenceOfMapsValue)
  (IInspectable* sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) PURE;

  STDMETHOD(CreateSequenceOfValuesValue)
  (IValue**  values, size_t size, IValue** out) PURE;

  STDMETHOD(CreateOneInputAcrossDevices)
  (const char* name, IValue* src, IValue** dest) PURE;

  STDMETHOD(CopyValueAcrossDevices)
  (IValue* src, IValue* dest) PURE;

  STDMETHOD(Run)
  (const char** input_names, IValue** inputs, size_t num_inputs, const char** output_names, IValue** outputs, size_t num_outputs) PURE;

  STDMETHOD(FillFromMapValue)
  (IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* value) PURE;

  STDMETHOD(FillSequenceOfMapsValue)
  (IInspectable* sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* value) PURE;

  STDMETHOD(GetSequenceOfTensorValues)
  (_In_ _winml::IValue* sequence_value, _Out_ std::vector<winrt::com_ptr<_winml::IValue>>& out_values) PURE;

  STDMETHOD(GetNumberOfIntraOpThreads)
  (uint32_t* num_threads) PURE;

  STDMETHOD(GetIntraOpThreadSpinning)
  (bool* allow_spinning) PURE;

  STDMETHOD(GetNamedDimensionOverrides)
  (wfc::IMapView<winrt::hstring, uint32_t>& overrides) PURE;
};

MIDL_INTERFACE("69fbc1d5-66ff-4c79-b14d-fad288dbf8ca")
IThreading : IUnknown {

};

MIDL_INTERFACE("8ac0b6b9-4561-492b-b63d-a07bdd8292c6")
IEngineBuilder : IUnknown {
  STDMETHOD(SetD3D12Resources)
  (ID3D12Device* device, ID3D12CommandQueue* queue) PURE;

  STDMETHOD(SetMetacommandsEnabled)
  (int enabled) PURE;

  STDMETHOD(GetD3D12Device)
  (_Outptr_ ID3D12Device** device) PURE;

  STDMETHOD(GetID3D12CommandQueue)
  (_Outptr_ ID3D12CommandQueue** queue) PURE;

  STDMETHOD(SetBatchSizeOverride)
  (uint32_t batch_size_override) PURE;

  STDMETHOD(SetNamedDimensionOverrides)
  (wfc::IMapView<winrt::hstring, uint32_t> named_dimension_overrides) PURE;

  STDMETHOD(SetIntraOpNumThreadsOverride)
  (uint32_t intra_op_num_threads) PURE;

  STDMETHOD(SetIntraOpThreadSpinning)
  (bool allow_spinning) PURE;

  STDMETHOD(SetThreadPool)
  (IThreading* thread_pool) PURE;

  STDMETHOD(RegisterCustomOpsLibrary)
  (const char* name) PURE;

  STDMETHOD(CreateEngine)
  (_Outptr_ IEngine** out) PURE;
};


MIDL_INTERFACE("5eddd25a-70ad-46ef-a445-78fbaf792c2f")
IEngineFactory : IUnknown {
  STDMETHOD(CreateModel)
  (_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) PURE;

  STDMETHOD(CreateModel)
  (_In_opt_ void* data, _In_ size_t size, _Outptr_ IModel** out) PURE;

  STDMETHOD(CreateEmptyModel)
  (_In_ int64_t opset, _Outptr_ IModel** out) PURE;

  STDMETHOD(CreateEngineBuilder)
  (_Outptr_ IEngineBuilder** engine_builder) PURE;

  STDMETHOD(EnableDebugOutput)
  (bool is_enabled) PURE;

  STDMETHOD(CreateCustomRegistry)
  (_Out_ IMLOperatorRegistry** registry) PURE;

  STDMETHOD(CreateTensorDescriptorInfo)
  (
      _In_ winml::TensorKind kind,
      _In_ int64_t* dims,
      _In_ size_t num_dims,
      _Out_ IDescriptorInfo** info) PURE;

  STDMETHOD(CreateSequenceDescriptorInfo)
  (_Out_ IDescriptorInfo** info) PURE;

  STDMETHOD(CreateMapDescriptorInfo)
  (_Out_ IDescriptorInfo** info) PURE;

  STDMETHOD(CreateThreadPool)
  (_In_ bool allow_spinning, _In_ uint32_t num_intra_op_threads, _Out_ IThreading** thread_pool) PURE;
};

// clang-format on

}  // namespace _winml
