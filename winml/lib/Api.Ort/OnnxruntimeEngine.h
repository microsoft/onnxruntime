// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "iengine.h"
#include "UniqueOrtPtr.h"

#include <memory>
#include <mutex>

namespace _winml {

class OnnxruntimeEngineBuilder;
class OnnxruntimeEngineFactory;
class OnnxruntimeEnvironment;
class OnnxruntimeModel;
class OnnxruntimeEngine;
class OnnxruntimeThreading;

struct IOrtSessionBuilder;

class OnnxruntimeValue
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IValue> {
 public:
  OnnxruntimeValue();
  ~OnnxruntimeValue();

  HRESULT RuntimeClassInitialize(OnnxruntimeEngine* engine, UniqueOrtValue&& value, UniqueOrtAllocator&& allocator);

  STDMETHOD(IsEmpty)
  (bool* out) override;
  STDMETHOD(IsCpu)
  (bool* out) override;
  STDMETHOD(GetResource)
  (_winml::Resource& resource) override;
  STDMETHOD(IsTensor)
  (bool* out) override;
  STDMETHOD(IsOfTensorType)
  (winml::TensorKind kind, bool* out) override;
  STDMETHOD(GetTensorShape)
  (std::vector<int64_t>& shape_vector) override;
  STDMETHOD(IsOfMapType)
  (winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) override;
  STDMETHOD(IsOfVectorMapType)
  (winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) override;
  STDMETHOD(IsOfVectorTensorType)
  (winml::TensorKind kind, bool* out) override;

  HRESULT(SetParameter)
  (IUnknown* param);
  OrtValue* UseOrtValue();
  HRESULT AssignOrtValue(OrtValue* ptr);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngine> engine_;
  Microsoft::WRL::ComPtr<IUnknown> param_;
  UniqueOrtValue value_;
  UniqueOrtAllocator allocator_;
};

class OnnxruntimeEngine
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IEngine> {
 public:
  OnnxruntimeEngine();
  ~OnnxruntimeEngine();

  HRESULT RuntimeClassInitialize(
    OnnxruntimeEngineFactory* engine_factory, UniqueOrtSession&& session, IOrtSessionBuilder* session_builder
  );

  STDMETHOD(LoadModel)
  (_In_ IModel* model) override;
  STDMETHOD(Initialize)
  () override;
  STDMETHOD(RegisterGraphTransformers)
  () override;
  STDMETHOD(RegisterCustomRegistry)
  (IMLOperatorRegistry* registry) override;
  STDMETHOD(EndProfiling)
  () override;
  STDMETHOD(StartProfiling)
  () override;
  STDMETHOD(FlushContext)
  () override;
  STDMETHOD(ReleaseCompletedReferences)
  () override;
  STDMETHOD(Sync)
  () override;
  STDMETHOD(CreateTensorValue)
  (const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) override;
  STDMETHOD(CreateTensorValueFromExternalD3DResource)
  (ID3D12Resource* resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) override;
  STDMETHOD(CreateTensorValueFromExternalBuffer)
  (void* data, size_t size_in_bytes, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out
  ) override;
  STDMETHOD(CreateStringTensorValueFromDataWithCopy)
  (const char* const* data, size_t num_elements, const int64_t* shape, size_t count, _Out_ IValue** out) override;
  STDMETHOD(CreateNullValue)
  (_Out_ IValue** out) override;
  STDMETHOD(CreateMapValue)
  (IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) override;
  STDMETHOD(CreateSequenceOfMapsValue)
  (IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out) override;
  STDMETHOD(CreateSequenceOfValuesValue)
  (IValue** values, size_t size, IValue** out) override;

  STDMETHOD(CreateOneInputAcrossDevices)
  (const char* name, IValue* src, IValue** dest) override;
  STDMETHOD(CopyValueAcrossDevices)
  (IValue* src, IValue* dest) override;
  STDMETHOD(Run)
  (const char** input_names,
   IValue** inputs,
   size_t num_inputs,
   const char** output_names,
   IValue** outputs,
   size_t num_outputs) override;
  STDMETHOD(FillFromMapValue)
  (IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* value) override;
  STDMETHOD(FillSequenceOfMapsValue)
  (IInspectable* sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* value) override;

  STDMETHOD(GetSequenceOfTensorValues)
  (_In_ _winml::IValue* sequence_value, _Out_ std::vector<winrt::com_ptr<_winml::IValue>>& out_values) override;

  STDMETHOD(GetNumberOfIntraOpThreads)
  (uint32_t* num_threads) override;

  STDMETHOD(GetIntraOpThreadSpinning)
  (bool* allow_spinning) override;

  STDMETHOD(GetNamedDimensionOverrides)
  (wfc::IMapView<winrt::hstring, uint32_t>& overrides) override;

  OrtSession* UseOrtSession();
  const OrtApi* UseOrtApi();
  OnnxruntimeEngineFactory* GetEngineFactory();
  HRESULT
  CreateTensorValueFromDefaultAllocator(const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out);

  HRESULT RegisterCustomOpLibraryHandles(const gsl::span<void*> handles);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  Microsoft::WRL::ComPtr<IOrtSessionBuilder> session_builder_;
  UniqueOrtSession session_;
  std::vector<void*> custom_op_library_handles_;
};

class OnnxruntimeEngineFactory
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IEngineFactory> {
 public:
  HRESULT RuntimeClassInitialize();
  STDMETHOD(CreateModel)
  (_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) override;
  STDMETHOD(CreateModel)
  (_In_opt_ void* data, _In_ size_t size, _Outptr_ IModel** out) override;
  STDMETHOD(CreateEmptyModel)
  (_In_ int64_t opset, _Outptr_ IModel** out) override;
  STDMETHOD(CreateEngineBuilder)
  (_Outptr_ IEngineBuilder** engine_builder) override;
  STDMETHOD(EnableDebugOutput)
  (bool is_enabled) override;
  STDMETHOD(CreateCustomRegistry)
  (_Out_ IMLOperatorRegistry** registry) override;

  const OrtApi* UseOrtApi();
  const WinmlAdapterApi* UseWinmlAdapterApi();
  HRESULT EnsureEnvironment();
  HRESULT GetOrtEnvironment(_Out_ OrtEnv** ort_env);

  STDMETHOD(CreateTensorDescriptorInfo)
  (_In_ winml::TensorKind kind, _In_ int64_t* dims, _In_ size_t num_dims, _Out_ IDescriptorInfo** info) override;

  STDMETHOD(CreateSequenceDescriptorInfo)
  (_Out_ IDescriptorInfo** info) override;

  STDMETHOD(CreateMapDescriptorInfo)
  (_Out_ IDescriptorInfo** info) override;

  STDMETHOD(CreateThreadPool)
  (_In_ bool allow_spinning, _In_ uint32_t num_intra_op_threads, _Out_ IThreading** thread_pool) override;

 private:
  const OrtApi* ort_api_ = nullptr;
  const WinmlAdapterApi* winml_adapter_api_ = nullptr;
  std::shared_ptr<OnnxruntimeEnvironment> environment_;
  std::mutex mutex_;
};

class OnnxruntimeThreading
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IThreading> {
 public:
  OnnxruntimeThreading();
  ~OnnxruntimeThreading();

  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory);

  HRESULT SetIntraOpThreadPool(UniqueOrtThreadPool&& intra_op_ort_pool);
  HRESULT SetInterOpThreadPool(UniqueOrtThreadPool&& inter_op_ort_pool);
  OrtThreadPool* UseIntraOpThreadPool();
  OrtThreadPool* UseInterOpThreadPool();

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_ = nullptr;
  UniqueOrtThreadPool inter_op_ort_pool_;
  UniqueOrtThreadPool intra_op_ort_pool_;
};

}  // namespace _winml
