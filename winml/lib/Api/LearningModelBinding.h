// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "LearningModelBinding.g.h"

#include "inc/ILotusValueProviderPrivate.h"
#include "core/providers/winml/winml_provider_factory.h"

namespace winrt::Windows::AI::MachineLearning::implementation {

struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, ILearningModelBindingNative> {
  struct ProviderInfo {
    Windows::Foundation::IInspectable CallerSpecifiedFeatureValue = nullptr;
    winrt::com_ptr<WinML::ILotusValueProviderPrivate> Provider = nullptr;
    WinML::BindingContext Context = {};
  };

 public:
  using KeyValuePair =
      Windows::Foundation::Collections::IKeyValuePair<hstring, Windows::Foundation::IInspectable>;

  LearningModelBinding() = delete;
  ~LearningModelBinding();
  LearningModelBinding(Windows::AI::MachineLearning::LearningModelSession const& session);

  void Bind(hstring const& name, Windows::Foundation::IInspectable const& value);
  void Bind(hstring const& name, Windows::Foundation::IInspectable const& value, Windows::Foundation::Collections::IPropertySet const& properties);
  void Clear();
  Windows::Foundation::Collections::IIterator<KeyValuePair> First();
  Windows::Foundation::IInspectable Lookup(hstring const& key);
  uint32_t Size();
  bool HasKey(hstring const& key);
  void Split(
      Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable>& first,
      Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable>& second);

  std::tuple<std::string, OrtValue*, WinML::BindingType, OrtAllocator*> CreateBinding(
      const std::string& name,
      const Windows::Foundation::IInspectable& value,
      Windows::Foundation::Collections::IPropertySet const& properties);

  std::unordered_map<std::string, Windows::Foundation::IInspectable> UpdateProviders();

  const Windows::AI::MachineLearning::LearningModelSession& GetSession() { return m_session; }

  STDMETHOD(Bind)
  (
      const wchar_t* name,
      UINT32 cchName,
      IUnknown* value);

  const std::vector<std::string>& LearningModelBinding::GetOutputNames() const;
  std::vector<Ort::Value>& LearningModelBinding::GetOutputs();
  const std::vector<std::string>& LearningModelBinding::GetInputNames() const;
  const std::vector<Ort::Value>& LearningModelBinding::GetInputs() const;
  HRESULT BindOutput(const std::string& name, Ort::Value&& ml_value, Ort::Allocator&& ort_allocator);
  void BindUnboundOutputs();

 private:
  void CacheProvider(std::string name, ProviderInfo& spProvider);
  Windows::Foundation::IInspectable CreateUnboundOutput(const std::string& name, Ort::Value& ort_value);
  ILearningModelFeatureValue CreateUnboundOuputFeatureValue(
      const Ort::Value& ort_value,
      ILearningModelFeatureDescriptor& descriptor);
  bool IsOfTensorType(const Ort::Value& ort_value, TensorKind kind);
  bool IsOfMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind);
  bool IsOfVectorMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind);
  HRESULT BindInput(const std::string& name, Ort::Value&& ml_value, Ort::Allocator&& ort_allocator);

 private:
  const Windows::AI::MachineLearning::LearningModelSession m_session;

  std::unordered_map<std::string, ProviderInfo> m_providers;

  com_ptr<winmla::IWinMLAdapter> adapter_;
  std::vector<std::string> input_names_;
  std::vector<Ort::Value> inputs_;
  std::vector<Ort::Allocator> input_allocators_;
  std::vector<std::string> output_names_;
  std::vector<Ort::Value> outputs_;
  std::vector<Ort::Allocator> output_allocators_;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, implementation::LearningModelBinding> {
};
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
