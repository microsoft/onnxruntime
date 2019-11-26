// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "LearningModelBinding.g.h"

#include "inc/ILotusValueProviderPrivate.h"
#include "WinMLAdapter.h"

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

  std::tuple<std::string, OrtValue*, WinML::BindingType> CreateBinding(
      const std::string& name,
      const Windows::Foundation::IInspectable& value,
      Windows::Foundation::Collections::IPropertySet const& properties);

  _winmla::IIOBinding* BindingCollection();
  std::unordered_map<std::string, Windows::Foundation::IInspectable> UpdateProviders();

  const Windows::AI::MachineLearning::LearningModelSession& GetSession() { return m_session; }

  STDMETHOD(Bind)
  (
      const wchar_t* name,
      UINT32 cchName,
      IUnknown* value);

 private:
  void CacheProvider(std::string name, ProviderInfo& spProvider);
  Windows::Foundation::IInspectable CreateUnboundOutput(const std::string& name, Ort::Value& ort_value);
  ILearningModelFeatureValue CreateUnboundOuputFeatureValue(
      const Ort::Value& ort_value,
      ILearningModelFeatureDescriptor& descriptor);
  bool IsOfTensorType(const Ort::Value& ort_value, TensorKind kind);
  bool IsOfMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind);
  bool IsOfVectorMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind);
  HRESULT BindInput(const std::string& name, const OrtValue& ml_value);
  HRESULT BindOutput(const std::string& name, const OrtValue& ml_value);

 private:
  const Windows::AI::MachineLearning::LearningModelSession m_session;

  std::unordered_map<std::string, ProviderInfo> m_providers;

  com_ptr<_winmla::IIOBinding> m_lotusBinding;
  com_ptr<_winmla::IWinMLAdapter> adapter_;
  std::vector<std::string> feed_names_;
  std::vector<const OrtValue*> feeds_;
  std::vector<std::string> output_names_;
  std::vector<const OrtValue*> outputs_;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, implementation::LearningModelBinding> {
};
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
