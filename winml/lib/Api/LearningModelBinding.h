// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelBinding.g.h"

#include "inc/ILotusValueProviderPrivate.h"
#include "core/providers/winml/winml_provider_factory.h"

namespace WINMLP {

struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, ILearningModelBindingNative> {
  struct ProviderInfo {
    wf::IInspectable CallerSpecifiedFeatureValue = nullptr;
    winrt::com_ptr<_winml::ILotusValueProviderPrivate> Provider = nullptr;
    _winml::BindingContext Context = {};
  };

 public:
  using KeyValuePair = wfc::IKeyValuePair<hstring, wf::IInspectable>;

  ~LearningModelBinding();

  LearningModelBinding() = delete;
  LearningModelBinding(winml::LearningModelSession const& session);

  void Bind(hstring const& name, wf::IInspectable const& value);
  void Bind(hstring const& name, wf::IInspectable const& value, wfc::IPropertySet const& properties);
  STDMETHOD(Bind)(const wchar_t* name, UINT32 cchName, IUnknown* value);

  void Clear();
  wfc::IIterator<KeyValuePair> First();
  wf::IInspectable Lookup(hstring const& key);
  uint32_t Size();
  bool HasKey(hstring const& key);
  void Split(wfc::IMapView<hstring, wf::IInspectable>& first, wfc::IMapView<hstring, wf::IInspectable>& second);

  std::tuple<std::string, winrt::com_ptr<_winml::IValue>, _winml::BindingType> CreateBinding(
    const std::string& name, const wf::IInspectable& value, wfc::IPropertySet const& properties
  );

  std::unordered_map<std::string, wf::IInspectable> UpdateProviders();

  const winml::LearningModelSession& GetSession() { return m_session; }

  const std::vector<std::string>& GetInputNames() const;
  const std::vector<std::string>& GetOutputNames() const;

  const std::vector<winrt::com_ptr<_winml::IValue>>& GetInputs() const;
  std::vector<winrt::com_ptr<_winml::IValue>>& GetOutputs();

  HRESULT BindOutput(const std::string& name, winrt::com_ptr<_winml::IValue> value);
  void BindUnboundOutputs();

 private:
  void CacheProvider(std::string name, ProviderInfo& spProvider);
  wf::IInspectable CreateUnboundOutput(const std::string& name, winrt::com_ptr<_winml::IValue> value);
  ILearningModelFeatureValue CreateUnboundOuputFeatureValue(
    const winrt::com_ptr<_winml::IValue> value, ILearningModelFeatureDescriptor& descriptor
  );
  HRESULT BindInput(const std::string& name, winrt::com_ptr<_winml::IValue> value);

 private:
  const winml::LearningModelSession m_session;

  std::unordered_map<std::string, ProviderInfo> m_providers;

  std::vector<std::string> input_names_;
  std::vector<winrt::com_ptr<_winml::IValue>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<winrt::com_ptr<_winml::IValue>> outputs_;
};
}  // namespace WINMLP

namespace WINML::factory_implementation {
struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, implementation::LearningModelBinding> {};
}  // namespace WINML::factory_implementation
