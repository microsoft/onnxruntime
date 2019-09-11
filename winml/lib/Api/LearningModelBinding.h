#pragma once

#include "LearningModelBinding.g.h"

#include "inc/ILotusValueProviderPrivate.h"

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

  std::tuple<std::string, OrtValue, WinML::BindingType> CreateBinding(
      const std::string& name,
      const Windows::Foundation::IInspectable& value,
      Windows::Foundation::Collections::IPropertySet const& properties);

  onnxruntime::IOBinding& BindingCollection();
  std::unordered_map<std::string, Windows::Foundation::IInspectable> UpdateProviders();

  const Windows::AI::MachineLearning::LearningModelSession& GetSession() { return m_session; }

  STDMETHOD(Bind)
  (
      const wchar_t* name,
      UINT32 cchName,
      IUnknown* value);

 private:
  void CacheProvider(std::string name, ProviderInfo& spProvider);
  Windows::Foundation::IInspectable CreateUnboundOutput(const std::string& name, OrtValue& mlValue);

 private:
  const Windows::AI::MachineLearning::LearningModelSession m_session;

  std::unordered_map<std::string, ProviderInfo> m_providers;

  std::unique_ptr<onnxruntime::IOBinding> m_lotusBinding;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
struct LearningModelBinding : LearningModelBindingT<LearningModelBinding, implementation::LearningModelBinding> {
};
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
