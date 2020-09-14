#include "pch.h"
#include "LearningModelInputs.h"
#include "LearningModelOperator.h"

#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

#include "..\Api\inc\ILotusValueProviderPrivate.h"

namespace WINML_EXPERIMENTALP {

LearningModelInputs::LearningModelInputs(winml_experimental::LearningModelBuilder builder) : builder_(builder),
                                                                                      input_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      input_default_values_(winrt::single_threaded_vector<wf::IInspectable>()),
                                                                                      constant_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      constant_values_(winrt::single_threaded_vector<wf::IInspectable>()) {
}

winml_experimental::LearningModelBuilder LearningModelInputs::AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant) {
  // Perform model update inside the builder
  auto model = builder_.as<winml_experimentalp::LearningModelBuilder>()->UseModel();
  auto descriptor_provider = input.as<_winml::IDescriptorInfoProvider>();
  auto input_name = _winml::Strings::UTF8FromHString(input.Name());

  winrt::com_ptr<_winml::IValue> default_value_ivalue;
  if (default_value) {
    auto default_value_value_provider = default_value.as<_winml::ILotusValueProviderPrivate>();
    _winml::BindingContext bc{};
    default_value_value_provider->GetValue(bc, default_value_ivalue.put());
  }

  model->AddModelInput(input_name.c_str(), descriptor_provider.get(), is_constant, default_value_ivalue.get());

  return builder_;
}

winml_experimental::LearningModelBuilder LearningModelInputs::Add(winml::ILearningModelFeatureDescriptor const& input) {
  return AddInput(input, nullptr, false);
}

winml_experimental::LearningModelBuilder LearningModelInputs::Add(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value) {
  return AddInput(input, default_value, false);
}

winml_experimental::LearningModelBuilder LearningModelInputs::AddConstant(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& value) {
  return AddInput(input, value, true);
}
}  // namespace WINML_EXPERIMENTALP