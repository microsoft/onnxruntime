#include "pch.h"
#include "LearningModelInputs.h"
#include "LearningModelOperator.h"

#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

namespace WINML_EXPERIMENTALP {

LearningModelInputs::LearningModelInputs(winml_experimental::LearningModelBuilder builder) : builder_(builder),
                                                                                      input_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      input_default_values_(winrt::single_threaded_vector<wf::IInspectable>()),
                                                                                      constant_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      constant_values_(winrt::single_threaded_vector<wf::IInspectable>()) {
}

winml_experimental::LearningModelOperator LearningModelInputs::Then(winml_experimental::LearningModelOperator const& next_operator) {
  auto operator_impl = next_operator.as<winml_experimentalp::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

winml_experimental::LearningModelOperator LearningModelInputs::Then(winml_experimental::LearningModelOperator const& next_operator, winml_experimental::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  auto operator_impl = next_operator.as<winml_experimentalp::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

winml_experimental::LearningModelBuilder LearningModelInputs::AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant) {
  // Perform model update inside the builder
  auto model = builder_.as<winml_experimentalp::LearningModelBuilder>()->UseModel();

  auto descriptor_provider = input.as<_winml::IDescriptorInfoProvider>();

  auto input_name = _winml::Strings::UTF8FromHString(input.Name());
  model->AddModelInput(input_name.c_str(), descriptor_provider.get(), is_constant);

  if (is_constant) {
    constant_descriptors_.Append(input);
    constant_values_.Append(default_value);
  } else {
    input_descriptors_.Append(input);
    input_default_values_.Append(default_value);
  }

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

wfc::IVector<winml::ILearningModelFeatureDescriptor> LearningModelInputs::Inputs() {
  auto all_inputs = winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>();

  std::vector<winml::ILearningModelFeatureDescriptor> all;

  std::copy(begin(input_descriptors_), end(input_descriptors_), std::back_inserter(all));
  std::copy(begin(constant_descriptors_), end(constant_descriptors_), std::back_inserter(all));

  return winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>(std::move(all));
}

}  // namespace WINML_EXPERIMENTALP