#include "pch.h"
#include "LearningModelInputs.h"
#include "LearningModelOperator.h"

#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation {
LearningModelInputs::LearningModelInputs(winml::More::LearningModelBuilder builder) : builder_(builder),
                                                                                      input_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      input_default_values_(winrt::single_threaded_vector<wf::IInspectable>()),
                                                                                      constant_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      constant_values_(winrt::single_threaded_vector<wf::IInspectable>()) {
}

more::LearningModelOperator LearningModelInputs::Then(more::LearningModelOperator const& next_operator) {
  auto operator_impl = next_operator.as<morep::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

more::LearningModelOperator LearningModelInputs::Then(more::LearningModelOperator const& next_operator, more::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  auto operator_impl = next_operator.as<morep::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

more::LearningModelBuilder LearningModelInputs::AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant) {
  // Perform model update inside the builder
  auto model = builder_.as<morep::LearningModelBuilder>()->UseModel();

  auto descriptor_provider = input.as<WinML::IDescriptorInfoProvider>();

  auto input_name = WinML::Strings::UTF8FromHString(input.Name());
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

more::LearningModelBuilder LearningModelInputs::Add(winml::ILearningModelFeatureDescriptor const& input) {
  return AddInput(input, nullptr, false);
}

more::LearningModelBuilder LearningModelInputs::Add(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value) {
  return AddInput(input, default_value, false);
}

more::LearningModelBuilder LearningModelInputs::AddConstant(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& value) {
  return AddInput(input, value, true);
}

wfc::IVector<winml::ILearningModelFeatureDescriptor> LearningModelInputs::Inputs() {
  auto all_inputs = winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>();

  std::vector<winml::ILearningModelFeatureDescriptor> all;

  std::copy(begin(input_descriptors_), end(input_descriptors_), std::back_inserter(all));
  std::copy(begin(constant_descriptors_), end(constant_descriptors_), std::back_inserter(all));

  return winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>(std::move(all));
}

}  // namespace winrt::Windows::AI::MachineLearning::More::implementation