#pragma once

#include "LearningModelInputs.g.h"
#include "LearningModelBuilder.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelInputs : LearningModelInputsT<LearningModelInputs> {
  LearningModelInputs(winml_experimental::LearningModelBuilder builder);

  winml_experimental::LearningModelBuilder Add(winml::ILearningModelFeatureDescriptor const& input);
  winml_experimental::LearningModelBuilder Add(hstring const& input_name, hstring const& input_description, Windows::Foundation::IInspectable const& default_value);
  winml_experimental::LearningModelBuilder AddConstant(hstring const& input_name, Windows::Foundation::IInspectable const& value);
  winml_experimental::LearningModelBuilder AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant);

 private:
  wfc::IVector<winml::ILearningModelFeatureDescriptor> input_descriptors_;
  wfc::IVector<wf::IInspectable> input_default_values_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> constant_descriptors_;
  wfc::IVector<wf::IInspectable> constant_values_;
  winml_experimental::LearningModelBuilder builder_;
};
}  // namespace WINML_EXPERIMENTALP