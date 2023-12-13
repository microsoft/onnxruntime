#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelOutputs.h"
#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

namespace WINML_EXPERIMENTALP {

LearningModelOutputs::LearningModelOutputs(winml_experimental::LearningModelBuilder builder)
  : output_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
    builder_(builder) {
}

winml_experimental::LearningModelBuilder LearningModelOutputs::Add(winml::ILearningModelFeatureDescriptor const& output
) {
  // Perform model update inside the builder
  auto model = builder_.as<winml_experimentalp::LearningModelBuilder>()->UseModel();
  auto descriptor_provider = output.as<_winml::IDescriptorInfoProvider>();
  auto name = _winml::Strings::UTF8FromHString(output.Name());
  model->AddModelOutput(name.c_str(), descriptor_provider.get());
  output_descriptors_.Append(output);
  return builder_;
}

}  // namespace WINML_EXPERIMENTALP
