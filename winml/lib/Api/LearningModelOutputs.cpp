#include "pch.h"
#include "LearningModelOutputs.h"
#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{

LearningModelOutputs::LearningModelOutputs(winml::More::LearningModelBuilder builder) :
    builder_(builder),
    output_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()) {
}

more::LearningModelBuilder LearningModelOutputs::Add(winml::ILearningModelFeatureDescriptor const& output)
{
  // Perform model update inside the builder
  auto model = builder_.as<morep::LearningModelBuilder>()->UseModel();

  auto descriptor_provider = output.as<WinML::IDescriptorInfoProvider>();

  auto name = WinML::Strings::UTF8FromHString(output.Name());
  model->AddModelOutput(name.c_str(), descriptor_provider.get());
  output_descriptors_.Append(output);

  return builder_;
}

}
