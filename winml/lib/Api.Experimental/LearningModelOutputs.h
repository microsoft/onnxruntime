#pragma once

#include "LearningModelOutputs.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelOutputs : LearningModelOutputsT<LearningModelOutputs> {
  LearningModelOutputs(winml_experimental::LearningModelBuilder builder);

  winml_experimental::LearningModelBuilder Add(winml::ILearningModelFeatureDescriptor const& output);

 private:
  wfc::IVector<winml::ILearningModelFeatureDescriptor> output_descriptors_;
  winml_experimental::LearningModelBuilder builder_;
};

}  // namespace WINML_EXPERIMENTALP
