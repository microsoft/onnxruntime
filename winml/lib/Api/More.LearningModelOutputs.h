#pragma once

#include "LearningModelOutputs.g.h"

namespace MOREP {

struct LearningModelOutputs : LearningModelOutputsT<LearningModelOutputs>
{
    LearningModelOutputs(more::LearningModelBuilder builder);

    more::LearningModelBuilder Add(winml::ILearningModelFeatureDescriptor const& output);

    private:
    wfc::IVector<winml::ILearningModelFeatureDescriptor> output_descriptors_;
    more::LearningModelBuilder builder_;

};

}  // namespace MOREP