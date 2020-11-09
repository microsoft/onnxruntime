#pragma once

#include "LearningModelSessionOptionsExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental> 
{
  LearningModelSessionOptionsExperimental(LearningModelSessionOptions options);

  Windows::Foundation::Collections::IMapView<winrt::hstring, uint32_t> GetNamedDimensionOverrides();
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental, implementation::LearningModelSessionOptionsExperimental> {
};

}  // namespace WINML_EXPERIMENTAL::factory_implementation