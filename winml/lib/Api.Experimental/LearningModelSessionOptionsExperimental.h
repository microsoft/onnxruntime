#pragma once

#include "LearningModelSessionOptionsExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental> 
{
  LearningModelSessionOptionsExperimental(const winml::LearningModelSession& options);

  wfc::IMapView<winrt::hstring, uint32_t> GetNamedDimensionOverrides();

 private:
  wfc::IMapView<winrt::hstring, uint32_t> overrides_;
};

}  // namespace WINML_EXPERIMENTALP
