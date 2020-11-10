#pragma once

#include "LearningModelSessionOptionsExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental> 
{
  LearningModelSessionOptionsExperimental(LearningModelSession const& options);

  wfc::IMapView<winrt::hstring, uint32_t> GetNamedDimensionOverrides();

private:
  wfc::IMapView<winrt::hstring, uint32_t> overrides_;
};
