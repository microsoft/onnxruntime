#pragma once

#include "LearningModelSessionOptionsExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental> 
{
  LearningModelSessionOptionsExperimental(const winml::LearningModelSession& session);
  LearningModelSessionOptionsExperimental(const winml::LearningModelSessionOptions& options);

  wfc::IMapView<winrt::hstring, uint32_t> GetNamedDimensionOverrides();
  
  WINML_EXPERIMENTAL::GraphOptimizationLevel OptimizationLevel();
  void OptimizationLevel(WINML_EXPERIMENTAL::GraphOptimizationLevel level);

 private:
  wfc::IMapView<winrt::hstring, uint32_t> overrides_;

  winml::LearningModelSessionOptions session_options_ = nullptr;
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental, implementation::LearningModelSessionOptionsExperimental> {
};

}  // namespace WINML_EXPERIMENTAL::factory_implementation
