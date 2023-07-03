#pragma once

#include "LearningModelSessionOptionsExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental>
{
  LearningModelSessionOptionsExperimental(const winml::LearningModelSessionOptions& options);
  LearningModelSessionOptionsExperimental(const winml::LearningModelSession& session);

  wfc::IMapView<winrt::hstring, uint32_t> GetNamedDimensionOverrides();
  void RegisterCustomOpsLibrary(const hstring& path);

 private:
  wfc::IMapView<winrt::hstring, uint32_t> overrides_;
  winml::LearningModelSessionOptions options_;
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelSessionOptionsExperimental : LearningModelSessionOptionsExperimentalT<LearningModelSessionOptionsExperimental, implementation::LearningModelSessionOptionsExperimental>
{
};

}
