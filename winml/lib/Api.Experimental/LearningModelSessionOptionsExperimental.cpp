#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "LearningModelSession.h"
#include "LearningModelSessionOptions.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {
LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(const winml::LearningModelSession& session) {
  com_ptr<WINMLP::LearningModelSession> session_impl = session.as<WINMLP::LearningModelSession>();
  _winml::IEngine* engine = session_impl->GetEngine();
  engine->GetNamedDimensionOverrides(overrides_);
  session_options_ = session_impl->Options();
}

LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(
  const winml::LearningModelSessionOptions& options) {

  auto options_impl = options.as<WINMLP::LearningModelSessionOptions>();
  overrides_ = options_impl->NamedDimensionOverrides();

  session_options_ = options;
}

wfc::IMapView<winrt::hstring, uint32_t> LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides() {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides");

  return overrides_;
}


GraphOptimizationLevel LearningModelSessionOptionsExperimental::OptimizationLevel()
{
  auto options_impl = session_options_.as<WINMLP::LearningModelSessionOptions>();
  return static_cast<GraphOptimizationLevel>(options_impl->OptimizationLevel());
}

void LearningModelSessionOptionsExperimental::OptimizationLevel(WINML_EXPERIMENTAL::GraphOptimizationLevel level)
{
  auto options_impl = session_options_.as<WINMLP::LearningModelSessionOptions>();
  options_impl->OptimizationLevel(static_cast<uint32_t>(level));
}

}  // namespace WINML_EXPERIMENTALP