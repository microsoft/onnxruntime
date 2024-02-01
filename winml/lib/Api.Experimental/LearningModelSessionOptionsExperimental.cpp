#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "LearningModelSession.h"
#include "LearningModelSessionOptions.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {
LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(
  const winml::LearningModelSessionOptions& options
)
  : options_(options) {
}

LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(
  const winml::LearningModelSession& session
)
  : options_(nullptr) {
  com_ptr<WINMLP::LearningModelSession> session_impl = session.as<WINMLP::LearningModelSession>();
  options_ = session_impl->Options();

  _winml::IEngine* engine = session_impl->GetEngine();
  engine->GetNamedDimensionOverrides(overrides_);
}

wfc::IMapView<winrt::hstring, uint32_t> LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides() {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides");
  return overrides_;
}

void LearningModelSessionOptionsExperimental::RegisterCustomOpsLibrary(const hstring& path) {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::RegisterCustomOpsLibrary");
  com_ptr<WINMLP::LearningModelSessionOptions> options_impl = options_.as<WINMLP::LearningModelSessionOptions>();
  options_impl->RegisterCustomOpsLibrary(path);
}

winml_experimental::GraphOptimizationPolicy LearningModelSessionOptionsExperimental::GraphOptimizationPolicy() {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::get_GraphOptimizationPolicy");
  com_ptr<WINMLP::LearningModelSessionOptions> options_impl = options_.as<WINMLP::LearningModelSessionOptions>();
  auto is_optimization_enabled = options_impl->GraphOptimizationEnabled();
  if (is_optimization_enabled) {
    return winml_experimental::GraphOptimizationPolicy::All;
  }
  return winml_experimental::GraphOptimizationPolicy::None;
}

void LearningModelSessionOptionsExperimental::GraphOptimizationPolicy(
  winml_experimental::GraphOptimizationPolicy const& value
) {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::put_GraphOptimizationPolicy");
  com_ptr<WINMLP::LearningModelSessionOptions> options_impl = options_.as<WINMLP::LearningModelSessionOptions>();

  if (value == winml_experimental::GraphOptimizationPolicy::All) {
    options_impl->GraphOptimizationEnabled(true);
  } else {
    options_impl->GraphOptimizationEnabled(false);
  }
}

}  // namespace WINML_EXPERIMENTALP
