#include "pch.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "LearningModelSession.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {
LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(const winml::LearningModelSession& session) {
  com_ptr<WINMLP::LearningModelSession> session_impl = session.as<WINMLP::LearningModelSession>();
  _winml::IEngine* engine = session_impl->GetEngine();
  engine->GetNamedDimensionOverrides(overrides_);
}

wfc::IMapView<winrt::hstring, uint32_t> LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides() {
  telemetry_helper.LogApiUsage("LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides");

  return overrides_;
}

}  // namespace WINML_EXPERIMENTALP