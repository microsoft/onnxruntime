#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "LearningModelSession.h"
#include "LearningModelSessionOptions.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {
LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(const winml::LearningModelSessionOptions& options) :
  options_(options)
{}

LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(const winml::LearningModelSession& session) :
  options_(nullptr)
{
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

}  // namespace WINML_EXPERIMENTALP
