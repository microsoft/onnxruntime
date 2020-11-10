#include "pch.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "LearningModelSession.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {

LearningModelSessionOptionsExperimental::LearningModelSessionOptionsExperimental(Microsoft::AI::MachineLearning::LearningModelSession const& session) {
  com_ptr<WINMLP::LearningModelSession> session_impl = session.as<WINMLP::LearningModelSession>();
  _winml::IEngine* engine = session_impl->GetEngine();
  engine->GetNamedDimensionOverrides(overrides_);
}

Windows::Foundation::Collections::IMapView<winrt::hstring, uint32_t> LearningModelSessionOptionsExperimental::GetNamedDimensionOverrides() {
  return overrides_;
}

}  // namespace WINML_EXPERIMENTALP