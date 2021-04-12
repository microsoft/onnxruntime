#include "pch.h"

#include "LearningModelSessionOptionsExperimental.h"
#include "LearningModelSessionExperimental.h"

namespace WINML_EXPERIMENTALP {

LearningModelSessionExperimental::LearningModelSessionExperimental(const winml::LearningModelSession& session) :
  _session(session) {
}

WINML_EXPERIMENTAL::LearningModelSessionOptionsExperimental LearningModelSessionExperimental::Options() {
  return winrt::make<WINML_EXPERIMENTALP::LearningModelSessionOptionsExperimental>(_session);
}

}  // namespace WINML_EXPERIMENTALP