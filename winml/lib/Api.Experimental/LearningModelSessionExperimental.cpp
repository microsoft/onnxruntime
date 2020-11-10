#include "pch.h"

#include "LearningModelSessionOptionsExperimental.h"
#include "LearningModelSessionExperimental.h"

namespace WINML_EXPERIMENTALP {

LearningModelSessionExperimental::LearningModelSessionExperimental(Microsoft::AI::MachineLearning::LearningModelSession const& session) : _session(session) {}

LearningModelSessionOptionsExperimental LearningModelSessionExperimental::Options() {
  return winrt::make<WINML_EXPERIMENTAL::LearningModelSessionOptionsExperimental>(_session);
}

}  // namespace WINML_EXPERIMENTALP