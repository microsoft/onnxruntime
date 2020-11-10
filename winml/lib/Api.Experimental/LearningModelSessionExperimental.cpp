#include "pch.h"

#include "LearningModelSessionOptionsExperimental.h"
#include "LearningModelSessionExperimental.h"

namespace WINML_EXPERIMENTALP {

LearningModelSessionExperimental::LearningModelSessionExperimental(Microsoft::AI::MachineLearning::LearningModelSession session) : _session(session) {}


WINML_EXPERIMENTAL::LearningModelSessionOptionsExperimental LearningModelSessionExperimental::Options() {
  WINML_EXPERIMENTAL::LearningModelSessionOptionsExperimental options(_session);
  return options;
}

}  // namespace WINML_EXPERIMENTALP