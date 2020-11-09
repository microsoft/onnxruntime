#include "pch.h"

#include "LearningModelSessionOptionsExperimental.h"
#include "LearningModelSessionExperimental.h"

namespace WINML_EXPERIMENTALP {

LearningModelSessionExperimental::LearningModelSessionExperimental(LearningModelSession session) {
  _session = session;
}

LearningModelSessionOptions LearningModelSessionExperimental::Options() {
  throw hresult_not_implemented();
}

}  // namespace WINML_EXPERIMENTALP