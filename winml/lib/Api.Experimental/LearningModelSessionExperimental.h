#pragma once

#include "LearningModelSessionExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionExperimental : LearningModelSessionExperimentalT<LearningModelSessionExperimental> {
  LearningModelSessionExperimental(LearningModelSession const& session);

  LearningModelSessionOptionsExperimental Options();

private:
  LearningModelSession _session;
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelSessionExperimental : LearningModelSessionExperimentalT<LearningModelSessionExperimental, implementation::LearningModelSessionExperimental> {
};

}  // namespace WINML_EXPERIMENTAL::factory_implementation