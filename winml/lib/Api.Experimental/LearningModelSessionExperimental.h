#pragma once

#include "LearningModelSessionExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelSessionExperimental : LearningModelSessionExperimentalT<LearningModelSessionExperimental> {
  LearningModelSessionExperimental(const winml::LearningModelSession& session);

  WINML_EXPERIMENTAL::LearningModelSessionOptionsExperimental Options();

private:
  winml::LearningModelSession _session;
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelSessionExperimental : LearningModelSessionExperimentalT<LearningModelSessionExperimental, implementation::LearningModelSessionExperimental> {
};

}  // namespace WINML_EXPERIMENTAL::factory_implementation