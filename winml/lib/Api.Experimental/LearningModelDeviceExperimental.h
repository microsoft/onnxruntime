#pragma once
#include "LearningModelDeviceExperimental.g.h"

namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation {
struct LearningModelDeviceExperimental {
  LearningModelDeviceExperimental() = default;

  static Microsoft::AI::MachineLearning::LearningModelDevice CreateOpenVinoDevice();
};
}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation
namespace winrt::Microsoft::AI::MachineLearning::Experimental::factory_implementation {
struct LearningModelDeviceExperimental : LearningModelDeviceExperimentalT<LearningModelDeviceExperimental, implementation::LearningModelDeviceExperimental> {
};
}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::factory_implementation
