#pragma once
#include "LearningModelDeviceExperimental.g.h"

namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation {
struct LearningModelDeviceExperimental {
  LearningModelDeviceExperimental() = default;

  static Microsoft::AI::MachineLearning::LearningModelDevice CreateDMLDevice();
#ifdef USE_OPENVINO
  static Microsoft::AI::MachineLearning::LearningModelDevice CreateOpenVinoDevice();
#endif
#ifdef USE_TENSORRT
  static Microsoft::AI::MachineLearning::LearningModelDevice CreateTensorRTDevice();
#endif
#ifdef USE_CUDA
  static Microsoft::AI::MachineLearning::LearningModelDevice CreateCUDADevice();
#endif

};
}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation
namespace winrt::Microsoft::AI::MachineLearning::Experimental::factory_implementation {
struct LearningModelDeviceExperimental : LearningModelDeviceExperimentalT<LearningModelDeviceExperimental, implementation::LearningModelDeviceExperimental> {
};
}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::factory_implementation
