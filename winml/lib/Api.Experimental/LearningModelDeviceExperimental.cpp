#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelDeviceExperimental.h"

#include "LearningModelDevice.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {
winml::LearningModelDevice LearningModelDeviceExperimental::CreateOpenVinoDevice() {
  auto options = std::make_unique<_winml::OpenVinoDeviceOptions>();
  return winrt::make<winmlp::LearningModelDevice>(std::move(options));
}
}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation
