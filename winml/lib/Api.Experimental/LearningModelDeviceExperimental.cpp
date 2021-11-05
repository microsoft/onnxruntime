#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelDeviceExperimental.h"

#include "LearningModelDevice.h"
#include "iengine.h"

#include <wrl.h>
#include <wrl/client.h>

#ifdef USE_OPENVINO
MIDL_INTERFACE("4e84bd04-4212-4ecd-92a4-0b403e75872c")
IOpenVinoProviderOptions : IUnknown {
  // nothing here yet
};

class OpenVinoExecutionProviderOptions : public Microsoft::WRL::RuntimeClass<
                                     Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                     _winml::IExecutionProviderOptions,
                                     IOpenVinoProviderOptions>
{
  // nothing here yet
};
#endif

#ifdef USE_TENSORRT
MIDL_INTERFACE("55411b0d-06d1-426d-94a1-cdd70802f1bd")
ITensorRTProviderOptions : IUnknown {
  // nothing here yet
};

class TensorRTExecutionProviderOptions : public Microsoft::WRL::RuntimeClass<
                                     Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                     _winml::IExecutionProviderOptions,
                                     ITensorRTProviderOptions>
{
  // nothing here yet
};
#endif

#ifdef USE_CUDA
MIDL_INTERFACE("34001d56-7780-4aa9-9e37-da05d2ead6d2")
ICUDAProviderOptions : IUnknown {
  // nothing here yet
};

class CUDAExecutionProviderOptions : public Microsoft::WRL::RuntimeClass<
                                     Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                     _winml::IExecutionProviderOptions,
                                     ICUDAProviderOptions>
{
  // nothing here yet
};
#endif

namespace WINML_EXPERIMENTALP {

winml::LearningModelDevice LearningModelDeviceExperimental::CreateDMLDevice() {
  return winrt::make<winmlp::LearningModelDevice>(winml::LearningModelDeviceKind::DirectXHighPerformance);
}

#ifdef USE_OPENVINO
winml::LearningModelDevice LearningModelDeviceExperimental::CreateOpenVinoDevice() {
  auto options = ::Microsoft::WRL::Make<OpenVinoExecutionProviderOptions>();
  return winrt::make<winmlp::LearningModelDevice>(options.Get());
}
#endif

#ifdef USE_TENSORRT
winml::LearningModelDevice LearningModelDeviceExperimental::CreateTensorRTDevice() {
  auto options = ::Microsoft::WRL::Make<TensorRTExecutionProviderOptions>();
  return winrt::make<winmlp::LearningModelDevice>(options.Get());
}
#endif

#ifdef USE_CUDA
winml::LearningModelDevice LearningModelDeviceExperimental::CreateCUDADevice() {
  auto options = ::Microsoft::WRL::Make<CUDAExecutionProviderOptions>();
  return winrt::make<winmlp::LearningModelDevice>(options.Get());
}
#endif

}  // namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation
