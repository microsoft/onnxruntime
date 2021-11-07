#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelDeviceExperimental.h"

#include "LearningModelDevice.h"
#include "iengine.h"

#include <wrl.h>
#include <wrl/client.h>

#ifdef USE_OPENVINO
MIDL_INTERFACE("f292bc6f-0dd3-423c-bb72-3575222285e1")
IOpenVinoProviderOptions : IUnknown {
  // nothing here yet
};

class OpenVinoExecutionProviderOptions : public Microsoft::WRL::RuntimeClass<
                                     Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                     _winml::IExecutionProviderOptions,
                                     IOpenVinoProviderOptions>
{
  STDMETHOD(ForceCpuBindings)
  (bool* options)
  {
    *options = false;
    return S_OK;
  }
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
  STDMETHOD(ForceCpuBindings)
  (bool* options)
  {
    *options = true;
    return S_OK;
  }

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
  STDMETHOD(ForceCpuBindings)
  (bool* options)
  {
    *options = true;
    return S_OK;
  }
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
