#pragma once
#define ONNX_ML
#define ONNX_NAMESPACE onnx
#include "core/framework/execution_provider.h"

namespace onnxruntime {

constexpr const char* custom_ep_type = "custom_ep";

struct CustomEpInfo {
  OrtDevice::DeviceId device_id{0};
  std::string some_config;
};

class CustomEp : public IExecutionProvider {
public:
  explicit CustomEp(const CustomEpInfo& info);
  virtual ~CustomEp() {}
};

}
