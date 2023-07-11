#include "custom_ep.h"
#include "custom_allocator.h"

namespace onnxruntime {

CustomEp::CustomEp(const CustomEpInfo& info) : IExecutionProvider{custom_ep_type} {
  AllocatorCreationInfo device_info{
      [](OrtDevice::DeviceId device_id) { return std::make_unique<CustomAllocator>(device_id); },
      info.device_id,
      true,
      {0, 1, -1, -1, -1, -1L}};
  std::cout<<"end of constructor of CustomEp\n";
}

}
