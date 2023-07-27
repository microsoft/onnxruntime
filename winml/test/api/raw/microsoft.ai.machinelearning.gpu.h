#pragma once

#ifndef MICROSOFT_AI_MACHINELEARNING_GPU_H_
#define MICROSOFT_AI_MACHINELEARNING_GPU_H_

#include "microsoft.ai.machinelearning.h"

namespace Microsoft {
namespace AI {
namespace MachineLearning {
namespace gpu {

enum directx_device_kind {
  directx,
  directx_high_power,
  directx_min_power
};

struct directx_device : public learning_model_device {
  directx_device(directx_device_kind kind) : learning_model_device(create_device(kind)) {}

  directx_device(ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice* d3dDevice)
    : learning_model_device(create_device(d3dDevice)) {}

  directx_device(ID3D12CommandQueue* queue) : learning_model_device(create_device(queue)) {}

 private:
  static Details::learning_model_device create_device(directx_device_kind kind) {
    switch (kind) {
      case directx_device_kind::directx:
        return Details::learning_model_device::create_directx_device();
      case directx_device_kind::directx_high_power:
        return Details::learning_model_device::create_directx_high_power_device();
      case directx_device_kind::directx_min_power:
        return Details::learning_model_device::create_directx_min_power_device();
    };

    return Details::learning_model_device();
  }

  static Details::learning_model_device create_device(
    ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice* d3dDevice
  ) {
    return Details::learning_model_device::create_directx_device(d3dDevice);
  }

  static Details::learning_model_device create_device(ID3D12CommandQueue* queue) {
    return Details::learning_model_device::create_directx_device(queue);
  }
};

}// namespace gpu
}// namespace MachineLearning
}// namespace AI
} // namespace Microsoft

#endif // MICROSOFT_AI_MACHINELEARNING_GPU_H
