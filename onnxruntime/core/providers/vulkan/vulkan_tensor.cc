// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_tensor.h"

namespace onnxruntime {

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

std::array<int64_t, 4> VulkanTensor::TensorShapeFormat(const TensorShape& tensor_shape) {
  auto rank = tensor_shape.GetDims().size();

  int64_t N = -1;
  int64_t C = -1;
  int64_t H = -1;
  int64_t W = -1;

  if (rank >= 3) {
    N = tensor_shape[0];
    C = tensor_shape[1];
    H = tensor_shape[2];
    W = 1;

    if (rank == 4) {
      W = tensor_shape[3];
    } else if (rank > 4) {
      for (size_t i = 3; i < rank; ++i) {
        W *= tensor_shape[i];
      }
    }
  } else if (rank == 2) {
    N = tensor_shape[0];
    C = tensor_shape[1];

    H = 1;
    W = 1;
  } else if (rank == 1) {
    N = 1;
    C = tensor_shape[0];

    H = 1;
    W = 1;
  } else {
    ORT_THROW("Unsupported tensor shape");
  }

  return {N, H, W, C};
}

int64_t VulkanTensor::GetAlignSize(const Tensor* tensor) {
  auto element_size = tensor->DataType()->Size();
  return ALIGN_UP4(element_size);
}

VulkanTensor::VulkanTensor(const TensorShape& tensor_shape, MLDataType data_type,
                           VulkanMemoryAllocationHelper& memory_alloc_helper,
                           const VkPhysicalDeviceLimits& memory_limits) {
  auto nhwc = TensorShapeFormat(tensor_shape);

  auto width = UP_DIV(nhwc[3], 4) * nhwc[2];
  auto height = nhwc[0] * nhwc[1];

  int64_t unit = memory_limits.maxImageDimension2D;
  blocks_[0] = UP_DIV(width, unit);
  blocks_[1] = UP_DIV(height, unit);

  size_ = std::move(nhwc);

  images_.resize(blocks_[0] * blocks_[1]);
  for (int64_t y = 0; y < blocks_[1]; ++y) {
    auto y_start = y * unit;
    auto y_finish = std::min(height, y_start + unit);

    auto h_real = y_finish - y_start;

    for (int64_t x = 0; x < blocks_[0]; ++x) {
      auto x_start = x * unit;
      auto x_finish = std::min(width, x_start + unit);

      auto w_real = x_finish - x_start;

      images_[y * blocks_[0] + x] = std::make_shared<VulkanImage>(memory_alloc_helper,
                                                                  std::vector<int64_t>{w_real, h_real},
                                                                  data_type);
    }
  }
}

void VulkanTensor::Release() {
  for (auto image : images_) {
    image->Release();
  }
}

}  // namespace onnxruntime
