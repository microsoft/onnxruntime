// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/weight_layout_transformer.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h"
#include <cstring>

namespace onnxruntime {
namespace webgpu {

// Template helper function to transpose weights from oihw to hwio layout
template <typename T>
void WeightLayoutTransformer::TransposeOIHWToHWIO(const T* src, T* dst,
                                                  int64_t O, int64_t I, int64_t H, int64_t W) {
  // Transpose from oihw to hwio
  // Source layout: [O][I][H][W]
  // Destination layout: [H][W][I][O]
  // Permutation: {2, 3, 1, 0}

  for (int64_t o = 0; o < O; ++o) {
    for (int64_t i = 0; i < I; ++i) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          // Source index: oihw
          const size_t src_idx = ((o * I + i) * H + h) * W + w;

          // Destination index: hwio
          const size_t dst_idx = ((h * W + w) * I + i) * O + o;

          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

// Template helper function to reorder weights from oihw to ABcd16a4b blocked format
template <typename T>
void WeightLayoutTransformer::ReorderToBlockedFormat(const T* src, T* dst,
                                                     int64_t O, int64_t I, int64_t H, int64_t W,
                                                     int64_t O_blocks, int64_t I_blocks,
                                                     int64_t block_o, int64_t block_i) {
  // Reorder from oihw to ABcd16a4b
  // Source layout: [O][I][H][W]
  // Destination layout: [O_blocks][I_blocks][H][W][block_o][block_i]
  //
  // Destination strides:
  // - O_blocks: I_blocks * H * W * block_o * block_i
  // - I_blocks: H * W * block_o * block_i
  // - H: W * block_o * block_i
  // - W: block_o * block_i
  // - block_o: block_i
  // - block_i: 1

  for (int64_t ob = 0; ob < O_blocks; ++ob) {
    for (int64_t ib = 0; ib < I_blocks; ++ib) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          for (int64_t o_in_block = 0; o_in_block < block_o; ++o_in_block) {
            for (int64_t i_in_block = 0; i_in_block < block_i; ++i_in_block) {
              const int64_t o = ob * block_o + o_in_block;
              const int64_t i = ib * block_i + i_in_block;

              // Calculate destination index for ABcd16a4b layout
              const size_t dst_idx =
                  ob * (I_blocks * H * W * block_o * block_i) +
                  ib * (H * W * block_o * block_i) +
                  h * (W * block_o * block_i) +
                  w * (block_o * block_i) +
                  o_in_block * block_i +
                  i_in_block;

              // Only copy if within original dimensions (handle padding)
              if (o < O && i < I) {
                // Source index: oihw format
                const size_t src_idx = ((o * I + i) * H + h) * W + w;
                dst[dst_idx] = src[src_idx];
              }
              // For padding (o >= O or i >= I), dst is already zero-initialized
            }
          }
        }
      }
    }
  }
}

Status WeightLayoutTransformer::TransformLayout(const Tensor& original_tensor,
                                                const std::string& format_descriptor,
                                                std::unique_ptr<Tensor>& transformed_tensor) {
  const auto& orig_shape = original_tensor.Shape();
  const auto* elem_type = original_tensor.DataType();

  // Only support 4D tensors
  if (orig_shape.NumDimensions() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported format transformation: ", format_descriptor);
  }

  // Validate tensor location (common for all formats)
  if (original_tensor.Location().device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Tensor is not on CPU, device type: ", original_tensor.Location().device.Type());
  }

  const int64_t O = orig_shape[0];
  const int64_t I = orig_shape[1];
  const int64_t H = orig_shape[2];
  const int64_t W = orig_shape[3];

  // Helper lambda to execute transformation for a specific data type
  auto execute_transform = [&]<typename T>(auto&& transform_func, const TensorShape& new_shape,
                                           size_t buffer_size = 0) -> Status {
    const T* src = original_tensor.Data<T>();
    if (!src) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Source tensor data pointer is null");
    }

    auto cpu_allocator = std::make_shared<CPUAllocator>();
    transformed_tensor = std::make_unique<Tensor>(elem_type, new_shape, cpu_allocator);
    T* dst = transformed_tensor->MutableData<T>();

    // Zero-initialize if buffer size is specified (for blocked formats with padding)
    if (buffer_size > 0) {
      std::memset(dst, 0, buffer_size * sizeof(T));
    }

    transform_func(src, dst);
    return Status::OK();
  };

  // Helper lambda to dispatch based on data type
  auto dispatch_by_type = [&](auto&& transform_func, const TensorShape& new_shape,
                              size_t buffer_size = 0, const char* error_msg = "Unsupported data type") -> Status {
    if (elem_type == DataTypeImpl::GetType<float>()) {
      return execute_transform.template operator()<float>(transform_func, new_shape, buffer_size);
    } else if (elem_type == DataTypeImpl::GetType<MLFloat16>()) {
      return execute_transform.template operator()<MLFloat16>(transform_func, new_shape, buffer_size);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
    }
  };

  if (format_descriptor == "hwio") {
    // Transpose from oihw to hwio
    // Transposed shape: [H, W, I, O]
    TensorShape new_shape({H, W, I, O});

    auto transform_func = [&](auto* src, auto* dst) {
      TransposeOIHWToHWIO(src, dst, O, I, H, W);
    };

    return dispatch_by_type(transform_func, new_shape, 0, "Unsupported data type for hwio transpose");
  } else if (format_descriptor == "ABcd16a4b") {
    // Reorder from oihw to blocked format
    constexpr int64_t block_o = 16;
    constexpr int64_t block_i = 4;

    const int64_t O_padded = ((O + block_o - 1) / block_o) * block_o;
    const int64_t I_padded = ((I + block_i - 1) / block_i) * block_i;
    const int64_t O_blocks = O_padded / block_o;
    const int64_t I_blocks = I_padded / block_i;

    // Keep 4D shape for kernel compatibility, but data is in blocked format
    // Shape: [O_padded, I_padded, H, W] with data internally blocked as ABcd16a4b
    TensorShape new_shape({O_padded, I_padded, H, W});
    const size_t buffer_size = O_blocks * I_blocks * H * W * block_o * block_i;

    auto transform_func = [&](auto* src, auto* dst) {
      ReorderToBlockedFormat(src, dst, O, I, H, W, O_blocks, I_blocks, block_o, block_i);
    };

    return dispatch_by_type(transform_func, new_shape, buffer_size, "Unsupported data type for blocked format");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported format transformation: ", format_descriptor);
}

}  // namespace webgpu
}  // namespace onnxruntime
