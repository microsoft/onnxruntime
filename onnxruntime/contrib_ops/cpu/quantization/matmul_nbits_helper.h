// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace matmul_nbits_helper {

template <typename T = Tensor>
Status CheckInputs(const T* /*activation*/,
                   const T* quantized_weight,
                   const T* scales,
                   const T* zero_points,
                   const T* group_index,
                   const T* bias,
                   int64_t n,
                   int64_t k,
                   int64_t block_size,
                   int64_t bits,
                   bool is_b_prepacked) {
  // activation (A)
  // quantized_weight (B) : (N, k_blocks, blob_size)
  //                        k_blocks = (K + block_size - 1) / block_size
  //                        blob_size = (block_size * bits + 7) / 8
  // scales               : (N, k_blocks)
  // zero_points          : (N, (k_blocks * bits + 7) / 8) for uint8
  //                        (N, k_blocks) for float types
  // group_index          : (K) or (k_blocks * block_size)
  // bias                 : (N)
  int64_t k_blocks = (k + block_size - 1) / block_size;
  int64_t blob_size = (block_size * bits + 7) / 8;

  if (bits != 4 && bits != 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bits should be 4 or 8, got ", bits);
  }

  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_size must be a power of 2, and >= 16. Got ", block_size);
  }

  if (!is_b_prepacked) {
    const auto& quantized_weight_dims = quantized_weight->Shape().GetDims();
    if (quantized_weight_dims.size() != 3) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT, "Input 'B' is expected to have 3 dimensions, got ",
          quantized_weight_dims.size());
    }
    if (quantized_weight_dims[0] != n) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'quantized_weight' dimension 0 should be N, got ",
                             quantized_weight_dims[0], ". Expected:", n);
    }
    if (quantized_weight_dims[1] != k_blocks) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'quantized_weight' dimension 1 should be equal to (K + block_size - 1) / block_size, got ",
          quantized_weight_dims[1], ". Exptected:", k_blocks);
    }
    if (quantized_weight_dims[2] != blob_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'quantized_weight' dimension 2 should be equal to (block_size * bits + 7) / 8, got ",
          quantized_weight_dims[2], ". Expected:", blob_size);
    }
  } else {
    // Do nothing. B is not available after prepacking. Assume that B tensor shape has been checked during prepacking.
  }

  const auto& scales_dims = scales->Shape().GetDims();
  if (scales_dims.size() == 2) {
    if (scales_dims[0] != n) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'scales' dimension 0 should be N, got ",
                             scales_dims[0]);
    }
    if (scales_dims[1] != k_blocks) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'scales' dimension 1 should be equal to quantized_weight dimension 1, got ", scales_dims[1]);
    }
  } else {
    if (scales_dims.size() == 1 && scales_dims[0] == n * k_blocks) {
      // Backward compatibility.
      // This format of 1D format is deprecated. We will remove the support of this format in the future.
    } else {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'scales' shape is not compatible with quantized_weight. Expected: (",
          n, ", ", k_blocks, ")", "Got ", scales->Shape());
    }
  }

  if (zero_points != nullptr) {
    const auto& zero_points_dims = zero_points->Shape().GetDims();
    if (zero_points_dims.size() == 2) {
      if (zero_points_dims[0] != n) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'zero_points' dimension 0 should be N, got ",
                               zero_points_dims[0]);
      }

      if (zero_points->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        const int64_t zero_point_blob_size = ((k_blocks * bits + 7) / 8);
        if (zero_points_dims[1] != zero_point_blob_size) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT,
              "Input 'zero_points' dimension 1 should be equal to ((k_blocks * bits + 7) / 8), got ",
              zero_points_dims[1], ". Expected:", zero_point_blob_size);
        }
      } else {
        if (zero_points->GetElementType() != scales->GetElementType()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Input 'zero_points' and 'scales' should have the same data type when zero_points is not uint8");
        }

        if (zero_points_dims[1] != k_blocks) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT,
              "Input 'zero_points' dimension 1 dimension 1 should be equal to quantized_weight dimension 1, got ",
              zero_points_dims[1]);
        }
      }
    } else if (zero_points_dims.size() == 1) {
      // Backward compatibility.
      // This format of 1D format is deprecated. We will remove the support of this format in the future.
      if (zero_points->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        const int64_t zero_point_blob_size = ((k_blocks * bits + 7) / 8);
        if (zero_points_dims[0] != n * zero_point_blob_size) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT, "Input 'zero_points' shape is not expected: ", zero_points->Shape());
        }
      } else {
        if (zero_points_dims[0] != n * k_blocks) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT, "Input 'zero_points' shape is not expected: ", zero_points->Shape());
        }
      }
    } else {
      if (zero_points_dims.size() != 2) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT, "Input 'zero_points' is expected to have 2 dimensions:", zero_points->Shape());
      }
    }
  }

  if (group_index != nullptr) {
    // Group_index is deprecated. We will remove the support in the future.
    const auto& group_index_dims = group_index->Shape().GetDims();
    if (group_index_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'group_index' is expected to have 1 dimension, got ",
                             group_index_dims.size());
    }
    if (group_index_dims[0] != k && group_index_dims[0] != k_blocks * block_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'group_index' dimension 0 should be equal to K, or K padded to multiple of block_size, got ",
          group_index_dims[0]);
    }
  }

  if (bias != nullptr) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'bias' is expected to have 1 dimension, got ", bias_dims.size());
    }
    if (bias_dims[0] != n) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'bias' dimension 0 should be N, got ", bias_dims[0]);
    }
  }

  return Status::OK();
}

}  // namespace matmul_nbits_helper
}  // namespace contrib
}  // namespace onnxruntime
