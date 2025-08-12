// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/util/shape_checker.h"

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
                   int64_t bits) {
  // activation (A)
  // quantized_weight (B) : (N, k_blocks, blob_size), or null after prepacking.
  //                        k_blocks = (K + block_size - 1) / block_size
  //                        blob_size = block_size * bits / 8
  // scales               : (N, k_blocks)
  // zero_points          : (N, (k_blocks * bits + 7) / 8) for uint8, (N, k_blocks) for other types, or null
  // group_index          : (K) or (k_blocks * block_size), or null
  // bias                 : (N), or null
  // Note that scales and zero_points can be 1D for backward compatibility.
  if (bits != 2 && bits != 4 && bits != 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bits should be 2, 4 or 8, got ", bits);
  }

  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_size must be a power of 2, and >= 16. Got ", block_size);
  }

  int64_t k_blocks = (k + block_size - 1) / block_size;
  int64_t blob_size = block_size * bits / 8;

  ASSERT_TENSOR_SHAPE(quantized_weight, make_shape(n, k_blocks, blob_size));

  // 1D shape is for backward compatibility for existing models.
  ASSERT_TENSOR_SHAPE_2(scales, make_shape(n * k_blocks), make_shape(n, k_blocks));

  if (zero_points != nullptr) {
    if (zero_points->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      const int64_t zero_point_blob_size = (k_blocks * bits + 7) / 8;

      ASSERT_TENSOR_SHAPE_2(zero_points, make_shape(n * zero_point_blob_size), make_shape(n, zero_point_blob_size));
    } else {
      if (zero_points->GetElementType() != scales->GetElementType()) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Input 'zero_points' and 'scales' should have the same data type when zero_points is not uint8");
      }

      ASSERT_TENSOR_SHAPE_2(zero_points, make_shape(n * k_blocks), make_shape(n, k_blocks));
    }
  }

  // Group_index shall be 1D of K, or K padded to multiple of block_size
  ASSERT_TENSOR_SHAPE_2(group_index, make_shape(k), make_shape(k_blocks * block_size));

  ASSERT_TENSOR_SHAPE(bias, make_shape(n));

  return Status::OK();
}

}  // namespace matmul_nbits_helper
}  // namespace contrib
}  // namespace onnxruntime
