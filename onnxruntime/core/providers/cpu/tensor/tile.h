// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#endif

#include "core/common/narrow.h"

namespace onnxruntime {

namespace TileOp {
// Function to determine if the tiling operation is just multiple copies
// of the input data buffer
// E.g.: input_shape: [1, 1, 256 * 50]
// repeats: [1, 200, 1]
// output shape: [1, 200, 256 * 50]

// As a slight extension, it also supports "batched" multiple copies of the input data buffer
// (`is_batched_memcpy` will be set to true)
// E.g.: input_shape: [5, 1, 256 * 50]
// repeats: [1, 200, 1]
// output shape: [5, 200, 256 * 50]

// Repeating the batch is also supported
// E.g.: input_shape: [5, 1, 256 * 50]
// repeats: [2, 200, 1]
// output shape: [10, 200, 256 * 50]

#ifdef SHARED_PROVIDER
bool IsTileMemcpy(const TensorShape& input_shape,
                  const int64_t* repeats,
                  size_t rank,
                  /*out*/ bool& is_batched_memcpy,
                  /*out*/ size_t& num_of_elements_per_batch,
                  /*out*/ size_t& num_of_copies_per_batch,
                  /*out*/ size_t& num_of_batch_copies);
#else
inline bool IsTileMemcpy(const TensorShape& input_shape,
                         const int64_t* repeats,
                         size_t rank,
                         /*out*/ bool& is_batched_memcpy,
                         /*out*/ size_t& num_of_elements_per_batch,
                         /*out*/ size_t& num_of_copies_per_batch,
                         /*out*/ size_t& num_of_batch_copies) {
  for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
    if (repeats[i] != 1) {
      if (input_shape.SizeToDimension(onnxruntime::narrow<size_t>(i)) == 1) {
        num_of_copies_per_batch = 1;
        for (int64_t j = 0; j <= i; ++j) {
          num_of_copies_per_batch *= onnxruntime::narrow<size_t>(repeats[onnxruntime::narrow<size_t>(j)]);
        }
        is_batched_memcpy = false;
        return true;
      } else if (i == 1) {
        num_of_elements_per_batch = static_cast<size_t>(input_shape.SizeFromDimension(1));
        num_of_copies_per_batch = onnxruntime::narrow<size_t>(repeats[onnxruntime::narrow<size_t>(i)]);
        num_of_batch_copies = onnxruntime::narrow<size_t>(repeats[0]);
        is_batched_memcpy = true;
        return true;
      } else {
        break;
      }
    }
  }
  return false;
}
#endif  // SHARED_PROVIDER
}  // namespace TileOp

struct Tile : OpKernel {
  explicit Tile(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};

}  // namespace onnxruntime
