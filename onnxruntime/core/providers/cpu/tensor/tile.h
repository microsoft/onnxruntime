// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace TileOp {
// Function to determine if the tiling operation is just multiple copies
// of the input data buffer
// E.g.: input_shape: [1, 1, 256 * 50]
// repeats: [1, 200, 1]
// output shape: [1, 200, 256 * 50]

bool IsTileMemcpy(const TensorShape& input_shape,
                  const int64_t* repeats,
                  size_t rank,
                  /*out*/ size_t& num_of_copies);
}  // namespace TileOp

struct Tile : OpKernel {
  explicit Tile(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};

}  // namespace onnxruntime
