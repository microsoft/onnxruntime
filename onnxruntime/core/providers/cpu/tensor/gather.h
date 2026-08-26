// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "gatherbase.h"

namespace onnxruntime {

namespace gather_internal {

struct GatherCopyIndex {
  int64_t batch;
  int64_t index;
};

ptrdiff_t GetGatherCopyWorkCount(int64_t num_batches, int64_t num_indices);
GatherCopyIndex GetGatherCopyIndex(ptrdiff_t index, int64_t num_indices);

template <typename Callback>
void ForEachGatherCopyIndex(ptrdiff_t first, ptrdiff_t last, int64_t num_indices, const Callback& callback) {
  for (ptrdiff_t index = first; index < last; ++index) {
    const auto gather_index = GetGatherCopyIndex(index, num_indices);
    callback(gather_index.batch, gather_index.index);
  }
}

}  // namespace gather_internal

class Gather : public OpKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : OpKernel(info), GatherBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
