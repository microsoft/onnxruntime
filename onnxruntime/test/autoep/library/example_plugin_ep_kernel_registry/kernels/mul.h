// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "base.h"
#include "../../plugin_ep_utils.h"
#include "../ep_allocator.h"

class Mul : public BaseKernelImpl {
 private:
  struct PrivateTag {};

  struct PackedWeightInfo {
    Ort::ConstMemoryInfo mem_info{nullptr};
    std::vector<int64_t> shape;
    ONNXTensorElementDataType elem_type;
    size_t num_bytes;
    AllocationUniquePtr data{};  // raw bytes
  };

 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Mul>& kernel);
  Mul(const OrtKernelInfo* info, void* state, PrivateTag);

 private:
  OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) override;
  OrtStatus* DoPrePackWeight(const OrtValue* tensor, int input_index, OrtAllocator* alloc,
                             OrtSharedPrePackedWeightCache* prepacked_weight_cache, /*out*/ bool& is_packed) override;
  OrtStatus* DoSetSharedPrePackedWeight(const void* const* buffer_data_ptrs, size_t num_buffers,
                                        int input_index) override;

  std::optional<PackedWeightInfo> packed_weight_1_info_ = std::nullopt;
};
