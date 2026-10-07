#pragma once

#include <cstdint>
#include <string>
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime::contrib::selection_merge {

inline int64_t ReadCapacity(const OpKernelInfo& info) {
  std::string policy;
  int64_t capacity;
  ORT_THROW_IF_ERROR(info.GetAttr("policy_mode", &policy));
  ORT_ENFORCE(policy == "append_range", "SparseAttentionSelectionMerge only supports append_range");
  ORT_THROW_IF_ERROR(info.GetAttr("max_output_entries", &capacity));
  ORT_ENFORCE(capacity > 0 && capacity <= (int64_t{1} << 30),
              "max_output_entries must be in [1, 2^30]");
  return capacity;
}

struct Dimensions {
  int32_t rows;
  int32_t capacity;
  int32_t queries;
  int32_t hash_capacity;
};

inline Status Validate(const Tensor* base, const Tensor* counts, const Tensor* rows,
                       const Tensor* starts, const Tensor* ends, int64_t output_capacity,
                       Dimensions& dimensions) {
  ORT_RETURN_IF_NOT(base && counts && rows && starts && ends, "All append_range inputs are required");
  ORT_RETURN_IF_NOT(base->Shape().NumDimensions() == 2 && rows->Shape().NumDimensions() == 1,
                    "base_indices must be rank 2 and base_row_indices rank 1");
  const int64_t cached_rows = base->Shape()[0];
  const int64_t base_capacity = base->Shape()[1];
  const int64_t queries = rows->Shape()[0];
  ORT_RETURN_IF(cached_rows > INT32_MAX || base_capacity > INT32_MAX || queries > INT32_MAX,
                "Selection dimensions must fit int32");
  ORT_RETURN_IF_NOT(counts->Shape() == TensorShape({cached_rows}) &&
                        starts->Shape() == rows->Shape() && ends->Shape() == rows->Shape(),
                    "Counts and range tensors must match their row dimensions");
  int64_t hash_capacity = 1;
  while (hash_capacity < output_capacity) hash_capacity *= 2;
  dimensions = {static_cast<int32_t>(cached_rows), static_cast<int32_t>(base_capacity),
                static_cast<int32_t>(queries), static_cast<int32_t>(hash_capacity)};
  return Status::OK();
}

}  // namespace onnxruntime::contrib::selection_merge