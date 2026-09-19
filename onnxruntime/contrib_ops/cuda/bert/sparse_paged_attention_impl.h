// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class SparseAttentionMode {
  kSelectedOnly,
  kLocalPlusSelected,
};

enum class SelectedKvSource {
  kMain,
  kAuxiliary,
};

template <typename T, typename TCACHE>
Status SparseQkvToContext(
    const cudaDeviceProp& device_prop,
    Stream* stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    const int* selected_indices,
    const int* selected_counts,
    int max_selected_entries,
    const T* auxiliary_key,
    const T* auxiliary_value,
    const int* auxiliary_lengths,
    int auxiliary_capacity,
    int auxiliary_num_heads,
    SparseAttentionMode attention_mode,
    SelectedKvSource selected_kv_source,
    bool auxiliary_kv_shared);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
