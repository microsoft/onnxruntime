// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_gqa_task.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_gqa_tc_sm80_task.cuh"

namespace onnxruntime::contrib::paged {

template <
    int NumThreads,
    int HeadSize,
    int PageSize,
    typename TI,
    typename TO,
    typename TKV,
    typename Worker,
    typename Config,
    typename KVConfig = DefaultKV>
struct PagedAttentionTaskSelector {
private:
  __device__ static constexpr auto select_task() {
    constexpr int RequestedNumQueriesPerCta = Config::NumQueriesPerCta;
    if constexpr (RequestedNumQueriesPerCta == 1) {
      return PagedAttentionTask<NumThreads, HeadSize, PageSize, TI, TO, TKV, Worker, Config, KVConfig>{};
    } else if constexpr (RequestedNumQueriesPerCta == 4) {
      if constexpr (std::is_same_v<TKV, float>) {
        return PagedGroupQueryAttentionTask<NumThreads, HeadSize, PageSize, TI, TO, TKV, Worker, Config, KVConfig>{};
      } else {
        return PagedGroupQueryAttentionTcSm80Task<NumThreads, HeadSize, PageSize, TI, TO, TKV, Worker, Config, KVConfig>{};
      }
    } else if constexpr (RequestedNumQueriesPerCta == 8) {
      return PagedGroupQueryAttentionTcSm80Task<NumThreads, HeadSize, PageSize, TI, TO, TKV, Worker, Config, KVConfig>{};
    } else {
      static_assert(always_false<Config>, "Unsupported TaskConfig");
      return;
    }
  }

public:
  using Task = decltype(PagedAttentionTaskSelector::select_task());
};

}  // namespace onnxruntime::contrib::paged
