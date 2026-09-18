// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_workspace_bounds.h"

#include <algorithm>
#include <array>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr GQAWorkspaceStatus Invalid(const char* message) noexcept {
  return {GQAWorkspaceError::InvalidArgument, message};
}

constexpr GQAWorkspaceStatus Unavailable(const char* message) noexcept {
  return {GQAWorkspaceError::Unavailable, message};
}

GQAWorkspaceStatus Mul(size_t a, size_t b, size_t& out) noexcept {
  return CheckedGQAWorkspaceMultiply(a, b, out);
}

GQAWorkspaceStatus Add(size_t a, size_t b, size_t& out) noexcept {
  return CheckedGQAWorkspaceAdd(a, b, out);
}

GQAWorkspaceStatus Mul4(size_t a, size_t b, size_t c, size_t d, size_t& out) noexcept {
  auto status = Mul(a, b, out);
  if (!status.IsOK()) return status;
  status = Mul(out, c, out);
  if (!status.IsOK()) return status;
  return Mul(out, d, out);
}

GQAWorkspaceStatus Append(size_t bytes, size_t& cursor) noexcept {
  if (bytes == 0) return {};
  size_t offset = 0;
  auto status = CheckedGQAWorkspaceAlign(cursor, kGQAWorkspaceAlignment, offset);
  if (!status.IsOK()) return status;
  return Add(offset, bytes, cursor);
}

GQAWorkspaceStatus Compose(size_t preparation, size_t backend, size_t& total) noexcept {
  if (backend == 0) {
    total = preparation;
    return {};
  }
  size_t offset = 0;
  auto status = CheckedGQAWorkspaceAlign(preparation, kGQAWorkspaceAlignment, offset);
  if (!status.IsOK()) return status;
  return Add(offset, backend, total);
}

GQAWorkspaceStatus EffectiveKvLengthBound(const GQAWorkspaceBounds& bounds,
                                          int64_t sequence_length,
                                          int64_t& effective_kv_length) noexcept {
  effective_kv_length = bounds.present_kv_cache_capacity_bound;
  if (!bounds.is_windowed_kv_cache || sequence_length == 1) return {};

  size_t staged_capacity = 0;
  auto status = Add(static_cast<size_t>(bounds.present_kv_cache_capacity_bound),
                    static_cast<size_t>(sequence_length), staged_capacity);
  if (!status.IsOK()) return status;
  if (staged_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Invalid("Windowed GQA effective KV length bound must fit int32.");
  }

  effective_kv_length = static_cast<int64_t>(staged_capacity);
  return {};
}

GQAWorkspaceProblem MakeProblem(const GQAWorkspaceBounds& bounds,
                                int64_t sequence_length,
                                int64_t head_size,
                                bool first_prompt) noexcept {
  GQAWorkspaceProblem problem;
  problem.qkv_element_size = bounds.qkv_element_size;
  problem.cache_element_size = bounds.cache_element_size;
  problem.batch_size = bounds.batch_size_bound;
  problem.sequence_length = sequence_length;
  problem.num_heads = bounds.num_heads;
  problem.kv_num_heads = bounds.kv_num_heads;
  problem.head_size = head_size;
  problem.present_kv_cache_capacity = bounds.present_kv_cache_capacity_bound;
  problem.kv_cache_bit_width = bounds.kv_cache_bit_width;
  problem.k_quantization = bounds.k_quantization;
  problem.v_quantization = bounds.v_quantization;
  problem.is_windowed_kv_cache = bounds.is_windowed_kv_cache;
  problem.is_first_prompt = first_prompt;
  problem.do_rotary = bounds.do_rotary;
  problem.is_packed_qkv = bounds.is_packed_qkv;
  problem.use_qk_norm = bounds.use_qk_norm;
  return problem;
}

int64_t LargestMultipleOfEight(int64_t bound, int64_t limit) noexcept {
  return (std::min(bound, limit) / 8) * 8;
}

GQAWorkspaceStatus FlashBackendEnvelope(const GQAWorkspaceProblem& problem,
                                        const GQAWorkspaceBounds& bounds,
                                        int64_t effective_kv_length,
                                        bool fast_decode,
                                        size_t& bytes) noexcept {
  // flash::num_splits_heuristic never selects more than this. Unlike evaluating
  // the heuristic at max KV length, this remains sound across its discontinuities.
  const size_t head_size = static_cast<size_t>(problem.head_size);
  const size_t block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  size_t kv_length = static_cast<size_t>(effective_kv_length);
  if (fast_decode && bounds.local_window_size > 0) {
    kv_length = std::min(kv_length, static_cast<size_t>(bounds.local_window_size));
  }
  size_t numerator = 0;
  auto status = Add(kv_length, block_n - 1, numerator);
  if (!status.IsOK()) return status;
  const size_t max_splits = std::min(
      {size_t{128}, static_cast<size_t>(bounds.multi_processor_count), numerator / block_n});

  size_t lse_bytes = 0;
  status = Mul4(static_cast<size_t>(problem.batch_size),
                static_cast<size_t>(problem.num_heads),
                static_cast<size_t>(problem.sequence_length), sizeof(float), lse_bytes);
  if (!status.IsOK()) return status;
  bytes = 0;
  status = Append(lse_bytes, bytes);
  if (!status.IsOK() || max_splits <= 1) return status;

  size_t split_elements = 0;
  status = Mul4(max_splits, static_cast<size_t>(problem.batch_size),
                static_cast<size_t>(problem.num_heads),
                static_cast<size_t>(problem.sequence_length), split_elements);
  if (!status.IsOK()) return status;
  size_t split_lse = 0;
  status = Mul(split_elements, sizeof(float), split_lse);
  if (!status.IsOK()) return status;
  status = Append(split_lse, bytes);
  if (!status.IsOK()) return status;
  size_t rounded_head_size = 0;
  status = CheckedGQAWorkspaceAlign(head_size, 32, rounded_head_size);
  if (!status.IsOK()) return status;
  size_t output = 0;
  status = Mul4(split_elements, rounded_head_size, sizeof(float), 1, output);
  if (!status.IsOK()) return status;
  return Append(output, bytes);
}

GQAWorkspaceStatus XqaBackendEnvelope(const GQAWorkspaceProblem& problem,
                                      const GQAWorkspaceBounds& bounds,
                                      size_t& bytes) noexcept {
  // Route eligibility is established by the graph adapter. This envelope
  // intentionally bounds GetXQAScratchSize without reapplying recipe guards.
  size_t sequence_count = 0;
  auto status = Mul(static_cast<size_t>(problem.batch_size),
                    static_cast<size_t>(problem.kv_num_heads), sequence_count);
  if (!status.IsOK()) return status;
  const size_t subsequences =
      std::max(sequence_count, static_cast<size_t>(bounds.multi_processor_count));
  size_t semaphore = 0;
  status = Mul(sequence_count, sizeof(uint32_t), semaphore);
  if (!status.IsOK()) return status;
  status = CheckedGQAWorkspaceAlign(semaphore, 128, bytes);
  if (!status.IsOK()) return status;
  size_t rows = 0;
  status = Mul(128, subsequences, rows);
  if (!status.IsOK()) return status;
  status = Add(bytes, rows, bytes);
  if (!status.IsOK()) return status;
  // Alignment phase varies with sequence_count. Add the maximum possible
  // padding instead of evaluating one endpoint.
  status = Add(bytes, 127, bytes);
  if (!status.IsOK()) return status;
  status = Add(bytes, rows, bytes);
  if (!status.IsOK()) return status;

  const int64_t group_size = problem.num_heads / problem.kv_num_heads;
  const size_t tile = group_size <= 8 ? 8 : (group_size <= 16 ? 16 : 32);
  size_t vector_bytes = 0;
  status = Mul4(static_cast<size_t>(problem.head_size), tile, 2, 1, vector_bytes);
  if (!status.IsOK()) return status;
  status = Add(bytes, vector_bytes - 1, bytes);
  if (!status.IsOK()) return status;
  size_t output = 0;
  status = Mul(vector_bytes, subsequences, output);
  if (!status.IsOK()) return status;
  status = Add(bytes, output, bytes);
  if (!status.IsOK()) return status;

  if (problem.do_rotary) {
    size_t q = 0;
    status = Mul4(static_cast<size_t>(problem.batch_size),
                  static_cast<size_t>(problem.num_heads),
                  static_cast<size_t>(problem.head_size),
                  problem.qkv_element_size, q);
    if (!status.IsOK()) return status;
    status = CheckedGQAWorkspaceAlign(q, kGQAWorkspaceAlignment, q);
    if (!status.IsOK()) return status;
    status = Add(bytes, q, bytes);
    if (!status.IsOK()) return status;
    size_t k = 0;
    status = Mul4(static_cast<size_t>(problem.batch_size),
                  static_cast<size_t>(problem.kv_num_heads),
                  static_cast<size_t>(problem.head_size),
                  problem.qkv_element_size, k);
    if (!status.IsOK()) return status;
    status = CheckedGQAWorkspaceAlign(k, kGQAWorkspaceAlignment, k);
    if (!status.IsOK()) return status;
    status = Add(bytes, k, bytes);
    if (!status.IsOK()) return status;
  }
  if (bounds.xqa_head_sink_storage == GQAXqaHeadSinkStorage::DynamicConversion) {
    size_t sink = 0;
    status = Mul(static_cast<size_t>(problem.num_heads), sizeof(float), sink);
    if (!status.IsOK()) return status;
    status = CheckedGQAWorkspaceAlign(sink, kGQAWorkspaceAlignment, sink);
    if (!status.IsOK()) return status;
    status = Add(bytes, sink, bytes);
  }
  return status;
}

}  // namespace

GQAWorkspaceAggregate GetGQAWorkspaceAggregateForBounds(
    const GQAWorkspaceBounds& bounds) noexcept {
  GQAWorkspaceAggregate aggregate;
  if (bounds.qkv_element_size != 2 ||
      (bounds.cache_element_size != 1 && bounds.cache_element_size != 2) ||
      bounds.batch_size_bound <= 0 || bounds.sequence_length_bound <= 0 ||
      bounds.num_heads <= 0 || bounds.kv_num_heads <= 0 ||
      bounds.num_heads % bounds.kv_num_heads != 0 ||
      bounds.head_size_bound < 8 || bounds.present_kv_cache_capacity_bound <= 0 ||
      bounds.multi_processor_count <= 0 ||
      (!bounds.prompt_reachable && !bounds.decode_reachable)) {
    aggregate.status = Invalid("GQA workspace bounds are incomplete or invalid.");
    return aggregate;
  }
  if (HasGQAReachableBackend(bounds.reachable_backends, GQAReachableBackend::Cudnn)) {
    aggregate.status = Unavailable(
        "A reachable cuDNN GQA route has no graph-free workspace oracle.");
    return aggregate;
  }
  const bool head_sink_is_or_may_be_prepacked =
      bounds.head_sink_may_be_prepacked ||
      bounds.xqa_head_sink_storage == GQAXqaHeadSinkStorage::PrepackedFp32;
  if (head_sink_is_or_may_be_prepacked) {
    aggregate.status = Mul(
        static_cast<size_t>(bounds.num_heads), sizeof(float),
        aggregate.persistent_prepack_bytes);
    if (!aggregate.status.IsOK()) return aggregate;
    aggregate.status = Mul(
        static_cast<size_t>(bounds.num_heads), bounds.qkv_element_size,
        aggregate.initialization_scratch_bytes);
    if (!aggregate.status.IsOK()) return aggregate;
  }

  const auto retain = [&aggregate](GQAReachableBackend backend,
                                   GQAWorkspaceStatus status,
                                   size_t bytes) {
    if (!aggregate.status.IsOK()) return;
    if (!status.IsOK()) {
      aggregate.status = status;
      return;
    }
    aggregate.total_workspace_bytes = std::max(aggregate.total_workspace_bytes, bytes);
    aggregate.sized_backends = aggregate.sized_backends | backend;
  };

  const std::array<int64_t, 2> sequence_candidates{
      1, bounds.sequence_length_bound};
  const auto size_complete = [&](GQABackend backend, int64_t head_size,
                                 int64_t sequence_length, bool first_prompt) {
    int64_t effective_kv_length = 0;
    const auto effective_status =
        EffectiveKvLengthBound(bounds, sequence_length, effective_kv_length);
    if (!effective_status.IsOK()) {
      GQACompleteWorkspaceResult result;
      result.status = effective_status;
      return result;
    }
    GQAConcreteRoute route;
    route.backend = backend;
    route.preparation.preprocess_mode =
        backend == GQABackend::MemoryEfficient ? GQAPreprocessMode::MemoryEfficient
                                               : GQAPreprocessMode::Unfused;
    route.unfused.total_sequence_length = effective_kv_length;
    return GetGQACompleteWorkspaceRecipe(
        MakeProblem(bounds, sequence_length, head_size, first_prompt), route);
  };

  const int64_t general_head = LargestMultipleOfEight(
      bounds.head_size_bound, std::numeric_limits<int32_t>::max());
  for (GQAReachableBackend backend :
       {GQAReachableBackend::MemoryEfficient, GQAReachableBackend::Unfused}) {
    if (!HasGQAReachableBackend(bounds.reachable_backends, backend)) continue;
    const int64_t head = backend == GQAReachableBackend::MemoryEfficient
                             ? LargestMultipleOfEight(bounds.head_size_bound, 1024)
                             : general_head;
    for (int64_t sequence : sequence_candidates) {
      if (sequence == 1 && !bounds.decode_reachable && bounds.sequence_length_bound != 1) continue;
      if (sequence > 1 && !bounds.prompt_reachable) continue;
      for (bool first : {false, true}) {
        if (first && !bounds.prompt_reachable) continue;
        const auto result = size_complete(
            backend == GQAReachableBackend::MemoryEfficient
                ? GQABackend::MemoryEfficient
                : GQABackend::Unfused,
            head, sequence, first);
        retain(backend, result.status, result.recipe.total_workspace_bytes);
      }
    }
  }

  const auto size_flash = [&](bool fast, int64_t sequence, bool first) {
    int64_t effective_kv_length = 0;
    auto status = EffectiveKvLengthBound(bounds, sequence, effective_kv_length);
    if (!status.IsOK()) {
      return std::pair<GQAWorkspaceStatus, size_t>{status, 0};
    }
    const int64_t head = LargestMultipleOfEight(bounds.head_size_bound, 256);
    const auto problem = MakeProblem(bounds, sequence, head, first);
    GQAPreparationRoute prep_route;
    prep_route.preprocess_mode = GQAPreprocessMode::Flash;
    prep_route.use_flash_attention_fast_decode = fast;
    const auto prep = GetGQAPreparationRecipe(problem, prep_route);
    if (!prep.status.IsOK()) return std::pair<GQAWorkspaceStatus, size_t>{prep.status, 0};
    size_t backend = 0;
    status = FlashBackendEnvelope(problem, bounds, effective_kv_length, fast, backend);
    size_t total = 0;
    if (status.IsOK()) status = Compose(prep.recipe.total_preparation_bytes, backend, total);
    return std::pair<GQAWorkspaceStatus, size_t>{status, total};
  };
  if (HasGQAReachableBackend(bounds.reachable_backends, GQAReachableBackend::Flash)) {
    for (int64_t sequence : sequence_candidates) {
      if (sequence == 1 && !bounds.decode_reachable && bounds.sequence_length_bound != 1) continue;
      if (sequence > 1 && !bounds.prompt_reachable) continue;
      for (bool first : {false, true}) {
        if (first && !bounds.prompt_reachable) continue;
        const auto result = size_flash(false, sequence, first);
        retain(GQAReachableBackend::Flash, result.first, result.second);
      }
    }
  }
  if (HasGQAReachableBackend(bounds.reachable_backends,
                             GQAReachableBackend::FlashFastDecode)) {
    // The current graph adapter rejects non-windowed inputs, so this route is
    // retained for graph-free callers and a future non-windowed adapter.
    // FlashBackendEnvelope reserves the maximum reachable split count instead
    // of evaluating the non-monotonic split heuristic. Its storage and the
    // preparation storage are nondecreasing in sequence length, so Smax covers
    // every dynamic S in the bounded domain.
    const auto result = size_flash(true, bounds.sequence_length_bound, false);
    retain(GQAReachableBackend::FlashFastDecode, result.first, result.second);
  }

  if (HasGQAReachableBackend(bounds.reachable_backends, GQAReachableBackend::Xqa)) {
    bool found_head = false;
    for (int64_t head : {int64_t{64}, int64_t{128}, int64_t{256}}) {
      if (head > bounds.head_size_bound) continue;
      found_head = true;
      const auto problem = MakeProblem(bounds, 1, head, false);
      GQAPreparationRoute route;
      route.preprocess_mode = GQAPreprocessMode::Xqa;
      const auto prep = GetGQAPreparationRecipe(problem, route);
      size_t backend = 0;
      auto status = prep.status;
      if (status.IsOK()) status = XqaBackendEnvelope(problem, bounds, backend);
      size_t total = 0;
      if (status.IsOK()) status = Compose(prep.recipe.total_preparation_bytes, backend, total);
      retain(GQAReachableBackend::Xqa, status, total);
    }
    if (!found_head) {
      aggregate.status = Unavailable("No XQA head geometry is reachable in the supplied bounds.");
    }
  }

  if (aggregate.status.IsOK() &&
      aggregate.sized_backends == GQAReachableBackend::None) {
    aggregate.status = Unavailable("No GQA backend route can be sized.");
  }
  return aggregate;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
