// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/capability_planning.h"

namespace onnxruntime {

class GraphViewer;

enum class DeclaredStateLifetime {
  Generator,
  Session,
};

struct DeclaredStatePaging {
  int64_t block_size = 0;
  int64_t num_blocks = 0;
  std::string block_table;
};

struct DeclaredStateSpec {
  std::string id;
  std::string input;
  std::string output;
  InlinedVector<int64_t> layers;
  int64_t capacity = 0;
  CapabilityInputOutputAlias alias = CapabilityInputOutputAlias::Unknown;
  std::optional<int64_t> window;
  DeclaredStateLifetime lifetime = DeclaredStateLifetime::Generator;
  std::optional<DeclaredStatePaging> paging;
};

struct DeclaredStateContract {
  int64_t version = 1;
  int64_t max_batch_size = 0;
  int64_t max_sequence_length = 0;
  int64_t max_live_instances = 0;
  InlinedVector<DeclaredStateSpec, 1> states;
};

struct ResolvedDeclaredStateTensor {
  std::string id;
  int64_t layer = 0;
  std::string input;
  std::string output;
  size_t consumer_node_index = 0;
  size_t input_index = 0;
  size_t output_index = 0;
  int32_t element_type = 0;
  TensorShapeVector max_shape;
  size_t size_bound_bytes = 0;
  CapabilityInputOutputAlias alias = CapabilityInputOutputAlias::Unknown;
  int64_t window = 0;
  bool sliding_window_cache = false;
  DeclaredStateLifetime lifetime = DeclaredStateLifetime::Generator;
  std::optional<DeclaredStatePaging> paging;
};

struct ResolvedDeclaredStateBlockTable {
  std::string input;
  int64_t block_size = 0;
  TensorShapeVector max_shape;
  size_t size_bound_bytes = 0;
  InlinedVector<size_t> consumer_node_indices;
};

struct ResolvedDeclaredStateContract {
  std::string graph_id;
  uint64_t graph_revision = 0;
  int64_t max_batch_size = 0;
  int64_t max_sequence_length = 0;
  int64_t max_live_instances = 0;
  InlinedVector<ResolvedDeclaredStateTensor, 1> states;
  InlinedVector<ResolvedDeclaredStateBlockTable> block_tables;
};

struct DeclaredStateBackingBounds {
  size_t cache_bytes = 0;
  size_t block_table_bytes = 0;
  size_t total_bytes = 0;
};

// V1: dense BNSH GQA. V2 also describes SEPARATE, unquantized PagedAttention pools and block tables.
// All helpers leave their output unchanged on failure. They do not admit or allocate a plan.
Status ParseDeclaredStateContract(std::string_view json, DeclaredStateContract& contract);

// Requires a resolved, unfiltered graph. No graph views are retained. Re-resolve after graph changes.
Status ResolveDeclaredStateContract(const DeclaredStateContract& contract, const GraphViewer& graph,
                                    std::string_view graph_id, uint64_t graph_revision,
                                    ResolvedDeclaredStateContract& resolved);

// Cache/pool backing plus unique graph block tables; excludes other metadata, workspace, and allocator overhead.
Status ComputeDeclaredStateBackingBounds(const ResolvedDeclaredStateContract& contract,
                                         DeclaredStateBackingBounds& bounds);

// Apply only to an unpublished query for the same graph/revision, then publish a fresh const snapshot.
// Lifetime, ownership, instance count, and placement policy stay in ORT, not in the EP query.
Status ProjectDeclaredStateToPlanningQuery(const ResolvedDeclaredStateContract& contract,
                                           CapabilityPlanningQuery& query);

}  // namespace onnxruntime
