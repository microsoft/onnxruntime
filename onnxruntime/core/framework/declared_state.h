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

struct DeclaredStateSpec {
  std::string id;
  std::string input;
  std::string output;
  InlinedVector<int64_t> layers;
  int64_t capacity = 0;
  CapabilityInputOutputAlias alias = CapabilityInputOutputAlias::Unknown;
  std::optional<int64_t> window;
  DeclaredStateLifetime lifetime = DeclaredStateLifetime::Generator;
};

struct DeclaredStateContract {
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
};

struct ResolvedDeclaredStateContract {
  std::string graph_id;
  uint64_t graph_revision = 0;
  int64_t max_batch_size = 0;
  int64_t max_sequence_length = 0;
  int64_t max_live_instances = 0;
  InlinedVector<ResolvedDeclaredStateTensor, 1> states;
};

// Draft v1: dense BNSH GQA state, literal names or one {layer} token, follow_consumer only.
// All helpers leave their output unchanged on failure. They do not admit or allocate a plan.
Status ParseDeclaredStateContract(std::string_view json, DeclaredStateContract& contract);

// Requires a resolved, unfiltered graph. No graph views are retained. Re-resolve after graph changes.
Status ResolveDeclaredStateContract(const DeclaredStateContract& contract, const GraphViewer& graph,
                                    std::string_view graph_id, uint64_t graph_revision,
                                    ResolvedDeclaredStateContract& resolved);

// Apply only to an unpublished query for the same graph/revision, then publish a fresh const snapshot.
// Lifetime, ownership, instance count, and placement policy stay in ORT, not in the EP query.
Status ProjectDeclaredStateToPlanningQuery(const ResolvedDeclaredStateContract& contract,
                                           CapabilityPlanningQuery& query);

}  // namespace onnxruntime
