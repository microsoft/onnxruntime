// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/runtime_optimization_record_container.h"

#include <algorithm>

#include "gsl/gsl"

#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"

namespace onnxruntime {

namespace fbs = experimental::fbs;

#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)
bool RuntimeOptimizationRecordContainer::AddRecord(const std::string& sat_key,
                                                   RuntimeOptimizationRecord&& runtime_optimization_record) {
  auto& optimizations = sat_to_optimizations_[sat_key];
  if (std::find(optimizations.begin(), optimizations.end(), runtime_optimization_record) != optimizations.end()) {
    return false;
  }

  optimizations.emplace_back(std::move(runtime_optimization_record));
  return true;
}
#endif

std::vector<RuntimeOptimizationRecord> RuntimeOptimizationRecordContainer::RemoveRecordsForKey(
    const std::string& sat_key) {
  std::vector<RuntimeOptimizationRecord> result{};
  if (auto it = sat_to_optimizations_.find(sat_key); it != sat_to_optimizations_.end()) {
    result = std::move(it->second);
    sat_to_optimizations_.erase(it);
  }
  return result;
}

static Status SaveRuntimeOptimizationRecordToOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const RuntimeOptimizationRecord& runtime_optimization_record,
    flatbuffers::Offset<fbs::RuntimeOptimizationRecord>& fbs_runtime_optimization_record) {
  const auto& nodes_to_optimize_indexes = runtime_optimization_record.nodes_to_optimize_indexes;

  const auto fbs_node_indexes = builder.CreateVector<uint32_t>(
      nodes_to_optimize_indexes.nodes.size(),
      [&](size_t i) { return gsl::narrow<uint32_t>(nodes_to_optimize_indexes.nodes[i]); });

  const auto fbs_nodes_to_optimize =
      fbs::CreateNodesToOptimizeIndexes(builder,
                                        fbs_node_indexes,
                                        nodes_to_optimize_indexes.num_inputs,
                                        nodes_to_optimize_indexes.num_outputs,
                                        nodes_to_optimize_indexes.variadic_input,
                                        nodes_to_optimize_indexes.variadic_output,
                                        nodes_to_optimize_indexes.num_variadic_inputs,
                                        nodes_to_optimize_indexes.num_variadic_outputs);

  fbs_runtime_optimization_record =
      fbs::CreateRuntimeOptimizationRecordDirect(builder,
                                                 runtime_optimization_record.selector_action_id.c_str(),
                                                 fbs_nodes_to_optimize,
                                                 &runtime_optimization_record.produced_node_kernel_def_hashes);

  return Status::OK();
}

Status RuntimeOptimizationRecordContainer::SaveToOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<FbsRuntimeOptimizationRecordContainer>& fbs_runtime_optimizations) const {
  std::vector<flatbuffers::Offset<fbs::RuntimeOptimizationRecordContainerEntry>> fbs_runtime_optimizations_vector;
  fbs_runtime_optimizations_vector.reserve(sat_to_optimizations_.size());
  for (const auto& [sat_name, records] : sat_to_optimizations_) {
    std::vector<flatbuffers::Offset<fbs::RuntimeOptimizationRecord>> fbs_records_vector;
    fbs_records_vector.reserve(records.size());
    for (const auto& record : records) {
      flatbuffers::Offset<fbs::RuntimeOptimizationRecord> fbs_record_offset;
      ORT_RETURN_IF_ERROR(SaveRuntimeOptimizationRecordToOrtFormat(builder, record, fbs_record_offset));
      fbs_records_vector.push_back(fbs_record_offset);
    }

    fbs_runtime_optimizations_vector.push_back(
        fbs::CreateRuntimeOptimizationRecordContainerEntryDirect(builder,
                                                                 sat_name.c_str(),
                                                                 &fbs_records_vector));
  }

  fbs_runtime_optimizations = builder.CreateVectorOfSortedTables(&fbs_runtime_optimizations_vector);
  return Status::OK();
}

#if defined(ENABLE_ORT_FORMAT_LOAD)
static Status LoadRuntimeOptimizationRecordFromOrtFormat(
    const fbs::RuntimeOptimizationRecord& fbs_runtime_optimization_record,
    RuntimeOptimizationRecord& runtime_optimization_record_out) {
  RuntimeOptimizationRecord runtime_optimization_record;

  experimental::utils::LoadStringFromOrtFormat(runtime_optimization_record.selector_action_id,
                                               fbs_runtime_optimization_record.selector_action_id());

  auto& nodes_to_optimize_indexes = runtime_optimization_record.nodes_to_optimize_indexes;
  if (const auto* fbs_nodes_to_optimize_indexes = fbs_runtime_optimization_record.nodes_to_optimize_indexes()) {
    if (const auto* fbs_node_indexes = fbs_nodes_to_optimize_indexes->node_indexes()) {
      nodes_to_optimize_indexes.nodes = [&]() {
        std::vector<NodeIndex> result;
        result.reserve(fbs_node_indexes->size());
        std::transform(fbs_node_indexes->begin(), fbs_node_indexes->end(), std::back_inserter(result),
                       [](const auto idx) { return static_cast<NodeIndex>(idx); });
        return result;
      }();
    }

    nodes_to_optimize_indexes.num_inputs = fbs_nodes_to_optimize_indexes->num_inputs();
    nodes_to_optimize_indexes.num_outputs = fbs_nodes_to_optimize_indexes->num_outputs();
    nodes_to_optimize_indexes.variadic_input = fbs_nodes_to_optimize_indexes->has_variadic_input();
    nodes_to_optimize_indexes.variadic_output = fbs_nodes_to_optimize_indexes->has_variadic_output();
    nodes_to_optimize_indexes.num_variadic_inputs = fbs_nodes_to_optimize_indexes->num_variadic_inputs();
    nodes_to_optimize_indexes.num_variadic_outputs = fbs_nodes_to_optimize_indexes->num_variadic_outputs();
  }

  if (const auto* fbs_kernel_def_hashes = fbs_runtime_optimization_record.produced_node_kernel_def_hashes()) {
    runtime_optimization_record.produced_node_kernel_def_hashes = std::vector<uint64_t>(fbs_kernel_def_hashes->begin(),
                                                                                        fbs_kernel_def_hashes->end());
  }

  runtime_optimization_record_out = std::move(runtime_optimization_record);
  return Status::OK();
}

Status RuntimeOptimizationRecordContainer::LoadFromOrtFormat(
    const FbsRuntimeOptimizationRecordContainer& fbs_runtime_optimizations) {
  SatToOptimizationRecordsMap sat_to_optimizations;
  for (const auto* fbs_runtime_optimization : fbs_runtime_optimizations) {
    if (!fbs_runtime_optimization) continue;

    std::string sat_name;
    experimental::utils::LoadStringFromOrtFormat(sat_name, fbs_runtime_optimization->optimizer_name());

    std::vector<RuntimeOptimizationRecord> records;
    if (const auto* fbs_runtime_optimization_records = fbs_runtime_optimization->runtime_optimization_records()) {
      records.reserve(fbs_runtime_optimization_records->size());
      for (const auto* fbs_runtime_optimization_record : *fbs_runtime_optimization_records) {
        if (!fbs_runtime_optimization_record) continue;

        RuntimeOptimizationRecord runtime_optimization_record;
        ORT_RETURN_IF_ERROR(LoadRuntimeOptimizationRecordFromOrtFormat(*fbs_runtime_optimization_record,
                                                                       runtime_optimization_record));
        records.emplace_back(std::move(runtime_optimization_record));
      }
    }

    ORT_RETURN_IF_NOT(sat_to_optimizations.emplace(sat_name, std::move(records)).second,
                      "Attempting to load runtime optimization records for a previously loaded optimizer: ", sat_name);
  }

  sat_to_optimizations_ = std::move(sat_to_optimizations);
  return Status::OK();
}
#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

}  // namespace onnxruntime
