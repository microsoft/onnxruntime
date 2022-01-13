// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)

#include "core/graph/runtime_optimization_record_container.h"

#include <algorithm>

#include "gsl/gsl"

#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"

namespace onnxruntime {

#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)
bool RuntimeOptimizationRecordContainer::RecordExists(const std::string& optimizer_name,
                                                      const std::string& action_id,
                                                      const NodesToOptimizeIndices& nodes_to_optimize_indices) const {
  const auto it = optimizer_name_to_records_.find(optimizer_name);
  if (it == optimizer_name_to_records_.end()) return false;

  const auto& records = it->second;
  return std::find_if(records.begin(), records.end(),
                      [&](const RuntimeOptimizationRecord& record) {
                        return record.action_id == action_id &&
                               record.nodes_to_optimize_indices == nodes_to_optimize_indices;
                      }) != records.end();
}

void RuntimeOptimizationRecordContainer::AddRecord(const std::string& optimizer_name,
                                                   RuntimeOptimizationRecord&& runtime_optimization_record) {
  auto& optimizations = optimizer_name_to_records_[optimizer_name];
  optimizations.emplace_back(std::move(runtime_optimization_record));
}
#endif

std::vector<RuntimeOptimizationRecord> RuntimeOptimizationRecordContainer::RemoveRecordsForOptimizer(
    const std::string& optimizer_name) {
  std::vector<RuntimeOptimizationRecord> records{};
  if (auto it = optimizer_name_to_records_.find(optimizer_name); it != optimizer_name_to_records_.end()) {
    records = std::move(it->second);
    optimizer_name_to_records_.erase(it);
  }
  return records;
}

static Status SaveRuntimeOptimizationRecordToOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const RuntimeOptimizationRecord& runtime_optimization_record,
    flatbuffers::Offset<fbs::RuntimeOptimizationRecord>& fbs_runtime_optimization_record) {
  const auto& nodes_to_optimize_indices = runtime_optimization_record.nodes_to_optimize_indices;

  const auto fbs_node_indices = builder.CreateVector<uint32_t>(
      nodes_to_optimize_indices.nodes.size(),
      [&](size_t i) { return gsl::narrow<uint32_t>(nodes_to_optimize_indices.nodes[i]); });

  const auto fbs_nodes_to_optimize =
      fbs::CreateNodesToOptimizeIndices(builder,
                                        fbs_node_indices,
                                        nodes_to_optimize_indices.num_inputs,
                                        nodes_to_optimize_indices.num_outputs,
                                        nodes_to_optimize_indices.variadic_input,
                                        nodes_to_optimize_indices.variadic_output,
                                        nodes_to_optimize_indices.num_variadic_inputs,
                                        nodes_to_optimize_indices.num_variadic_outputs);

  const auto fbs_produced_nodes = builder.CreateVector<flatbuffers::Offset<fbs::NodeIndexAndKernelDefHash>>(
      runtime_optimization_record.produced_nodes.size(),
      [&](size_t i) -> flatbuffers::Offset<fbs::NodeIndexAndKernelDefHash> {
        return fbs::CreateNodeIndexAndKernelDefHash(
            builder,
            gsl::narrow<uint32_t>(runtime_optimization_record.produced_nodes[i].node_index),
            runtime_optimization_record.produced_nodes[i].kernel_def_hash);
      });

  fbs_runtime_optimization_record =
      fbs::CreateRuntimeOptimizationRecord(builder,
                                           builder.CreateSharedString(runtime_optimization_record.action_id),
                                           fbs_nodes_to_optimize,
                                           fbs_produced_nodes);

  return Status::OK();
}

Status RuntimeOptimizationRecordContainer::SaveToOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<FbsRuntimeOptimizationRecordContainer>& fbs_runtime_optimizations) const {
  std::vector<flatbuffers::Offset<fbs::RuntimeOptimizationRecordContainerEntry>> fbs_runtime_optimizations_vector;
  fbs_runtime_optimizations_vector.reserve(optimizer_name_to_records_.size());
  for (const auto& [optimizer_name, records] : optimizer_name_to_records_) {
    std::vector<flatbuffers::Offset<fbs::RuntimeOptimizationRecord>> fbs_records_vector;
    fbs_records_vector.reserve(records.size());
    for (const auto& record : records) {
      flatbuffers::Offset<fbs::RuntimeOptimizationRecord> fbs_record_offset;
      ORT_RETURN_IF_ERROR(SaveRuntimeOptimizationRecordToOrtFormat(builder, record, fbs_record_offset));
      fbs_records_vector.push_back(fbs_record_offset);
    }

    fbs_runtime_optimizations_vector.push_back(
        fbs::CreateRuntimeOptimizationRecordContainerEntryDirect(builder,
                                                                 optimizer_name.c_str(),
                                                                 &fbs_records_vector));
  }

  fbs_runtime_optimizations = builder.CreateVectorOfSortedTables(&fbs_runtime_optimizations_vector);
  return Status::OK();
}

static Status LoadRuntimeOptimizationRecordFromOrtFormat(
    const fbs::RuntimeOptimizationRecord& fbs_runtime_optimization_record,
    RuntimeOptimizationRecord& runtime_optimization_record_out) {
  RuntimeOptimizationRecord runtime_optimization_record;

  fbs::utils::LoadStringFromOrtFormat(runtime_optimization_record.action_id,
                                      fbs_runtime_optimization_record.action_id());

  auto& nodes_to_optimize_indices = runtime_optimization_record.nodes_to_optimize_indices;
  if (const auto* fbs_nodes_to_optimize_indices = fbs_runtime_optimization_record.nodes_to_optimize_indices()) {
    if (const auto* fbs_node_indices = fbs_nodes_to_optimize_indices->node_indices()) {
      nodes_to_optimize_indices.nodes = [&]() {
        std::vector<NodeIndex> result;
        result.reserve(fbs_node_indices->size());
        std::transform(fbs_node_indices->begin(), fbs_node_indices->end(), std::back_inserter(result),
                       [](const uint32_t idx) { return static_cast<NodeIndex>(idx); });
        return result;
      }();
    }

    nodes_to_optimize_indices.num_inputs = fbs_nodes_to_optimize_indices->num_inputs();
    nodes_to_optimize_indices.num_outputs = fbs_nodes_to_optimize_indices->num_outputs();
    nodes_to_optimize_indices.variadic_input = fbs_nodes_to_optimize_indices->has_variadic_input();
    nodes_to_optimize_indices.variadic_output = fbs_nodes_to_optimize_indices->has_variadic_output();
    nodes_to_optimize_indices.num_variadic_inputs = fbs_nodes_to_optimize_indices->num_variadic_inputs();
    nodes_to_optimize_indices.num_variadic_outputs = fbs_nodes_to_optimize_indices->num_variadic_outputs();
  }

  if (const auto* fbs_produced_nodes = fbs_runtime_optimization_record.produced_nodes()) {
    runtime_optimization_record.produced_nodes.reserve(fbs_produced_nodes->size());
    for (const auto* fbs_node_index_and_kernel_def_hash : *fbs_produced_nodes) {
      if (!fbs_node_index_and_kernel_def_hash) continue;

      runtime_optimization_record.produced_nodes.push_back(
          NodeIndexAndKernelDefHash{static_cast<NodeIndex>(fbs_node_index_and_kernel_def_hash->node_index()),
                                    fbs_node_index_and_kernel_def_hash->kernel_def_hash()});
    }
  }

  runtime_optimization_record_out = std::move(runtime_optimization_record);
  return Status::OK();
}

Status RuntimeOptimizationRecordContainer::LoadFromOrtFormat(
    const FbsRuntimeOptimizationRecordContainer& fbs_runtime_optimizations) {
  OptimizerNameToRecordsMap optimizer_name_to_records;
  for (const auto* fbs_runtime_optimization : fbs_runtime_optimizations) {
    if (!fbs_runtime_optimization) continue;

    std::string optimizer_name;
    fbs::utils::LoadStringFromOrtFormat(optimizer_name, fbs_runtime_optimization->optimizer_name());

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

    ORT_RETURN_IF_NOT(optimizer_name_to_records.emplace(optimizer_name, std::move(records)).second,
                      "Attempting to load runtime optimization records for a previously loaded optimizer: ", optimizer_name);
  }

  optimizer_name_to_records_ = std::move(optimizer_name_to_records);
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)
