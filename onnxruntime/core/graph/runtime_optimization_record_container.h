// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/graph/runtime_optimization_record.h"

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
template <typename T>
class Vector;
}  // namespace flatbuffers

namespace onnxruntime {

namespace fbs {
struct RuntimeOptimizationRecordContainerEntry;
}  // namespace fbs

class RuntimeOptimizationRecordContainer {
 public:
  bool IsEmpty() const { return optimizer_name_to_records_.empty(); }

#if !defined(ORT_MINIMAL_BUILD)

  bool RecordExists(const std::string& optimizer_name,
                    const std::string& action_id,
                    const NodesToOptimizeIndices& nodes_to_optimize_indices) const;

  void AddRecord(const std::string& optimizer_name, RuntimeOptimizationRecord&& runtime_optimization_record);

#endif  // !defined(ORT_MINIMAL_BUILD)

  std::vector<RuntimeOptimizationRecord> RemoveRecordsForOptimizer(const std::string& optimizer_name);

  using FbsRuntimeOptimizationRecordContainer =
      flatbuffers::Vector<flatbuffers::Offset<
          onnxruntime::fbs::RuntimeOptimizationRecordContainerEntry>>;

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<FbsRuntimeOptimizationRecordContainer>& fbs_runtime_optimizations) const;

  Status LoadFromOrtFormat(const FbsRuntimeOptimizationRecordContainer& fbs_runtime_optimizations);

 private:
  using OptimizerNameToRecordsMap = std::unordered_map<std::string, std::vector<RuntimeOptimizationRecord>>;
  OptimizerNameToRecordsMap optimizer_name_to_records_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
