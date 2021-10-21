// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

namespace experimental::fbs {
struct RuntimeOptimizationRecordContainerEntry;
}  // namespace experimental::fbs

class RuntimeOptimizationRecordContainer {
 public:
#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)
  bool AddRecord(const std::string& sat_key, RuntimeOptimizationRecord&& runtime_optimization_record);
#endif

  std::vector<RuntimeOptimizationRecord> RemoveRecordsForKey(const std::string& sat_key);

  using FbsRuntimeOptimizationRecordContainer =
      flatbuffers::Vector<flatbuffers::Offset<
          onnxruntime::experimental::fbs::RuntimeOptimizationRecordContainerEntry>>;

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<FbsRuntimeOptimizationRecordContainer>& fbs_runtime_optimizations) const;

#if defined(ENABLE_ORT_FORMAT_LOAD)
  Status LoadFromOrtFormat(const FbsRuntimeOptimizationRecordContainer& fbs_runtime_optimizations);
#endif

 private:
  using SatToOptimizationRecordsMap = std::unordered_map<std::string, std::vector<RuntimeOptimizationRecord>>;
  SatToOptimizationRecordsMap sat_to_optimizations_;
};

}  // namespace onnxruntime
