// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/graph/runtime_optimization_record.h"

#if !defined(ORT_MINIMAL_BUILD)
#define ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS
#endif  // !defined(ORT_MINIMAL_BUILD)

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
  void AddRecord(const std::string& optimizer_key, RuntimeOptimizationRecord&& runtime_optimization_record);
#endif

  // TODO add a way to access and remove them

  using FbsRuntimeOptimizationRecordContainer =
      flatbuffers::Vector<flatbuffers::Offset<
          onnxruntime::experimental::fbs::RuntimeOptimizationRecordContainerEntry>>;

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<FbsRuntimeOptimizationRecordContainer>& fbs_runtime_optimizations) const;

  Status LoadFromOrtFormat(const FbsRuntimeOptimizationRecordContainer& fbs_runtime_optimizations);

 private:
  using SatToOptimizationRecordsMap = std::unordered_map<std::string, std::vector<RuntimeOptimizationRecord>>;
  SatToOptimizationRecordsMap sat_to_optimizations_;
};

}  // namespace onnxruntime

#endif  // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
