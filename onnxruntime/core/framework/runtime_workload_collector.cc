// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/runtime_workload_collector.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <limits>
#include <new>
#include <utility>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace {

void IncrementDropCount(uint64_t& count) {
  if (count != std::numeric_limits<uint64_t>::max()) {
    ++count;
  }
}

bool PayloadSize(std::string_view graph_path, std::string_view provider,
                 const MatMulNBitsRuntimeMetadata& metadata, size_t& bytes) {
  size_t dimensions;
  size_t shape_bytes;
  bytes = sizeof(RuntimeWorkloadRecord);
  return SafeAdd(metadata.input_shape.size(), metadata.output_shape.size(), dimensions) &&
         SafeMultiply(dimensions, sizeof(int64_t), shape_bytes) &&
         SafeAdd(bytes, shape_bytes, bytes) &&
         SafeAdd(bytes, graph_path.size(), bytes) &&
         SafeAdd(bytes, provider.size(), bytes) &&
         SafeAdd(bytes, size_t{2}, bytes);
}

}  // namespace

RuntimeWorkloadCollector::RuntimeWorkloadCollector(uint64_t session_id, uint64_t run_id,
                                                   RuntimeWorkloadLimits limits)
    : session_id_(session_id), run_id_(run_id), limits_(limits) {}

Status RuntimeWorkloadCollector::RecordMatMulNBits(std::string_view graph_path, size_t node_index,
                                                   std::string_view provider,
                                                   const MatMulNBitsRuntimeMetadata& metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (records_.size() >= limits_.max_records) {
    stats_.collection_complete = false;
    IncrementDropCount(stats_.record_limit_drops);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector record limit reached.");
  }

  size_t bytes;
  if (!PayloadSize(graph_path, provider, metadata, bytes) ||
      bytes > limits_.max_payload_bytes - stats_.retained_payload_bytes) {
    stats_.collection_complete = false;
    IncrementDropCount(stats_.payload_limit_drops);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector payload limit reached.");
  }

  Status status;
  ORT_TRY {
    RuntimeWorkloadRecord record{records_.size(), std::string(graph_path), node_index,
                                 std::string(provider), metadata};
    records_.push_back(std::move(record));
    stats_.retained_payload_bytes += bytes;
    return Status::OK();
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() {
      stats_.collection_complete = false;
      IncrementDropCount(stats_.allocation_failure_drops);
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector allocation failed.");
    });
  }
  return status;
}

Status RuntimeWorkloadCollector::GetSnapshot(RuntimeWorkloadSnapshot& output) const {
  std::lock_guard<std::mutex> lock(mutex_);
  Status status;
  ORT_TRY {
    RuntimeWorkloadSnapshot snapshot{session_id_, run_id_, records_, stats_};
    output = std::move(snapshot);
    return Status::OK();
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload snapshot allocation failed.");
    });
  }
  return status;
}

}  // namespace onnxruntime

#endif
