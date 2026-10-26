// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

enum class MatMulNBitsRuntimeRoute {
  EmptyOutput,
  PackedLut,
  PackedQNBit,
  Unpacked,
};

// These describe the effective backend interface, not the activation tensor's dtype
// or a claim that all arithmetic in that route uses this type.
enum class MatMulNBitsRuntimeComputeMode {
  NotExecuted,
  Float32Gemm,
  Lut,
  SQNBitFp32,
  HQNBitFp16,
  BHQNBitBf16,
  SQNBitInt8,
  HQNBitInt8,
};

struct MatMulNBitsRuntimeMetadata {
  TensorShapeVector input_shape;
  TensorShapeVector output_shape;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  size_t batch_count = 0;
  bool bias_present = false;
  int32_t input_element_type = 0;
  uint32_t bits = 0;
  uint32_t block_size = 0;
  std::optional<MatMulNBitsRuntimeRoute> route;
  std::optional<MatMulNBitsRuntimeComputeMode> compute_mode;
};

struct RuntimeWorkloadLimits {
  size_t max_records;
  // Accounting budget: sizeof(record) plus copied string bytes (including terminators)
  // and shape elements. Not a hard allocator/peak-memory limit; capacity and snapshots add overhead.
  size_t max_payload_bytes;
};

struct RuntimeWorkloadRecord {
  // Unique within the collector; commit order, not execution order.
  size_t invocation_id;
  std::string graph_path;
  size_t node_index;
  std::string provider;
  MatMulNBitsRuntimeMetadata matmul_nbits;
};

struct RuntimeWorkloadCollectionStats {
  size_t retained_payload_bytes = 0;
  // Drop counters saturate rather than wrap.
  uint64_t record_limit_drops = 0;
  uint64_t payload_limit_drops = 0;
  uint64_t allocation_failure_drops = 0;
  // Completeness of submissions only, not operator/component coverage or run success.
  bool collection_complete = true;
};

struct RuntimeWorkloadSnapshot {
  uint64_t session_id = 0;
  uint64_t run_id = 0;
  InlinedVector<RuntimeWorkloadRecord, 1> records;
  RuntimeWorkloadCollectionStats stats;
};

// One collector per run. Callers own its lifetime and must keep it alive until all submitters finish.
// This internal storage primitive does not validate operator semantics, estimate counts, or export reports.
class RuntimeWorkloadCollector {
 public:
  RuntimeWorkloadCollector(uint64_t session_id, uint64_t run_id, RuntimeWorkloadLimits limits);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RuntimeWorkloadCollector);

  // Copies metadata without retaining tensors, caller storage, or graph objects.
  // Rejection is an explicit diagnostic error; future execution integration must not fail inference on it.
  [[nodiscard]] Status RecordMatMulNBits(std::string_view graph_path, size_t node_index,
                                         std::string_view provider, const MatMulNBitsRuntimeMetadata& metadata);
  // Copies a consistent snapshot. On allocation failure, returns an error without modifying output.
  [[nodiscard]] Status GetSnapshot(RuntimeWorkloadSnapshot& output) const;

 private:
  const uint64_t session_id_;
  const uint64_t run_id_;
  const RuntimeWorkloadLimits limits_;
  mutable std::mutex mutex_;
  InlinedVector<RuntimeWorkloadRecord, 1> records_;
  RuntimeWorkloadCollectionStats stats_;
};

}  // namespace onnxruntime

#endif
