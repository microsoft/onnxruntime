// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <array>
#include <atomic>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/runtime_workload_collector.h"
#include "core/graph/constants.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

MatMulNBitsRuntimeMetadata Metadata() {
  MatMulNBitsRuntimeMetadata metadata;
  metadata.input_shape = {2, 3, 64};
  metadata.output_shape = {2, 3, 32};
  metadata.m = 3;
  metadata.n = 32;
  metadata.k = 64;
  metadata.batch_count = 2;
  metadata.bias_present = true;
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  metadata.bits = 4;
  metadata.block_size = 32;
  metadata.route = MatMulNBitsRuntimeRoute::PackedQNBit;
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::SQNBitInt8;
  return metadata;
}

size_t PayloadBytes(const MatMulNBitsRuntimeMetadata& metadata) {
  return sizeof(RuntimeWorkloadRecord) + std::string("main").size() +
         std::string(kCpuExecutionProvider).size() + 2 +
         (metadata.input_shape.size() + metadata.output_shape.size()) * sizeof(int64_t);
}

TEST(RuntimeWorkloadCollectorTest, EmptySnapshotHasRunIdentity) {
  RuntimeWorkloadCollector collector(12, 34, {10, 8192});
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.session_id, 12u);
  EXPECT_EQ(snapshot.run_id, 34u);
  EXPECT_TRUE(snapshot.records.empty());
  EXPECT_TRUE(snapshot.stats.collection_complete);
  EXPECT_EQ(snapshot.stats.retained_payload_bytes, 0u);
}

TEST(RuntimeWorkloadCollectorTest, OwnsSubmittedMetadataAndSnapshotCopies) {
  RuntimeWorkloadCollector collector(12, 34, {10, 8192});
  auto metadata = Metadata();
  std::string graph = "main/2/then_branch";
  std::string provider = kCpuExecutionProvider;
  ASSERT_STATUS_OK(collector.RecordMatMulNBits(graph, 7, provider, metadata));
  graph.clear();
  provider.clear();
  metadata.input_shape[0] = 99;
  metadata.output_shape.clear();
  metadata.route.reset();
  metadata.compute_mode.reset();

  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), 1u);
  const auto& record = snapshot.records[0];
  EXPECT_EQ(record.invocation_id, 0u);
  EXPECT_EQ(record.graph_path, "main/2/then_branch");
  EXPECT_EQ(record.node_index, 7u);
  EXPECT_EQ(record.provider, kCpuExecutionProvider);
  EXPECT_EQ(record.matmul_nbits.input_shape, (TensorShapeVector{2, 3, 64}));
  EXPECT_EQ(record.matmul_nbits.output_shape, (TensorShapeVector{2, 3, 32}));
  EXPECT_EQ(record.matmul_nbits.m, 3);
  EXPECT_EQ(record.matmul_nbits.n, 32);
  EXPECT_EQ(record.matmul_nbits.k, 64);
  EXPECT_EQ(record.matmul_nbits.batch_count, 2u);
  EXPECT_TRUE(record.matmul_nbits.bias_present);
  EXPECT_EQ(record.matmul_nbits.input_element_type, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  EXPECT_EQ(record.matmul_nbits.bits, 4u);
  EXPECT_EQ(record.matmul_nbits.block_size, 32u);
  EXPECT_EQ(record.matmul_nbits.route, MatMulNBitsRuntimeRoute::PackedQNBit);
  EXPECT_EQ(record.matmul_nbits.compute_mode, MatMulNBitsRuntimeComputeMode::SQNBitInt8);

  snapshot.records[0].matmul_nbits.input_shape.clear();
  snapshot.records.clear();
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.input_shape, (TensorShapeVector{2, 3, 64}));
}

TEST(RuntimeWorkloadCollectorTest, RepeatedNodeKeepsDistinctInvocationsAndChangingShapes) {
  RuntimeWorkloadCollector collector(1, 2, {10, 8192});
  auto metadata = Metadata();
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  metadata.input_shape = {1, 5, 64};
  metadata.output_shape = {1, 5, 32};
  metadata.m = 5;
  metadata.batch_count = 1;
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), 2u);
  EXPECT_EQ(snapshot.records[0].invocation_id, 0u);
  EXPECT_EQ(snapshot.records[1].invocation_id, 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.input_shape, (TensorShapeVector{2, 3, 64}));
  EXPECT_EQ(snapshot.records[1].matmul_nbits.input_shape, (TensorShapeVector{1, 5, 64}));
  EXPECT_EQ(snapshot.records[1].matmul_nbits.m, 5);
}

TEST(RuntimeWorkloadCollectorTest, MissingRouteMetadataRemainsMissing) {
  RuntimeWorkloadCollector collector(1, 2, {10, 8192});
  auto metadata = Metadata();
  metadata.route.reset();
  metadata.compute_mode.reset();
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_FALSE(snapshot.records[0].matmul_nbits.route.has_value());
  EXPECT_FALSE(snapshot.records[0].matmul_nbits.compute_mode.has_value());
  EXPECT_TRUE(snapshot.stats.collection_complete);
}

TEST(RuntimeWorkloadCollectorTest, EmptyOutputRetainsExplicitNonExecutionMode) {
  RuntimeWorkloadCollector collector(1, 2, {10, 8192});
  auto metadata = Metadata();
  metadata.input_shape = {2, 0, 64};
  metadata.output_shape = {2, 0, 32};
  metadata.m = 0;
  metadata.route = MatMulNBitsRuntimeRoute::EmptyOutput;
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::NotExecuted;
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.records[0].matmul_nbits.route, MatMulNBitsRuntimeRoute::EmptyOutput);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.compute_mode, MatMulNBitsRuntimeComputeMode::NotExecuted);
}

TEST(RuntimeWorkloadCollectorTest, RecordLimitReturnsErrorAndPreservesAcceptedData) {
  RuntimeWorkloadCollector collector(1, 2, {1, 8192});
  const auto metadata = Metadata();
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  EXPECT_FALSE(collector.RecordMatMulNBits("main", 1, kCpuExecutionProvider, metadata).IsOK());
  EXPECT_FALSE(collector.RecordMatMulNBits("main", 2, kCpuExecutionProvider, metadata).IsOK());
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].node_index, 0u);
  EXPECT_EQ(snapshot.stats.record_limit_drops, 2u);
  EXPECT_EQ(snapshot.stats.payload_limit_drops, 0u);
  EXPECT_FALSE(snapshot.stats.collection_complete);
}

TEST(RuntimeWorkloadCollectorTest, PayloadBoundaryAndRejectedRecordDoNotConsumeBudget) {
  const auto metadata = Metadata();
  const auto bytes = PayloadBytes(metadata);
  RuntimeWorkloadCollector collector(1, 2, {10, bytes});
  EXPECT_FALSE(collector.RecordMatMulNBits("main/0/longer", 0, kCpuExecutionProvider, metadata).IsOK());
  ASSERT_STATUS_OK(collector.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata));
  EXPECT_FALSE(collector.RecordMatMulNBits("main", 1, kCpuExecutionProvider, metadata).IsOK());
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].invocation_id, 0u);
  EXPECT_EQ(snapshot.stats.retained_payload_bytes, bytes);
  EXPECT_EQ(snapshot.stats.payload_limit_drops, 2u);
  EXPECT_FALSE(snapshot.stats.collection_complete);

  RuntimeWorkloadCollector below_boundary(1, 3, {10, bytes - 1});
  EXPECT_FALSE(below_boundary.RecordMatMulNBits("main", 0, kCpuExecutionProvider, metadata).IsOK());
  ASSERT_STATUS_OK(below_boundary.GetSnapshot(snapshot));
  EXPECT_TRUE(snapshot.records.empty());
  EXPECT_EQ(snapshot.stats.retained_payload_bytes, 0u);
}

TEST(RuntimeWorkloadCollectorTest, ZeroLimitsExplicitlyRejectSubmissions) {
  RuntimeWorkloadCollector no_records(1, 2, {0, 8192});
  RuntimeWorkloadCollector no_bytes(1, 3, {10, 0});
  EXPECT_FALSE(no_records.RecordMatMulNBits("main", 0, kCpuExecutionProvider, Metadata()).IsOK());
  EXPECT_FALSE(no_bytes.RecordMatMulNBits("main", 0, kCpuExecutionProvider, Metadata()).IsOK());
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(no_records.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.stats.record_limit_drops, 1u);
  EXPECT_FALSE(snapshot.stats.collection_complete);
  ASSERT_STATUS_OK(no_bytes.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.stats.payload_limit_drops, 1u);
  EXPECT_FALSE(snapshot.stats.collection_complete);
}

TEST(RuntimeWorkloadCollectorTest, SeparateRunsDoNotShareRecordsOrIds) {
  RuntimeWorkloadCollector first(1, 2, {10, 8192});
  RuntimeWorkloadCollector second(1, 3, {10, 8192});
  ASSERT_STATUS_OK(first.RecordMatMulNBits("main", 7, kCpuExecutionProvider, Metadata()));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(second.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.run_id, 3u);
  EXPECT_TRUE(snapshot.records.empty());
  ASSERT_STATUS_OK(second.RecordMatMulNBits("main", 8, kCpuExecutionProvider, Metadata()));
  ASSERT_STATUS_OK(second.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.records[0].invocation_id, 0u);
  EXPECT_EQ(snapshot.records[0].node_index, 8u);
  ASSERT_STATUS_OK(first.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.run_id, 2u);
  EXPECT_EQ(snapshot.records[0].node_index, 7u);
}

TEST(RuntimeWorkloadCollectorTest, ConcurrentSubmissionsAndSnapshotsRespectLimits) {
  constexpr size_t limit = 100;
  constexpr size_t submissions_per_thread = 50;
  RuntimeWorkloadCollector collector(1, 2, {limit, std::numeric_limits<size_t>::max()});
  std::atomic<size_t> accepted{0};
  std::atomic<bool> workers_done{false};
  std::atomic<bool> snapshot_failed{false};
  std::thread reader([&]() {
    while (!workers_done.load()) {
      RuntimeWorkloadSnapshot snapshot;
      if (!collector.GetSnapshot(snapshot).IsOK() || snapshot.records.size() > limit) {
        snapshot_failed = true;
      }
    }
  });
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i]() {
      const auto metadata = Metadata();
      for (size_t j = 0; j < submissions_per_thread; ++j) {
        if (collector.RecordMatMulNBits("main", i, kCpuExecutionProvider, metadata).IsOK()) {
          ++accepted;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  workers_done = true;
  reader.join();
  EXPECT_FALSE(snapshot_failed);
  EXPECT_EQ(accepted, limit);
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  ASSERT_EQ(snapshot.records.size(), limit);
  EXPECT_EQ(snapshot.stats.record_limit_drops, workers.size() * submissions_per_thread - limit);
  EXPECT_EQ(snapshot.stats.retained_payload_bytes, limit * PayloadBytes(Metadata()));
  EXPECT_FALSE(snapshot.stats.collection_complete);
  for (size_t i = 0; i < limit; ++i) {
    EXPECT_EQ(snapshot.records[i].invocation_id, i);
    EXPECT_LT(snapshot.records[i].node_index, workers.size());
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
