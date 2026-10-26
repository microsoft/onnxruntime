// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <array>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "core/framework/compute_estimate_report.h"
#include "core/framework/runtime_workload_collector.h"
#include "core/graph/constants.h"
#include "test/util/include/asserts.h"
#if !defined(DISABLE_CONTRIB_OPS)
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#endif

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

#if !defined(DISABLE_CONTRIB_OPS)
nlohmann::json Report(const RuntimeWorkloadSnapshot& snapshot) {
  std::string text;
  ORT_THROW_IF_ERROR(BuildComputeEstimateReport(snapshot, text));
  return nlohmann::json::parse(text);
}
#endif

TEST(RuntimeWorkloadCollectorTest, EmptySnapshotHasRunIdentity) {
  RuntimeWorkloadCollector collector(12, 34, {10, 8192});
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.session_id, 12u);
  EXPECT_EQ(snapshot.run_id, 34u);
  EXPECT_TRUE(snapshot.records.empty());
  EXPECT_TRUE(snapshot.stats.collection_complete);
  EXPECT_EQ(snapshot.stats.retained_payload_bytes, 0u);
  EXPECT_EQ(snapshot.limits.max_records, 10u);
  EXPECT_EQ(snapshot.limits.max_payload_bytes, 8192u);
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

TEST(RuntimeWorkloadCollectorTest, RunLifecycleAndCollectionFailuresAreIndependent) {
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  EXPECT_TRUE(collector.BeginRun(23));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Running);
  EXPECT_EQ(snapshot.session_id, 23u);
  EXPECT_NE(snapshot.run_id, 0u);
  EXPECT_FALSE(collector.BeginRun(24));
  collector.RecordCollectionFailure(true);
  collector.RecordCollectionFailure(false);
  collector.EndRun(false);
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Failed);
  EXPECT_EQ(snapshot.session_id, 23u);
  EXPECT_EQ(snapshot.stats.run_reuse_rejections, 1u);
  EXPECT_EQ(snapshot.stats.allocation_failure_drops, 1u);
  EXPECT_EQ(snapshot.stats.metadata_failure_drops, 1u);
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

#if !defined(DISABLE_CONTRIB_OPS)

using ONNX_NAMESPACE::GraphProto;
using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::ValueInfoProto;

void SetValueType(ValueInfoProto& value, const char* name, int type, int rank) {
  value.set_name(name);
  auto* tensor_type = value.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(type);
  for (int i = 0; i < rank; ++i) {
    tensor_type->mutable_shape()->add_dim()->set_dim_param(std::string(name) + std::to_string(i));
  }
  if (rank == 0) {
    tensor_type->mutable_shape();
  }
}

void AddIntegerAttribute(ONNX_NAMESPACE::NodeProto& node, const char* name, int64_t value) {
  auto* attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}

ModelProto WorkloadModel(bool dynamic_weights = false, bool bias = true, int bits = 4,
                         bool half = false, int rank = 3, int64_t n = 32, int64_t k = 64) {
  ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(17);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);
  auto& graph = *model.mutable_graph();
  graph.set_name("runtime_workload");
  const int type = half ? TensorProto::FLOAT16 : TensorProto::FLOAT;
  SetValueType(*graph.add_input(), "A", type, rank);
  SetValueType(*graph.add_output(), "Y", type, rank);
  if (dynamic_weights) {
    SetValueType(*graph.add_input(), "B", TensorProto::UINT8, 3);
  } else {
    auto* b = graph.add_initializer();
    b->set_name("B");
    b->set_data_type(TensorProto::UINT8);
    b->add_dims(n);
    b->add_dims(k / 32);
    b->add_dims(32 * bits / 8);
    b->set_raw_data(std::string(static_cast<size_t>(n * k * bits / 8), bits == 2 ? '\xff' : '\x99'));
  }
  auto add_float_initializer = [&](const char* name, int64_t size, float value) {
    auto* tensor = graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(type);
    tensor->add_dims(size);
    for (int64_t i = 0; i < size; ++i) {
      if (half) {
        tensor->add_int32_data(MLFloat16(value).val);
      } else {
        tensor->add_float_data(value);
      }
    }
  };
  add_float_initializer("scales", n * k / 32, 1.f);
  if (bias) {
    add_float_initializer("bias", n, 2.f);
  }
  auto* node = graph.add_node();
  node->set_op_type("MatMulNBits");
  node->set_domain(kMSDomain);
  for (const char* input : {"A", "B", "scales"}) {
    node->add_input(input);
  }
  if (bias) {
    node->add_input("");
    node->add_input("");
    node->add_input("bias");
  }
  node->add_output("Y");
  AddIntegerAttribute(*node, "K", k);
  AddIntegerAttribute(*node, "N", n);
  AddIntegerAttribute(*node, "bits", bits);
  AddIntegerAttribute(*node, "block_size", 32);
  AddIntegerAttribute(*node, "accuracy_level", 4);
  return model;
}

ModelProto LoopWorkloadModel() {
  auto model = WorkloadModel();
  auto body = model.graph();
  body.set_name("workload_body");
  body.clear_input();
  SetValueType(*body.add_input(), "iteration", TensorProto::INT64, 0);
  SetValueType(*body.add_input(), "condition", TensorProto::BOOL, 0);
  body.clear_output();
  SetValueType(*body.add_output(), "condition_out", TensorProto::BOOL, 0);
  SetValueType(*body.add_output(), "Y", TensorProto::FLOAT, 3);
  auto* identity = body.add_node();
  identity->set_op_type("Identity");
  identity->add_input("condition");
  identity->add_output("condition_out");

  auto& graph = *model.mutable_graph();
  graph.clear_initializer();
  graph.clear_node();
  graph.clear_output();
  SetValueType(*graph.add_output(), "Y", TensorProto::FLOAT, 4);
  auto* iterations = graph.add_initializer();
  iterations->set_name("iterations");
  iterations->set_data_type(TensorProto::INT64);
  iterations->add_int64_data(3);
  auto* condition = graph.add_initializer();
  condition->set_name("condition");
  condition->set_data_type(TensorProto::BOOL);
  condition->add_int32_data(1);
  auto* loop = graph.add_node();
  loop->set_op_type("Loop");
  loop->add_input("iterations");
  loop->add_input("condition");
  loop->add_output("Y");
  auto* attribute = loop->add_attribute();
  attribute->set_name("body");
  attribute->set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
  *attribute->mutable_g() = std::move(body);
  return model;
}

template <typename T>
OrtValue FilledTensor(const std::vector<int64_t>& shape, T value);

TEST(RuntimeWorkloadExecutionTest, TransformerEncoderAndDecoderPropagateCollector) {
  ModelProto model;
  std::ifstream stream("testdata/dummy_t5.onnx", std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  ASSERT_TRUE(model.ParseFromIstream(&stream));
  size_t instrumented = 0;
  for (auto& node : *model.mutable_graph()->mutable_node()) {
    for (auto& attribute : *node.mutable_attribute()) {
      if (attribute.name() != "encoder" && attribute.name() != "decoder") {
        continue;
      }
      auto& graph = *attribute.mutable_g();
      ASSERT_GT(graph.output_size(), 0);
      const auto output_name = graph.output(0).name();
      const std::string prefix = "workload_probe_";
      for (auto& existing : *graph.mutable_node()) {
        for (auto& output : *existing.mutable_output()) {
          if (output == output_name) output = prefix + "original";
        }
        for (auto& input : *existing.mutable_input()) {
          if (input == output_name) input = prefix + "original";
        }
      }
      const auto probe = WorkloadModel(false, false);
      for (auto initializer : probe.graph().initializer()) {
        initializer.set_name(prefix + initializer.name());
        *graph.add_initializer() = std::move(initializer);
      }
      auto* a = graph.add_initializer();
      a->set_name(prefix + "A");
      a->set_data_type(TensorProto::FLOAT);
      for (int64_t dim : {1, 1, 64}) a->add_dims(dim);
      a->set_raw_data(std::string(64 * sizeof(float), '\0'));
      for (auto probe_node : probe.graph().node()) {
        for (auto& input : *probe_node.mutable_input()) {
          if (!input.empty()) input = prefix + input;
        }
        for (auto& output : *probe_node.mutable_output()) output = prefix + output;
        *graph.add_node() = std::move(probe_node);
      }
      auto* reduce = graph.add_node();
      reduce->set_op_type("ReduceSum");
      reduce->add_input(prefix + "Y");
      reduce->add_output(prefix + "zero");
      AddIntegerAttribute(*reduce, "keepdims", 0);
      auto* add = graph.add_node();
      add->set_op_type("Add");
      add->add_input(prefix + "original");
      add->add_input(prefix + "zero");
      add->add_output(output_name);
      ++instrumented;
    }
  }
  ASSERT_EQ(instrumented, 2u);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  InferenceSession session(options, GetEnvironment());
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  OrtValue input = FilledTensor<int32_t>({1, 5}, 0);
  const std::array<int32_t, 5> tokens{14, 6, 13, 9, 7};
  std::copy(tokens.begin(), tokens.end(), input.GetMutable<Tensor>()->MutableData<int32_t>());
  RuntimeWorkloadCollector collector(0, 0, {100, 1024 * 1024});
  RunOptions run_options;
  run_options.runtime_workload_collector = &collector;
  const std::vector<std::string> input_names{"encoder_input_ids"}, output_names{"sequences"};
  const std::vector<OrtValue> feeds{input};
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(session.Run(run_options, input_names, feeds, output_names, &outputs));
  RuntimeWorkloadSnapshot snapshot;
  ASSERT_STATUS_OK(collector.GetSnapshot(snapshot));
  size_t encoder_records = 0, decoder_records = 0;
  for (const auto& record : snapshot.records) {
    if (record.graph_path.find("/encoder") != std::string::npos) ++encoder_records;
    if (record.graph_path.find("/decoder") != std::string::npos) ++decoder_records;
  }
  EXPECT_EQ(encoder_records, 1u);
  EXPECT_GT(decoder_records, 1u);
  EXPECT_TRUE(snapshot.stats.collection_complete);
  const auto report = Report(snapshot);
  EXPECT_EQ(report["invocations"].size(), snapshot.records.size());
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], true);
  const std::vector<int32_t> expected{
      2, 16, 6, 14, 1, 15, 6, 14, 1, 15, 2, 3, 4, 15, 6, 14, 1, 15, 6, 14,
      2, 16, 6, 14, 1, 15, 6, 14, 1, 14};
  ASSERT_EQ(outputs.size(), 1u);
  const auto& output = outputs[0].Get<Tensor>();
  ASSERT_EQ(output.Shape().Size(), static_cast<int64_t>(expected.size()));
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), output.Data<int32_t>()));
}

std::unique_ptr<InferenceSession> WorkloadSession(const ModelProto& model, bool force_fp32 = false,
                                                  bool lut = false, bool parallel = false) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  options.inter_op_param.thread_pool_size = 2;
  options.execution_mode = parallel ? ORT_PARALLEL : ORT_SEQUENTIAL;
  if (force_fp32) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsMlasQNBitForceFp32, "1"));
  }
  if (lut) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsMlasLutGemm, "1"));
  }
  auto session = std::make_unique<InferenceSession>(options, GetEnvironment());
  const auto bytes = model.SerializeAsString();
  ORT_THROW_IF_ERROR(session->Load(bytes.data(), static_cast<int>(bytes.size())));
  ORT_THROW_IF_ERROR(session->Initialize());
  return session;
}

template <typename T>
OrtValue FilledTensor(const std::vector<int64_t>& shape, T value) {
  OrtValue result;
  Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape(shape), std::make_shared<CPUAllocator>(), result);
  auto* tensor = result.GetMutable<Tensor>();
  std::fill_n(tensor->MutableData<T>(), static_cast<size_t>(tensor->Shape().Size()), value);
  return result;
}

Status RunWorkload(InferenceSession& session, RuntimeWorkloadCollector* collector,
                   std::vector<OrtValue>& outputs, const std::vector<int64_t>& shape = {2, 3, 64},
                   bool dynamic_weights = false, bool half = false, bool terminate = false) {
  RunOptions options;
  options.runtime_workload_collector = collector;
  options.terminate = terminate;
  std::vector<std::string> names{"A"};
  std::vector<OrtValue> feeds;
  feeds.push_back(half ? FilledTensor(shape, MLFloat16(1.f)) : FilledTensor(shape, 1.f));
  if (dynamic_weights) {
    names.push_back("B");
    feeds.push_back(FilledTensor<uint8_t>({32, 2, 16}, 0x99));
  }
  const std::vector<std::string> output_names{"Y"};
  return session.Run(options, names, feeds, output_names, &outputs);
}

void ExpectOutput(const std::vector<OrtValue>& outputs, float expected, bool half = false) {
  ASSERT_EQ(outputs.size(), 1u);
  const auto& tensor = outputs[0].Get<Tensor>();
  for (int64_t i = 0; i < tensor.Shape().Size(); ++i) {
    EXPECT_FLOAT_EQ(half ? tensor.Data<MLFloat16>()[i].ToFloat() : tensor.Data<float>()[i], expected);
  }
}

RuntimeWorkloadSnapshot Snapshot(const RuntimeWorkloadCollector& collector) {
  RuntimeWorkloadSnapshot snapshot;
  ORT_THROW_IF_ERROR(collector.GetSnapshot(snapshot));
  return snapshot;
}

TEST(RuntimeWorkloadExecutionTest, DynamicRepeatedRunsAndDisabledCapturePreserveOutputs) {
  auto session = WorkloadSession(WorkloadModel(true));
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, nullptr, outputs, {2, 3, 64}, true));
  ExpectOutput(outputs, 66.f);
  uint64_t previous_run = 0;
  for (int64_t rows : {3, 5}) {
    RuntimeWorkloadCollector collector(0, 0, {10, 8192});
    outputs.clear();
    ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {2, rows, 64}, true));
    ExpectOutput(outputs, 66.f);
    auto snapshot = Snapshot(collector);
    ASSERT_EQ(snapshot.records.size(), 1u);
    EXPECT_EQ(snapshot.session_id, session->GetCurrentSessionId());
    EXPECT_NE(snapshot.run_id, previous_run);
    previous_run = snapshot.run_id;
    EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Succeeded);
    EXPECT_TRUE(snapshot.stats.collection_complete);
    const auto& record = snapshot.records[0];
    EXPECT_EQ(record.invocation_id, 0u);
    EXPECT_EQ(record.graph_path, "main");
    EXPECT_EQ(record.node_index, 0u);
    EXPECT_EQ(record.provider, kCpuExecutionProvider);
    const auto& metadata = record.matmul_nbits;
    EXPECT_EQ(metadata.input_shape, (TensorShapeVector{2, rows, 64}));
    EXPECT_EQ(metadata.output_shape, (TensorShapeVector{2, rows, 32}));
    // MatMulComputeHelper flattens the shared-B batches for the actual backend call.
    EXPECT_EQ(metadata.m, rows * 2);
    EXPECT_EQ(metadata.batch_count, 1u);
    EXPECT_EQ(metadata.n, 32);
    EXPECT_EQ(metadata.k, 64);
    EXPECT_TRUE(metadata.bias_present);
    EXPECT_EQ(metadata.bits, 4u);
    EXPECT_EQ(metadata.block_size, 32u);
    EXPECT_EQ(metadata.route, MatMulNBitsRuntimeRoute::Unpacked);
    EXPECT_EQ(metadata.compute_mode, MatMulNBitsRuntimeComputeMode::Float32Gemm);
  }
}

TEST(RuntimeWorkloadExecutionTest, PackedBackendModeIsEffectiveAndHalfTypeIsIndependent) {
  if (!MlasIsQNBitGemmAvailable(4, 32, SQNBIT_CompInt8)) {
    GTEST_SKIP() << "SQNBit int8 backend is unavailable.";
  }
  for (bool half : {false, true}) {
    auto session = WorkloadSession(WorkloadModel(false, false, 4, half));
    RuntimeWorkloadCollector collector(0, 0, {10, 8192});
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {2, 3, 64}, false, half));
    ExpectOutput(outputs, 64.f, half);
    auto snapshot = Snapshot(collector);
    ASSERT_EQ(snapshot.records.size(), 1u);
    const auto& metadata = snapshot.records[0].matmul_nbits;
    EXPECT_EQ(metadata.route, MatMulNBitsRuntimeRoute::PackedQNBit);
    EXPECT_EQ(metadata.compute_mode, MatMulNBitsRuntimeComputeMode::SQNBitInt8);
    EXPECT_EQ(metadata.input_element_type, half ? TensorProto::FLOAT16 : TensorProto::FLOAT);
    EXPECT_FALSE(metadata.bias_present);
  }
}

TEST(RuntimeWorkloadExecutionTest, ForceFp32RecordsActualPackedMode) {
#if defined(MLAS_TARGET_AMD64_IX86)
  if (!MlasIsQNBitGemmAvailable(4, 32, SQNBIT_CompFp32)) {
    GTEST_SKIP() << "SQNBit fp32 backend is unavailable.";
  }
  auto session = WorkloadSession(WorkloadModel(), true);
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs));
  ExpectOutput(outputs, 66.f);
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.route, MatMulNBitsRuntimeRoute::PackedQNBit);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.compute_mode, MatMulNBitsRuntimeComputeMode::SQNBitFp32);
#else
  GTEST_SKIP() << "Force-fp32 session option is x86-only.";
#endif
}

TEST(RuntimeWorkloadExecutionTest, EmptyOutputIsExplicitAndNoBackendIsClaimed) {
  auto session = WorkloadSession(WorkloadModel());
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {2, 0, 64}));
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].Get<Tensor>().Shape(), TensorShape({2, 0, 32}));
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.m, 0);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.route, MatMulNBitsRuntimeRoute::EmptyOutput);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.compute_mode, MatMulNBitsRuntimeComputeMode::NotExecuted);
  EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Succeeded);
}

TEST(RuntimeWorkloadExecutionTest, LutBackendIsRecordedOnlyWhenItActuallyRuns) {
  if (!MlasIsLutGemmAvailable(64, 128, 4, 32)) {
    GTEST_SKIP() << "LUT backend is unavailable.";
  }
  auto session = WorkloadSession(WorkloadModel(false, true, 4, false, 3, 64, 128), false, true);
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {2, 3, 128}));
  ExpectOutput(outputs, 130.f);
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.route, MatMulNBitsRuntimeRoute::PackedLut);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.compute_mode, MatMulNBitsRuntimeComputeMode::Lut);
}

TEST(RuntimeWorkloadExecutionTest, TwoBitVectorInvocationKeepsActualGeometry) {
  auto session = WorkloadSession(WorkloadModel(false, false, 2, false, 1));
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {64}));
  ExpectOutput(outputs, 64.f);
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 1u);
  const auto& metadata = snapshot.records[0].matmul_nbits;
  EXPECT_EQ(metadata.input_shape, TensorShapeVector{64});
  EXPECT_EQ(metadata.output_shape, TensorShapeVector{32});
  EXPECT_EQ(metadata.bits, 2u);
  EXPECT_EQ(metadata.m, 1);
  EXPECT_EQ(metadata.n, 32);
  EXPECT_EQ(metadata.k, 64);
  EXPECT_EQ(metadata.batch_count, 1u);
}

TEST(RuntimeWorkloadExecutionTest, KernelFailureAndTerminationDoNotCommitInvocations) {
  auto session = WorkloadSession(WorkloadModel());
  for (bool terminate : {false, true}) {
    RuntimeWorkloadCollector collector(0, 0, {10, 8192});
    std::vector<OrtValue> outputs;
    EXPECT_FALSE(RunWorkload(*session, &collector, outputs, {2, 3, terminate ? 64 : 63},
                             false, false, terminate)
                     .IsOK());
    auto snapshot = Snapshot(collector);
    EXPECT_TRUE(snapshot.records.empty());
    EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Failed);
    EXPECT_TRUE(snapshot.stats.collection_complete);
  }
}

TEST(RuntimeWorkloadExecutionTest, FailedRunRetainsEarlierSuccessfulInvocation) {
  auto model = WorkloadModel();
  auto* reshape = model.mutable_graph()->add_node();
  reshape->set_op_type("Reshape");
  reshape->add_input("Y");
  reshape->add_input("bad_shape");
  reshape->add_output("bad_output");
  auto* shape = model.mutable_graph()->add_initializer();
  shape->set_name("bad_shape");
  shape->set_data_type(TensorProto::INT64);
  shape->add_dims(1);
  shape->add_int64_data(5);
  model.mutable_graph()->clear_output();
  SetValueType(*model.mutable_graph()->add_output(), "bad_output", TensorProto::FLOAT, 1);
  auto session = WorkloadSession(model);
  RuntimeWorkloadCollector collector(0, 0, {10, 8192});
  RunOptions options;
  options.runtime_workload_collector = &collector;
  const std::vector<std::string> names{"A"}, output_names{"bad_output"};
  const std::vector<OrtValue> feeds{FilledTensor<float>({2, 3, 64}, 1.f)};
  std::vector<OrtValue> outputs;
  EXPECT_FALSE(session->Run(options, names, feeds, output_names, &outputs).IsOK());
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 1u);
  EXPECT_EQ(snapshot.records[0].matmul_nbits.route.has_value(), true);
  EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Failed);
  const auto report = Report(snapshot);
  EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], "24576");
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
}

TEST(RuntimeWorkloadExecutionTest, LoopInvocationsAndLimitsKeepSubgraphIdentity) {
  auto session = WorkloadSession(LoopWorkloadModel());
  for (size_t max_records : {size_t{10}, size_t{1}, size_t{0}}) {
    RuntimeWorkloadCollector collector(0, 0, {max_records, 8192});
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs));
    ExpectOutput(outputs, 66.f);
    EXPECT_EQ(outputs[0].Get<Tensor>().Shape(), TensorShape({3, 2, 3, 32}));
    auto snapshot = Snapshot(collector);
    ASSERT_EQ(snapshot.records.size(), std::min(max_records, size_t{3}));
    EXPECT_EQ(snapshot.stats.record_limit_drops, 3 - snapshot.records.size());
    EXPECT_EQ(snapshot.stats.collection_complete, max_records >= 3);
    EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Succeeded);
    const auto report = Report(snapshot);
    EXPECT_EQ(report["component_summaries"][0]["known_operation_count"],
              std::to_string(snapshot.records.size() * 24576));
    EXPECT_EQ(report["component_summaries"][1]["known_operation_count"],
              std::to_string(snapshot.records.size() * 192));
    EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], max_records >= 3);
    for (size_t i = 0; i < snapshot.records.size(); ++i) {
      EXPECT_EQ(snapshot.records[i].invocation_id, i);
      EXPECT_EQ(snapshot.records[i].graph_path, "main/0/body");
      EXPECT_EQ(snapshot.records[i].node_index, 0u);
    }
  }
}

TEST(RuntimeWorkloadExecutionTest, PayloadFailureAndCollectorReuseDoNotFailInference) {
  auto session = WorkloadSession(WorkloadModel(true));
  RuntimeWorkloadCollector collector(0, 0, {10, 0});
  for (int i = 0; i < 2; ++i) {
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {2, 3, 64}, true));
    ExpectOutput(outputs, 66.f);
  }
  auto snapshot = Snapshot(collector);
  EXPECT_TRUE(snapshot.records.empty());
  EXPECT_EQ(snapshot.stats.payload_limit_drops, 1u);
  EXPECT_EQ(snapshot.stats.run_reuse_rejections, 1u);
  EXPECT_FALSE(snapshot.stats.collection_complete);
  EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Succeeded);
}

TEST(RuntimeWorkloadExecutionTest, DistinctNodesKeepTheirExecutableNodeIndices) {
  auto model = WorkloadModel();
  auto& graph = *model.mutable_graph();
  auto second = graph.node(0);
  second.set_output(0, "second");
  *graph.add_node() = std::move(second);
  auto output = graph.output(0);
  output.set_name("second");
  *graph.add_output() = std::move(output);
  auto session = WorkloadSession(model, false, false, true);
  RuntimeWorkloadCollector collector({10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs));
  ExpectOutput(outputs, 66.f);
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 2u);
  EXPECT_NE(snapshot.records[0].node_index, snapshot.records[1].node_index);
  EXPECT_LT(snapshot.records[0].node_index, 2u);
  EXPECT_LT(snapshot.records[1].node_index, 2u);
  EXPECT_EQ(snapshot.records[0].invocation_id, 0u);
  EXPECT_EQ(snapshot.records[1].invocation_id, 1u);
  EXPECT_EQ(snapshot.records[0].graph_path, "main");
  EXPECT_EQ(snapshot.records[1].graph_path, "main");
}

TEST(RuntimeWorkloadExecutionTest, IfBranchesWithSameNodeIndexHaveDistinctGraphIdentities) {
  auto model = WorkloadModel();
  auto branch = model.graph();
  branch.clear_input();
  auto& graph = *model.mutable_graph();
  graph.clear_initializer();
  graph.clear_node();
  SetValueType(*graph.add_input(), "condition", TensorProto::BOOL, 0);
  auto* node = graph.add_node();
  node->set_op_type("If");
  node->add_input("condition");
  node->add_output("Y");
  for (const char* name : {"then_branch", "else_branch"}) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
    *attribute->mutable_g() = branch;
    attribute->mutable_g()->set_name(name);
  }
  auto session = WorkloadSession(model);
  for (bool condition : {true, false}) {
    RuntimeWorkloadCollector collector({10, 8192});
    RunOptions options;
    options.runtime_workload_collector = &collector;
    const std::vector<std::string> names{"A", "condition"}, output_names{"Y"};
    const std::vector<OrtValue> feeds{FilledTensor<float>({2, 3, 64}, 1.f), FilledTensor<bool>({}, condition)};
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session->Run(options, names, feeds, output_names, &outputs));
    ExpectOutput(outputs, 66.f);
    auto snapshot = Snapshot(collector);
    ASSERT_EQ(snapshot.records.size(), 1u);
    EXPECT_EQ(snapshot.records[0].node_index, 0u);
    EXPECT_EQ(snapshot.records[0].graph_path, condition ? "main/0/then_branch" : "main/0/else_branch");
    const auto report = Report(snapshot);
    EXPECT_EQ(report["invocations"].size(), 1u);
    EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], "24576");
  }
}

TEST(RuntimeWorkloadExecutionTest, ScanInvocationsShareIdentityButNotInvocationId) {
  auto model = WorkloadModel();
  auto body = model.graph();
  body.set_name("scan_body");
  auto& graph = *model.mutable_graph();
  graph.clear_initializer();
  graph.clear_node();
  graph.clear_input();
  graph.clear_output();
  SetValueType(*graph.add_input(), "A", TensorProto::FLOAT, 4);
  SetValueType(*graph.add_output(), "Y", TensorProto::FLOAT, 4);
  auto* node = graph.add_node();
  node->set_op_type("Scan");
  node->add_input("A");
  node->add_output("Y");
  AddIntegerAttribute(*node, "num_scan_inputs", 1);
  auto* attribute = node->add_attribute();
  attribute->set_name("body");
  attribute->set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
  *attribute->mutable_g() = std::move(body);
  auto session = WorkloadSession(model);
  RuntimeWorkloadCollector collector({10, 8192});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(RunWorkload(*session, &collector, outputs, {3, 2, 3, 64}));
  ExpectOutput(outputs, 66.f);
  auto snapshot = Snapshot(collector);
  ASSERT_EQ(snapshot.records.size(), 3u);
  EXPECT_EQ(Report(snapshot)["component_summaries"][0]["known_operation_count"], "73728");
  for (size_t i = 0; i < snapshot.records.size(); ++i) {
    EXPECT_EQ(snapshot.records[i].graph_path, "main/0/body");
    EXPECT_EQ(snapshot.records[i].node_index, 0u);
    EXPECT_EQ(snapshot.records[i].invocation_id, i);
    EXPECT_EQ(snapshot.records[i].matmul_nbits.input_shape, (TensorShapeVector{2, 3, 64}));
  }
}

TEST(RuntimeWorkloadExecutionTest, ConcurrentRunsAndSessionsAreIsolated) {
  auto first = WorkloadSession(WorkloadModel(true), false, false, true);
  auto second = WorkloadSession(WorkloadModel(true));
  std::array<RuntimeWorkloadSnapshot, 8> snapshots;
  std::array<std::thread, 8> workers;
  std::atomic<size_t> failures{0};
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i]() {
      auto& session = i < 6 ? *first : *second;
      RuntimeWorkloadCollector collector(0, 0, {10, 8192});
      std::vector<OrtValue> outputs;
      if (!RunWorkload(session, &collector, outputs, {2, static_cast<int64_t>(i + 1), 64}, true).IsOK() ||
          !collector.GetSnapshot(snapshots[i]).IsOK()) {
        ++failures;
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  ASSERT_EQ(failures, 0u);
  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& snapshot = snapshots[i];
    EXPECT_EQ(snapshot.session_id, (i < 6 ? first : second)->GetCurrentSessionId());
    EXPECT_EQ(snapshot.run_state, RuntimeWorkloadRunState::Succeeded);
    ASSERT_EQ(snapshot.records.size(), 1u);
    EXPECT_EQ(snapshot.records[0].invocation_id, 0u);
    EXPECT_EQ(snapshot.records[0].matmul_nbits.input_shape[1], static_cast<int64_t>(i + 1));
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(snapshot.run_id, snapshots[j].run_id);
    }
  }
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
