// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/allocator_stats.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr int64_t kK = 256;
constexpr int64_t kM = 512;
constexpr int64_t kBlockSize = 32;
constexpr int64_t kBits = 4;
constexpr std::array<int64_t, 3> kOutputWidths{2048, 256, 1024};
constexpr int64_t kHyMT2SequenceLength = 64;
constexpr int64_t kHyMT2PastSequenceLength = 1;
constexpr int64_t kHyMT2NumLayers = 32;
constexpr int64_t kHyMT2NumKeyValueHeads = 4;
constexpr int64_t kHyMT2HeadSize = 128;
constexpr int64_t kHyMT2VocabSize = 120818;

void SetFp16MatrixShape(ONNX_NAMESPACE::ValueInfoProto* value_info, int64_t rows, int64_t columns) {
  auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  auto* shape = tensor_type->mutable_shape();
  shape->add_dim()->set_dim_value(rows);
  shape->add_dim()->set_dim_value(columns);
}

void AddMatMulNBitsNode(ONNX_NAMESPACE::GraphProto* graph,
                        const std::string& node_name,
                        const std::string& input_name,
                        const std::string& output_name,
                        int64_t k,
                        int64_t n) {
  const int64_t k_blocks = (k + kBlockSize - 1) / kBlockSize;
  const int64_t blob_size = (kBlockSize * kBits + 7) / 8;
  const std::string weight_name = node_name + "_B";
  const std::string scales_name = node_name + "_scales";

  auto* weight = graph->add_initializer();
  weight->set_name(weight_name);
  weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  weight->add_dims(n);
  weight->add_dims(k_blocks);
  weight->add_dims(blob_size);
  weight->mutable_raw_data()->assign(
      static_cast<size_t>(n * k_blocks * blob_size), static_cast<char>(0x11));

  auto* scales = graph->add_initializer();
  scales->set_name(scales_name);
  scales->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  scales->add_dims(n);
  scales->add_dims(k_blocks);
  const size_t scale_count = static_cast<size_t>(n * k_blocks);
  std::string scale_raw(scale_count * sizeof(uint16_t), '\0');
  const uint16_t scale_bits = MLFloat16(1.0f / 256.0f).val;
  for (size_t i = 0; i < scale_count; ++i) {
    std::memcpy(&scale_raw[i * sizeof(uint16_t)], &scale_bits, sizeof(uint16_t));
  }
  *scales->mutable_raw_data() = std::move(scale_raw);

  auto* node = graph->add_node();
  node->set_op_type("MatMulNBits");
  node->set_domain("com.microsoft");
  node->set_name(node_name);
  node->add_input(input_name);
  node->add_input(weight_name);
  node->add_input(scales_name);
  node->add_output(output_name);
  auto add_int_attribute = [node](const char* name, int64_t value) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  };
  add_int_attribute("K", k);
  add_int_attribute("N", n);
  add_int_attribute("block_size", kBlockSize);
  add_int_attribute("bits", kBits);
  add_int_attribute("accuracy_level", 0);
}

std::string BuildMatMulNBitsChainModelBytes() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(17);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("matmul_nbits_workspace_chain");
  auto* input = graph->add_input();
  input->set_name("A");
  SetFp16MatrixShape(input, kM, kK);

  std::string input_name = "A";
  int64_t input_width = kK;
  for (size_t i = 0; i < kOutputWidths.size(); ++i) {
    const std::string node_name = "matmul_nbits_" + std::to_string(i);
    const std::string output_name = "Y" + std::to_string(i);
    const int64_t output_width = kOutputWidths[i];
    AddMatMulNBitsNode(graph, node_name, input_name, output_name, input_width, output_width);
    if (i + 1 < kOutputWidths.size()) {
      auto* value_info = graph->add_value_info();
      value_info->set_name(output_name);
      SetFp16MatrixShape(value_info, kM, output_width);
    }
    input_name = output_name;
    input_width = output_width;
  }

  auto* output = graph->add_output();
  output->set_name(input_name);
  SetFp16MatrixShape(output, kM, kOutputWidths.back());

  std::string bytes;
  ORT_ENFORCE(model.SerializeToString(&bytes));
  return bytes;
}

void DumpModelIfRequested(const std::string& model_bytes) {
  const std::string model_dump_path =
      Env::Default().GetEnvironmentVar("ORT_MATMUL_NBITS_WORKSPACE_MODEL_PATH");
  if (model_dump_path.empty()) {
    return;
  }

  std::ofstream model_file(model_dump_path, std::ios::binary | std::ios::trunc);
  ORT_ENFORCE(model_file.is_open(), "Failed to open model dump path: ", model_dump_path);
  model_file.write(model_bytes.data(), static_cast<std::streamsize>(model_bytes.size()));
  ORT_ENFORCE(model_file.good(), "Failed to write model dump: ", model_dump_path);
  std::cout << "[ MODEL DUMP ] " << model_dump_path << std::endl;
}

std::vector<const Node*> FindMatMulNBitsNodes(const Graph& graph) {
  std::vector<const Node*> nodes;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBits") {
      nodes.push_back(&node);
    }
  }
  return nodes;
}

std::string BuildHyMT2MaxShapeOverride() {
  std::ostringstream shapes;
  shapes << "input_ids:[1," << kHyMT2SequenceLength << "]"
         << ";attention_mask:[1," << kHyMT2PastSequenceLength + kHyMT2SequenceLength << "]";
  for (int64_t layer = 0; layer < kHyMT2NumLayers; ++layer) {
    const std::string shape = ":[1," + std::to_string(kHyMT2NumKeyValueHeads) + "," +
                              std::to_string(kHyMT2PastSequenceLength) + "," +
                              std::to_string(kHyMT2HeadSize) + "]";
    shapes << ";past_key_values." << layer << ".key" << shape
           << ";past_key_values." << layer << ".value" << shape;
  }
  return shapes.str();
}

bool BlocksOverlap(const MemoryBlock& lhs, const MemoryBlock& rhs) {
  if (lhs.offset_ <= rhs.offset_) {
    return lhs.size_ > rhs.offset_ - lhs.offset_;
  }
  return rhs.size_ > lhs.offset_ - rhs.offset_;
}

struct ArenaMeasurement {
  int64_t second_run_cuda_allocated_bytes;
  size_t pattern_peak_bytes;
  size_t workspace_bytes;
};

ArenaMeasurement MeasureArenaReservation(const std::string& model_bytes,
                                         size_t node_count,
                                         bool enable_workspace_preallocation) {
  OrtCUDAProviderOptionsV2 cuda_options{};
  cuda_options.arena_extend_strategy = ArenaExtendStrategy::kSameAsRequested;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.use_tf32 = false;
  auto cuda_ep = CudaExecutionProviderWithOptions(&cuda_options);
  ORT_ENFORCE(cuda_ep != nullptr);

  SessionOptions session_options;
  session_options.session_logid =
      enable_workspace_preallocation ? "MatMulNBitsArenaPlanned" : "MatMulNBitsArenaScratch";
  ORT_THROW_IF_ERROR(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation,
      enable_workspace_preallocation ? "1" : "0"));

  InferenceSessionWrapper session(session_options, GetEnvironment());
  ORT_THROW_IF_ERROR(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ORT_THROW_IF_ERROR(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ORT_THROW_IF_ERROR(session.Initialize());

  const SessionState& session_state = session.GetSessionState();
  const auto matmul_nodes = FindMatMulNBitsNodes(session.GetGraph());
  ORT_ENFORCE(matmul_nodes.size() == node_count);

  std::vector<MLFloat16> input_data(static_cast<size_t>(kM * kK), MLFloat16(0.25f));
  OrtValue input_value;
  CreateMLValue<MLFloat16>(
      std::array<int64_t, 2>{kM, kK}, input_data.data(), OrtMemoryInfo(), &input_value);
  NameMLValMap feeds;
  feeds.emplace("A", input_value);
  const std::vector<std::string> output_names{"Y" + std::to_string(node_count - 1)};
  std::vector<OrtValue> fetches;

  // The first run traces lifetimes and builds the shape-specific memory pattern.
  ORT_THROW_IF_ERROR(session.Run(feeds, output_names, &fetches));
  fetches.clear();

  int input_index = -1;
  ORT_THROW_IF_ERROR(session_state.GetOrtValueNameIdxMap().GetIdx("A", input_index));
  const InlinedHashMap<int, TensorShape>* inferred_shapes = nullptr;
  const MemoryPatternGroup* pattern_group = session_state.GetMemoryPatternGroup(
      gsl::make_span(&input_value, 1), gsl::make_span(&input_index, 1), inferred_shapes);
  ORT_ENFORCE(pattern_group != nullptr);

  const OrtDevice cuda_device(
      OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);
  const MemoryPattern* pattern = pattern_group->GetPatterns(cuda_device);
  ORT_ENFORCE(pattern != nullptr);

  size_t workspace_bytes = 0;
  if (enable_workspace_preallocation) {
    const SequentialExecutionPlan* execution_plan = session_state.GetExecutionPlan();
    ORT_ENFORCE(execution_plan != nullptr);
    for (const Node* node : matmul_nodes) {
      const auto plan_it = execution_plan->workspace_allocation_plan.find(node->Index());
      ORT_ENFORCE(plan_it != execution_plan->workspace_allocation_plan.end());
      ORT_ENFORCE(plan_it->second.size() == 1);
      workspace_bytes = std::max(workspace_bytes, plan_it->second.front().allocation_bytes);
    }
  }

  AllocatorPtr cuda_allocator = session_state.GetAllocator(cuda_device);
  ORT_ENFORCE(cuda_allocator != nullptr);
  IArena* cuda_arena = IArena::SafeArenaCast(cuda_allocator.get());
  ORT_ENFORCE(cuda_arena != nullptr);

  // Remove all free first-run regions. With kSameAsRequested, the second run's increase in
  // total_allocated_bytes is the real amount newly reserved from CUDA for this cached pattern.
  ORT_THROW_IF_ERROR(cuda_arena->Shrink());
  AllocatorStats before_second_run;
  cuda_allocator->GetStats(&before_second_run);

  ORT_THROW_IF_ERROR(session.Run(feeds, output_names, &fetches));
  AllocatorStats after_second_run;
  cuda_allocator->GetStats(&after_second_run);

  return ArenaMeasurement{
      after_second_run.total_allocated_bytes - before_second_run.total_allocated_bytes,
      pattern->PeakSize(),
      workspace_bytes};
}

TEST(MatMulNBitsWorkspace, SequentialChainUsesSharedPlannedWorkspace) {
  constexpr size_t kNodeCount = kOutputWidths.size();
  ScopedEnvironmentVariables scoped_env(EnvVarMap{
      {"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}},
      {"ORT_FPA_INTB_PROFILE_M", optional<std::string>{std::to_string(kM)}},
  });
  const std::string model_bytes = BuildMatMulNBitsChainModelBytes();
  DumpModelIfRequested(model_bytes);

  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }

  SessionOptions session_options;
  session_options.session_logid = "MatMulNBitsWorkspaceChain";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation, "1"));
  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const SessionState& session_state = session.GetSessionState();
  const auto matmul_nodes = FindMatMulNBitsNodes(session.GetGraph());
  ASSERT_EQ(matmul_nodes.size(), kNodeCount);
  const SequentialExecutionPlan* execution_plan = session_state.GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);

  InlinedHashSet<int> workspace_pattern_ids;
  for (const Node* node : matmul_nodes) {
    ASSERT_EQ(node->GetExecutionProviderType(), kCudaExecutionProvider);
    const auto plan_it = execution_plan->workspace_allocation_plan.find(node->Index());
    ASSERT_NE(plan_it, execution_plan->workspace_allocation_plan.end());
    ASSERT_EQ(plan_it->second.size(), static_cast<size_t>(1));
    EXPECT_TRUE(workspace_pattern_ids.insert(plan_it->second.front().pattern_id).second);
  }

  std::vector<MLFloat16> input_data(static_cast<size_t>(kM * kK), MLFloat16(0.25f));
  OrtValue input_value;
  CreateMLValue<MLFloat16>(
      std::array<int64_t, 2>{kM, kK}, input_data.data(), OrtMemoryInfo(), &input_value);
  NameMLValMap feeds;
  feeds.emplace("A", input_value);
  const std::vector<std::string> output_names{"Y" + std::to_string(kNodeCount - 1)};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const auto first_output = fetches.front().Get<Tensor>().DataAsSpan<MLFloat16>();
  ASSERT_TRUE(std::any_of(first_output.begin(), first_output.end(),
                          [](MLFloat16 value) { return value.val != 0; }));
  const std::vector<MLFloat16> expected_output(first_output.begin(), first_output.end());

  int input_index = -1;
  ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx("A", input_index));
  const InlinedHashMap<int, TensorShape>* inferred_shapes = nullptr;
  const MemoryPatternGroup* pattern_group = session_state.GetMemoryPatternGroup(
      gsl::make_span(&input_value, 1), gsl::make_span(&input_index, 1), inferred_shapes);
  ASSERT_NE(pattern_group, nullptr);

  bool workspace_reuses_activation = false;
  for (const Node* node : matmul_nodes) {
    const auto& workspace_plan = execution_plan->workspace_allocation_plan.at(node->Index()).front();
    const MemoryPattern* pattern = pattern_group->GetPatterns(workspace_plan.location);
    ASSERT_NE(pattern, nullptr);
    const MemoryBlock* block = pattern->GetBlock(workspace_plan.pattern_id);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->size_, workspace_plan.allocation_bytes);
    for (const auto& [pattern_id, candidate] : pattern->GetPatternsMap()) {
      if (pattern_id >= 0 && BlocksOverlap(*block, candidate)) {
        workspace_reuses_activation = true;
        break;
      }
    }
  }
  EXPECT_TRUE(workspace_reuses_activation)
      << "At least one workspace should reuse memory occupied by a non-overlapping activation.";

  fetches.clear();
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const auto planned_output = fetches.front().Get<Tensor>().DataAsSpan<MLFloat16>();
  ASSERT_EQ(planned_output.size(), expected_output.size());
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(planned_output[i].val, expected_output[i].val);
  }
}

TEST(MatMulNBitsWorkspace, ReportsRealArenaMemorySavings) {
  constexpr size_t kNodeCount = kOutputWidths.size();
  ScopedEnvironmentVariables scoped_env(EnvVarMap{
      {"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}},
      {"ORT_FPA_INTB_PROFILE_M", optional<std::string>{std::to_string(kM)}},
  });
  const std::string model_bytes = BuildMatMulNBitsChainModelBytes();
  DumpModelIfRequested(model_bytes);

  if (!DefaultCudaExecutionProvider()) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }

  const ArenaMeasurement scratch = MeasureArenaReservation(
      model_bytes, kNodeCount, /*enable_workspace_preallocation=*/false);
  const ArenaMeasurement planned = MeasureArenaReservation(
      model_bytes, kNodeCount, /*enable_workspace_preallocation=*/true);

  ASSERT_GT(planned.workspace_bytes, static_cast<size_t>(0));
  ASSERT_GE(scratch.second_run_cuda_allocated_bytes, planned.second_run_cuda_allocated_bytes);
  const int64_t saved_bytes =
      scratch.second_run_cuda_allocated_bytes - planned.second_run_cuda_allocated_bytes;
  const int64_t input_copy_bytes = kM * kK * static_cast<int64_t>(sizeof(MLFloat16));
  const int64_t graph_output_bytes =
      kM * kOutputWidths.back() * static_cast<int64_t>(sizeof(MLFloat16));
  const int64_t scratch_component_total =
      input_copy_bytes +
      static_cast<int64_t>(scratch.pattern_peak_bytes) +
      graph_output_bytes +
      static_cast<int64_t>(planned.workspace_bytes);
  const int64_t planned_component_total =
      input_copy_bytes +
      static_cast<int64_t>(planned.pattern_peak_bytes) +
      graph_output_bytes;
  const int64_t pattern_growth =
      static_cast<int64_t>(planned.pattern_peak_bytes) -
      static_cast<int64_t>(scratch.pattern_peak_bytes);

  // Expected result for this model on the T1000/SM75 test configuration:
  //
  // [ ARENA MEMORY ] scratch path second-run components:
  //   CUDA input copy:                  262144 bytes
  //   activation-only pattern buffer:  2359296 bytes
  //   graph output buffer:             1048576 bytes
  //   separate CUTLASS workspace:      28672 bytes
  //   total new CUDA bytes:            3698688 bytes
  //
  // [ ARENA MEMORY ] planned path second-run components:
  //   CUDA input copy:                  262144 bytes
  //   activation+workspace buffer:     2362880 bytes
  //   graph output buffer:             1048576 bytes
  //   separate CUTLASS workspace:      0 bytes
  //   total new CUDA bytes:            3673600 bytes
  //
  // [ ARENA MEMORY ] pattern growth for workspace: 3584 bytes
  //   = 2362880 activation+workspace pattern - 2359296 activation-only pattern
  // [ ARENA MEMORY ] real CUDA bytes saved:        25088 bytes
  //   = 28672 separate scratch workspace - 3584 planned-pattern growth
  //   = 3698688 scratch-path total - 3673600 planned-path total
  //
  // The other 25088 bytes of the 28672-byte workspace overlap dead activation memory, so they
  // add no new CUDA allocation to the planned path.
  EXPECT_EQ(scratch.second_run_cuda_allocated_bytes, scratch_component_total);
  EXPECT_EQ(planned.second_run_cuda_allocated_bytes, planned_component_total);

  std::cout << "[ ARENA MEMORY ] shape: M=" << kM << ", K=" << kK
            << ", output widths=" << kOutputWidths[0] << "->" << kOutputWidths[1] << "->"
            << kOutputWidths[2] << '\n'
            << "[ ARENA MEMORY ] CUTLASS workspace: " << planned.workspace_bytes << " bytes\n"
            << "[ ARENA MEMORY ] scratch path second-run components:\n"
            << "  CUDA input copy:                  " << input_copy_bytes << " bytes\n"
            << "  activation-only pattern buffer:  " << scratch.pattern_peak_bytes << " bytes\n"
            << "  graph output buffer:             " << graph_output_bytes << " bytes\n"
            << "  separate CUTLASS workspace:      " << planned.workspace_bytes << " bytes\n"
            << "  total new CUDA bytes:            " << scratch.second_run_cuda_allocated_bytes << " bytes\n"
            << "[ ARENA MEMORY ] planned path second-run components:\n"
            << "  CUDA input copy:                  " << input_copy_bytes << " bytes\n"
            << "  activation+workspace buffer:     " << planned.pattern_peak_bytes << " bytes\n"
            << "  graph output buffer:             " << graph_output_bytes << " bytes\n"
            << "  separate CUTLASS workspace:      0 bytes\n"
            << "  total new CUDA bytes:            " << planned.second_run_cuda_allocated_bytes << " bytes\n"
            << "[ ARENA MEMORY ] pattern growth for workspace: " << pattern_growth << " bytes\n"
            << "[ ARENA MEMORY ] real CUDA bytes saved:        " << saved_bytes << " bytes"
            << std::endl;

  EXPECT_GT(saved_bytes, 0)
      << "The controlled kSameAsRequested arena run did not show a physical reservation reduction.";
}

TEST(MatMulNBitsWorkspace, HyMT2ModelRunsWithoutWorkspacePreallocation) {
  std::string model_path_utf8 = Env::Default().GetEnvironmentVar("ORT_HY_MT2_MODEL_PATH");
  if (model_path_utf8.empty()) {
    model_path_utf8 =
        "C:\\Users\\lochi\\repos\\onnxruntime\\Hy-MT2-1.8B-ONNX\\Q4_KQuant_tie\\cuda\\model.onnx";
  }
  const std::filesystem::path model_path(ToPathString(model_path_utf8));
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Hy-MT2 model not found at " << model_path_utf8
                 << "; set ORT_HY_MT2_MODEL_PATH to model.onnx.";
  }
  if (!DefaultCudaExecutionProvider()) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }

  SessionOptions session_options;
  session_options.session_logid = "HyMT2WithoutWorkspacePreallocation";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsMaxShapeOverride, BuildHyMT2MaxShapeOverride().c_str()));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsCudaFpAIntBGemm, "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation, "0"));

  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_path.native().c_str()));
  ASSERT_STATUS_OK(session.Initialize());

  const auto matmul_nodes = FindMatMulNBitsNodes(session.GetGraph());
  ASSERT_EQ(matmul_nodes.size(), static_cast<size_t>(225));
  const SequentialExecutionPlan* execution_plan = session.GetSessionState().GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);

  size_t cuda_matmul_nodes = 0;
  size_t planned_workspace_nodes = 0;
  size_t largest_workspace_bytes = 0;
  for (const Node* node : matmul_nodes) {
    if (node->GetExecutionProviderType() != kCudaExecutionProvider) {
      continue;
    }
    ++cuda_matmul_nodes;
    const auto plan_it = execution_plan->workspace_allocation_plan.find(node->Index());
    if (plan_it == execution_plan->workspace_allocation_plan.end()) {
      continue;
    }
    ASSERT_EQ(plan_it->second.size(), static_cast<size_t>(1));
    ++planned_workspace_nodes;
    largest_workspace_bytes =
        std::max(largest_workspace_bytes, plan_it->second.front().allocation_bytes);
  }
  EXPECT_EQ(cuda_matmul_nodes, matmul_nodes.size());
  EXPECT_EQ(planned_workspace_nodes, static_cast<size_t>(0));
  EXPECT_EQ(largest_workspace_bytes, static_cast<size_t>(0));

  std::vector<int64_t> input_ids(static_cast<size_t>(kHyMT2SequenceLength), 120000);
  std::vector<int64_t> attention_mask(
      static_cast<size_t>(kHyMT2PastSequenceLength + kHyMT2SequenceLength), 1);
  std::vector<MLFloat16> past_data(
      static_cast<size_t>(kHyMT2NumKeyValueHeads * kHyMT2PastSequenceLength * kHyMT2HeadSize),
      MLFloat16(0.0f));

  NameMLValMap feeds;
  OrtValue input_ids_value;
  CreateMLValue<int64_t>(
      std::array<int64_t, 2>{1, kHyMT2SequenceLength},
      input_ids.data(), OrtMemoryInfo(), &input_ids_value);
  feeds.emplace("input_ids", input_ids_value);

  OrtValue attention_mask_value;
  CreateMLValue<int64_t>(
      std::array<int64_t, 2>{1, kHyMT2PastSequenceLength + kHyMT2SequenceLength},
      attention_mask.data(), OrtMemoryInfo(), &attention_mask_value);
  feeds.emplace("attention_mask", attention_mask_value);

  const std::array<int64_t, 4> past_shape{
      1, kHyMT2NumKeyValueHeads, kHyMT2PastSequenceLength, kHyMT2HeadSize};
  for (int64_t layer = 0; layer < kHyMT2NumLayers; ++layer) {
    for (const char* kind : {"key", "value"}) {
      OrtValue past_value;
      CreateMLValue<MLFloat16>(
          past_shape, past_data.data(), OrtMemoryInfo(), &past_value);
      feeds.emplace(
          "past_key_values." + std::to_string(layer) + "." + kind, std::move(past_value));
    }
  }

  const std::vector<std::string> output_names{"logits"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const TensorShape expected_logits_shape({1, kHyMT2SequenceLength, kHyMT2VocabSize});
  ASSERT_EQ(fetches.front().Get<Tensor>().Shape(), expected_logits_shape);
  const auto first_logits = fetches.front().Get<Tensor>().DataAsSpan<MLFloat16>();
  const std::array<uint16_t, 3> expected_samples{
      first_logits.front().val,
      first_logits[first_logits.size() / 2].val,
      first_logits.back().val};
  const size_t expected_logits_size = first_logits.size();

  fetches.clear();
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const auto second_logits = fetches.front().Get<Tensor>().DataAsSpan<MLFloat16>();
  ASSERT_EQ(second_logits.size(), expected_logits_size);
  EXPECT_EQ(second_logits.front().val, expected_samples[0]);
  EXPECT_EQ(second_logits[second_logits.size() / 2].val, expected_samples[1]);
  EXPECT_EQ(second_logits.back().val, expected_samples[2]);

  std::cout << "[ HY-MT2 WORKSPACE ] CUDA MatMulNBits nodes: " << cuda_matmul_nodes
            << ", planned workspace nodes: " << planned_workspace_nodes
            << ", largest workspace: " << largest_workspace_bytes << " bytes" << std::endl;
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
#endif
