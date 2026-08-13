// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Cross-level agreement test (Major 1, Phase-A memory roadmap, issue microsoft/onnxruntime#29775)
// for the two-level MatMulNBits workspace-estimation pilot: it proves that the workspace *estimator
// function* agrees with the kernel-instance estimate and with the real runtime workspace request.
//
//   Level 1  : EstimateMatMulNBitsMemory(node, [shape,] device_prop) -- the same estimator function
//                                                                          that GetCapability() calls at
//                                                                          partition time; here it is invoked
//                                                                          DIRECTLY (no kernel).
//   Level 2  : OpKernel::DeclareWorkspaceRequirements(shapes)     -- constructed kernel instance
//                                                                    (virtual dispatch into MatMulNBits).
//   Runtime  : MatMulNBits<MLFloat16>::LastComputeWorkspaceBytes  -- recorded inside the CUTLASS GEMM
//                                                                    branch of the real Compute()
//                                                                    (read through the provider-world
//                                                                    probe in the .cc companion TU).
//
// The agreement tests exercise the estimator directly. GetCapabilityBudgetUsesLevel1Estimate separately
// drives the full estimator -> CUDA GetCapability -> resource-accountant budget path and verifies
// acceptance/rejection around the structured estimate.
// It also covers run-scoped memory-pattern integration with a sequential chain of MatMulNBits nodes:
// the first run traces each synthetic workspace lifetime, and the second run resolves all workspaces
// from one shared non-overlapping region in the cached activation pattern.
//
// This translation unit runs a real InferenceSession, so it includes the core framework headers.
// Those cannot coexist with the CUDA-provider (shared-provider bridge) headers in one TU, so the two
// provider-world pieces it needs (the Level-1 estimate and the runtime probe) are reached through
// slim, bridge-free declarations. It lives in the CUDA-only unit-test module because that is the only
// place these provider-internal symbols are linkable. Requires a real CUDA device; skips otherwise.

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/common/inlined_containers.h"
#include "core/framework/max_shape_inference.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/node_shape_resolver.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/workspace_requirement.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "contrib_ops/cuda/quantization/matmul_nbits_workspace_estimate.h"

#include "test/providers/cuda/test_cases/matmul_nbits_workspace_test_probe.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {

namespace {

// Representative fpA_intB-eligible configuration (matches the .cc file's CheckDefault): fp16 A, int4
// weights, block_size 32, N/K aligned. M is chosen >= 16 so the tactic profiler selects the CUTLASS
// GEMM tactic (not the GEMV cuda kernel); the GEMM branch is the one that records the runtime
// workspace size.
constexpr int64_t kE2eN = 256;
constexpr int64_t kE2eK = 1024;
constexpr int64_t kE2eM = 256;
constexpr int64_t kE2eBlockSize = 32;
constexpr int64_t kE2eBits = 4;
constexpr int64_t kWeightPrepackedSm80 = 1;

void SetFp16MatrixShape(ONNX_NAMESPACE::ValueInfoProto* value_info, const char* d0_param,
                        int64_t d0, int64_t d1) {
  auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  auto* shape = tensor_type->mutable_shape();
  auto* dim0 = shape->add_dim();
  if (d0_param != nullptr) {
    dim0->set_dim_param(d0_param);
  } else {
    dim0->set_dim_value(d0);
  }
  shape->add_dim()->set_dim_value(d1);
}

void AddMatMulNBitsNode(ONNX_NAMESPACE::GraphProto* graph, const std::string& node_name,
                        const std::string& input_name, const std::string& output_name,
                        int64_t k, int64_t n, uint8_t packed_weight = 0,
                        float scale_value = 1.0f, int64_t weight_prepacked = 0) {
  const int64_t k_blocks = (k + kE2eBlockSize - 1) / kE2eBlockSize;
  const int64_t blob_size = (kE2eBlockSize * kE2eBits + 7) / 8;
  const std::string weight_name = node_name + "_B";
  const std::string scales_name = node_name + "_scales";

  auto* weight = graph->add_initializer();
  weight->set_name(weight_name);
  weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  weight->add_dims(n);
  weight->add_dims(k_blocks);
  weight->add_dims(blob_size);
  weight->mutable_raw_data()->assign(
      static_cast<size_t>(n * k_blocks * blob_size), static_cast<char>(packed_weight));

  auto* scales = graph->add_initializer();
  scales->set_name(scales_name);
  scales->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  scales->add_dims(n);
  scales->add_dims(k_blocks);
  const size_t n_scales = static_cast<size_t>(n * k_blocks);
  std::string scale_raw(n_scales * sizeof(uint16_t), '\0');
  const uint16_t scale_bits = MLFloat16(scale_value).val;
  for (size_t i = 0; i < n_scales; ++i) {
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
  auto add_int_attr = [node](const char* name, int64_t value) {
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(value);
  };
  add_int_attr("K", k);
  add_int_attr("N", n);
  add_int_attr("block_size", kE2eBlockSize);
  add_int_attr("bits", kE2eBits);
  add_int_attr("accuracy_level", 0);
  if (weight_prepacked != 0) {
    add_int_attr("weight_prepacked", weight_prepacked);
  }
}

// Builds a minimal single-node MatMulNBits model and returns its serialized ModelProto bytes.
// The compact fpA_intB path does not support bias. Other builds use the valid optional-input
// layout [A, B, scales, "", "", bias] to exercise positional Level-2 shape resolution.
//
// When `m_dim_param` is null (default), input A's leading dimension is the fully-static value
// `kE2eM` (a concrete dim_value). When `m_dim_param` is non-null, that leading dimension is instead
// a *symbolic* dim (dim_param, e.g. "seq") with NO dim_value -- a genuinely dynamic shape. This is
// used by Test D (dynamic, no override -> graceful fallback) and by Test C, where a
// FreeDimensionOverrideByName later rewrites the symbolic dim into a concrete value at session init.
std::string BuildMatMulNBitsModelBytes(const char* m_dim_param = nullptr,
                                       int64_t weight_prepacked = 0) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  {
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");
    onnx_opset->set_version(17);
    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain("com.microsoft");
    ms_opset->set_version(1);
  }

  auto* graph = model.mutable_graph();
  graph->set_name("matmul_nbits_workspace_e2e");

  // Graph input A and output Y (both fp16). A's leading dim mirrors `m_dim_param`; Y's leading dim
  // mirrors it too so the declared output shape stays consistent with shape inference.
  auto* a = graph->add_input();
  a->set_name("A");
  SetFp16MatrixShape(a, m_dim_param, kE2eM, kE2eK);
  auto* y = graph->add_output();
  y->set_name("Y");
  SetFp16MatrixShape(y, m_dim_param, kE2eM, kE2eN);

  AddMatMulNBitsNode(
      graph, "matmul_nbits", "A", "Y", kE2eK, kE2eN,
      /*packed_weight=*/0, /*scale_value=*/1.0f, weight_prepacked);

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

std::string BuildMatMulNBitsChainModelBytes(size_t node_count) {
  ORT_ENFORCE(node_count > 1);

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  {
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");
    onnx_opset->set_version(17);
    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain("com.microsoft");
    ms_opset->set_version(1);
  }

  auto* graph = model.mutable_graph();
  graph->set_name("matmul_nbits_workspace_chain");

  auto* input = graph->add_input();
  input->set_name("A");
  SetFp16MatrixShape(input, nullptr, kE2eM, kE2eK);

  std::string input_name = "A";
  int64_t input_width = kE2eK;
  for (size_t i = 0; i < node_count; ++i) {
    const std::string node_name = "matmul_nbits_" + std::to_string(i);
    const std::string output_name = "Y" + std::to_string(i);
    AddMatMulNBitsNode(
        graph, node_name, input_name, output_name, input_width, kE2eN,
        /*packed_weight=*/0x11, /*scale_value=*/1.0f / 256.0f);

    if (i + 1 < node_count) {
      auto* value_info = graph->add_value_info();
      value_info->set_name(output_name);
      SetFp16MatrixShape(value_info, nullptr, kE2eM, kE2eN);
    }
    input_name = output_name;
    input_width = kE2eN;
    input_width = kE2eN;
  }

  auto* output = graph->add_output();
  output->set_name(input_name);
  SetFp16MatrixShape(output, nullptr, kE2eM, kE2eN);

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

// Builds a trivial single-node fp32 Add model (X + Y -> Z, all [kAddM, kAddN]) and returns its
// serialized ModelProto bytes. Used by Test E as a control: Add does NOT override
// DeclareWorkspaceRequirements, so it must hit the OpKernel base-class no-op.
constexpr int64_t kAddM = 4;
constexpr int64_t kAddN = 8;

std::string BuildAddModelBytes() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(17);

  auto* graph = model.mutable_graph();
  graph->set_name("add_workspace_control");

  auto set_fp32_shape = [](ONNX_NAMESPACE::ValueInfoProto* vi, int64_t d0, int64_t d1) {
    auto* tt = vi->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(d0);
    shape->add_dim()->set_dim_value(d1);
  };

  auto* x = graph->add_input();
  x->set_name("X");
  set_fp32_shape(x, kAddM, kAddN);
  auto* y = graph->add_input();
  y->set_name("Y");
  set_fp32_shape(y, kAddM, kAddN);
  auto* z = graph->add_output();
  z->set_name("Z");
  set_fp32_shape(z, kAddM, kAddN);

  auto* node = graph->add_node();
  node->set_op_type("Add");
  node->set_domain("");
  node->set_name("add");
  node->add_input("X");
  node->add_input("Y");
  node->add_output("Z");

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

// fpA_intB eligibility requires compute capability >= 7.5 (see CheckFpAIntBEligibility in
// matmul_nbits.cc). Returns the compute capability (major*10+minor) of CUDA device 0, or -1 when no
// device is present. Tests that exercise the fpA_intB-eligible path must skip on older GPUs (rather
// than fail) so they stay portable across the varying-compute-capability CI GPUs.
int CudaDeviceComputeCapabilityOrNegative() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return -1;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return -1;
  }
  return prop.major * 10 + prop.minor;
}

// Minimum compute capability for the fpA_intB path (mirrors the production eligibility gate).
constexpr int kMinFpAIntBSm = 75;

// Returns the first node in `graph` whose op type matches `op_type`, or nullptr if none.
const Node* FindNodeByOpType(const Graph& graph, const std::string& op_type) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) {
      return &node;
    }
  }
  return nullptr;
}

std::optional<size_t> EstimateWorkspaceFromGraphProtoShape(
    gsl::span<const int64_t> input_a_shape) {
  ONNX_NAMESPACE::TypeProto input_type;
  auto* tensor_type = input_type.mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  auto* shape = tensor_type->mutable_shape();
  for (const int64_t dimension : input_a_shape) {
    shape->add_dim()->set_dim_value(dimension);
  }
  NodeArg input_a{"A", &input_type};

  NodeAttributes attributes;
  auto add_int_attribute = [&attributes](const char* name, int64_t value) {
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_name(name);
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute.set_i(value);
    attributes.emplace(name, std::move(attribute));
  };
  add_int_attribute("N", kE2eN);
  add_int_attribute("K", kE2eK);
  add_int_attribute("block_size", kE2eBlockSize);
  add_int_attribute("bits", kE2eBits);

  const std::vector<NodeArg*> inputs{&input_a};
  const std::vector<NodeArg*> outputs;
  Node node{"matmul_nbits", "MatMulNBits", "", inputs, outputs, &attributes, "com.microsoft"};

  cudaDeviceProp device_prop{};
  device_prop.major = 8;
  device_prop.minor = 0;
  device_prop.multiProcessorCount = 100;
  const auto estimate = onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(node, device_prop);
  return estimate.has_value() ? estimate->runtime_workspace_bytes : std::nullopt;
}

std::vector<const Node*> FindNodesByOpType(const Graph& graph, const std::string& op_type) {
  std::vector<const Node*> nodes;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) {
      nodes.push_back(&node);
    }
  }
  return nodes;
}

}  // namespace

TEST(MatMulNBitsWorkspace, KnownZeroPartialLeadingShapesLevel1GraphProto) {
  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});

  const std::vector<TensorShapeVector> known_zero_shapes{
      {0, -1, kE2eK},
      {-1, 0, kE2eK},
      {std::numeric_limits<int64_t>::max(), 2, 0, kE2eK},
  };
  for (const auto& shape : known_zero_shapes) {
    SCOPED_TRACE(TensorShape{shape}.ToString());
    EXPECT_EQ(EstimateWorkspaceFromGraphProtoShape(shape), std::optional<size_t>{0});
  }

  const TensorShapeVector unknown_shape{-1, 2, kE2eK};
  EXPECT_FALSE(EstimateWorkspaceFromGraphProtoShape(unknown_shape).has_value());
  const TensorShapeVector overflowing_shape{
      std::numeric_limits<int64_t>::max(), 2, kE2eK};
  EXPECT_FALSE(EstimateWorkspaceFromGraphProtoShape(overflowing_shape).has_value());
}

TEST(MatMulNBitsWorkspace, KnownZeroPartialLeadingShapesLevel2TensorShape) {
  const std::vector<TensorShapeVector> known_zero_shapes{
      {0, -1, kE2eK},
      {-1, 0, kE2eK},
      {std::numeric_limits<int64_t>::max(), 2, 0, kE2eK},
  };
  for (const auto& dimensions : known_zero_shapes) {
    const TensorShape shape{dimensions};
    SCOPED_TRACE(shape.ToString());
    EXPECT_EQ(onnxruntime::contrib::cuda::ComputeMatMulNBitsLeadingDimProduct(shape.GetDims()),
              std::optional<int64_t>{0});
  }

  const TensorShape unknown_shape({-1, 2, kE2eK});
  EXPECT_FALSE(onnxruntime::contrib::cuda::ComputeMatMulNBitsLeadingDimProduct(
                   unknown_shape.GetDims())
                   .has_value());
  const TensorShape overflowing_shape(
      {std::numeric_limits<int64_t>::max(), 2, kE2eK});
  EXPECT_FALSE(onnxruntime::contrib::cuda::ComputeMatMulNBitsLeadingDimProduct(
                   overflowing_shape.GetDims())
                   .has_value());
}

TEST(MatMulNBitsWorkspace, GetCapabilityBudgetUsesLevel1Estimate) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping budget integration test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible.";
  }

  // Prove that Level 1 and the kernel constructor both honor session config
  // over conflicting process-wide environment variables.
  ScopedEnvironmentVariables scoped_env(
      EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"0"}},
                {"ORT_FPA_INTB_PROFILE_M", optional<std::string>{"2048"}}});
  const std::string model_bytes = BuildMatMulNBitsModelBytes();

  // For this model the legacy 1.5x fallback is below 700 KiB, while the
  // structured Level-1 total (runtime workspace + persistent/temporary
  // initialization memory, including tactic-profiler scratch) is above
  // 700 KiB and below 720 KiB. The two budgets
  // therefore prove that CUDA GetCapability applies the estimator result.
  {
    SessionOptions so;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaFpAIntBGemm, "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaFpAIntBProfileM, "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsResourceCudaPartitioningSettings, "720,"));
    InferenceSessionWrapper session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{})));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* mm_node = FindNodeByOpType(session.GetGraph(), "MatMulNBits");
    ASSERT_NE(mm_node, nullptr);
    EXPECT_EQ(mm_node->GetExecutionProviderType(), kCudaExecutionProvider);
  }

  {
    SessionOptions so;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaFpAIntBGemm, "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaFpAIntBProfileM, "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsResourceCudaPartitioningSettings, "700,"));
    InferenceSessionWrapper session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{})));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const Node* mm_node = FindNodeByOpType(session.GetGraph(), "MatMulNBits");
    ASSERT_NE(mm_node, nullptr);
    EXPECT_NE(mm_node->GetExecutionProviderType(), kCudaExecutionProvider);
  }
}

TEST(MatMulNBitsWorkspace, GetCapabilityBudgetDoesNotDuplicateOfflinePrepackedGpuWeight) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping budget integration test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible.";
  }

  const std::string model_bytes =
      BuildMatMulNBitsModelBytes(nullptr, kWeightPrepackedSm80);

  ScopedEnvironmentVariables scoped_env(
      EnvVarMap{{"ORT_FPA_INTB_PROFILE_M", optional<std::string>{"1"}}});

  // The 500 KiB budget is above initializer + output + scale destination +
  // runtime workspace + tactic-profiler scratch, but below that value plus a
  // duplicate packed-weight destination. CUDA must accept the node because
  // PrePack_B reuses the already GPU-resident offline-prepacked initializer.
  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "500,"));
  InferenceSessionWrapper session(so, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(
      std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{})));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Node* mm_node = FindNodeByOpType(session.GetGraph(), "MatMulNBits");
  ASSERT_NE(mm_node, nullptr);
  EXPECT_EQ(mm_node->GetExecutionProviderType(), kCudaExecutionProvider);
}

TEST(MatMulNBitsWorkspace, EndToEndWorkspaceAgreement) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping end-to-end workspace test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible, so the eligible-path assertions "
                    "cannot hold. Skipping.";
  }

  // Enable the fpA_intB path via the ENV var (not the session config) so that BOTH Level 1 - which
  // can only read the env var (see the Major-2 known limitation in EstimateMatMulNBitsMemory) -
  // and the kernel constructor observe it enabled, keeping the two eligibility decisions in sync.
  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});

  // Keep M dynamic so the same session and kernel instance can execute both the ordinary and
  // empty-output cases.
  const std::string model_bytes = BuildMatMulNBitsModelBytes("seq");

  SessionOptions so;
  so.session_logid = "MatMulNBitsWorkspaceE2E";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation, "1"));
  InferenceSessionWrapper session(so, GetEnvironment());

  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));

  // Production computes estimation-only shadow shapes before initialization can prepack and remove
  // initializer data from the executable graph. Do the same for the ordinary and empty shapes.
  const TensorShape positive_a_shape({kE2eM, kE2eK});
  MaxShapeOverrideMap positive_shape_override;
  positive_shape_override.emplace("A", positive_a_shape);
  MaxShapeInferenceResult positive_inferred_shapes;
  ASSERT_STATUS_OK(InferMaxShapes(session.GetGraph(), positive_shape_override, positive_inferred_shapes));

  const TensorShape zero_m_a_shape({0, kE2eK});
  MaxShapeOverrideMap zero_m_shape_override;
  zero_m_shape_override.emplace("A", zero_m_a_shape);
  MaxShapeInferenceResult zero_m_inferred_shapes;
  ASSERT_STATUS_OK(InferMaxShapes(session.GetGraph(), zero_m_shape_override, zero_m_inferred_shapes));

  ASSERT_STATUS_OK(session.Initialize());

  // Locate the MatMulNBits node and confirm it was assigned to the CUDA EP (fpA_intB eligible).
  const Graph& graph = session.GetGraph();
  const Node* mm_node = FindNodeByOpType(graph, "MatMulNBits");
  ASSERT_NE(mm_node, nullptr) << "MatMulNBits node not found in the graph.";
  ASSERT_EQ(mm_node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
      << "MatMulNBits node was not assigned to the CUDA EP.";

  // ---- Level 1: the estimator function GetCapability() uses, invoked with the concrete estimation
  //      shape for this run (this is not a full GetCapability()-driven partition-time run). ----
  const std::optional<Level1MemoryEstimate> level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(
          *mm_node, positive_a_shape.GetDims(), cuda_ep->GetDeviceProp());
  ASSERT_TRUE(level1.has_value()) << "Level-1 estimate returned nullopt for an eligible node.";
  ASSERT_TRUE(level1->runtime_workspace_bytes.has_value())
      << "Level-1 runtime workspace was not estimable for the concrete estimation shape.";
  const size_t level1_runtime_workspace = *level1->runtime_workspace_bytes;
  EXPECT_EQ(level1_runtime_workspace, 1792u);

  // ---- Level 2: instance-level estimate from the constructed kernel and production positional
  //      shape resolver. The two internal missing inputs must not prevent the override from being
  //      reached, and the known bias after those holes must retain index 5. ----
  const OpKernel* op_kernel = session.GetSessionState().GetKernel(mm_node->Index());
  ASSERT_NE(op_kernel, nullptr) << "No kernel constructed for the MatMulNBits node.";

  const auto input_shapes =
      ResolveNodeInputShapes(*mm_node, &graph, positive_inferred_shapes);
  ASSERT_NE(input_shapes[0].GetShape(), nullptr);
  EXPECT_EQ(*input_shapes[0].GetShape(), TensorShape({kE2eM, kE2eK}));
  ASSERT_EQ(input_shapes.size(), 6u);
  EXPECT_EQ(input_shapes[3].GetState(), WorkspaceInputShapeState::Missing);
  EXPECT_EQ(input_shapes[4].GetState(), WorkspaceInputShapeState::Missing);
#if USE_COMPACT_FPA_INTB_GEMM
  EXPECT_EQ(input_shapes[5].GetState(), WorkspaceInputShapeState::Missing);
#else
  EXPECT_EQ(input_shapes[5].GetState(), WorkspaceInputShapeState::PresentWithShape);
#endif
  InlinedVector<WorkspaceRequirement> requirements;
  // DeclareWorkspaceRequirements is virtual on OpKernel; this dispatches into the MatMulNBits override.
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(gsl::make_span(input_shapes), requirements));
  ASSERT_EQ(requirements.size(), static_cast<size_t>(1))
      << "Level-2 DeclareWorkspaceRequirements did not report exactly one workspace slot.";
  const size_t level2 = requirements[0].size_bytes;
  EXPECT_EQ(level2, 1792u);

  auto shapeless_optional_shapes = input_shapes;
  shapeless_optional_shapes[3] = WorkspaceInputShape::PresentWithoutShape();
  InlinedVector<WorkspaceRequirement> shapeless_optional_requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapeless_optional_shapes), shapeless_optional_requirements));
  ASSERT_EQ(shapeless_optional_requirements.size(), 1u);
  EXPECT_EQ(shapeless_optional_requirements[0].size_bytes, level2);

  // K comes from the kernel attribute, so a partial input-A shape with an unknown final K dimension
  // remains estimable as long as every leading dimension is known.
  const std::array<WorkspaceInputShape, 1> unknown_k_shapes{
      WorkspaceInputShape::PresentWithShape(TensorShape({4, -1}))};
  InlinedVector<WorkspaceRequirement> unknown_k_requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
      gsl::make_span(unknown_k_shapes), unknown_k_requirements));
  ASSERT_EQ(unknown_k_requirements.size(), 1u);
  EXPECT_GT(unknown_k_requirements[0].size_bytes, 0u);

  // A zero in any leading dimension proves m == 0 even if another leading dimension is unknown
  // or a huge earlier prefix would overflow. Level 2 omits zero-sized slots by contract.
  const std::vector<TensorShapeVector> known_zero_partial_shapes{
      {0, -1, kE2eK},
      {-1, 0, kE2eK},
      {std::numeric_limits<int64_t>::max(), 2, 0, kE2eK},
  };
  for (const auto& dimensions : known_zero_partial_shapes) {
    SCOPED_TRACE(TensorShape{dimensions}.ToString());
    const std::array<WorkspaceInputShape, 1> shapes{
        WorkspaceInputShape::PresentWithShape(TensorShape{dimensions})};
    InlinedVector<WorkspaceRequirement> known_zero_requirements;
    ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
        gsl::make_span(shapes), known_zero_requirements));
    EXPECT_TRUE(known_zero_requirements.empty());
  }

  const std::array<WorkspaceInputShape, 1> shapeless_a{
      WorkspaceInputShape::PresentWithoutShape()};
  InlinedVector<WorkspaceRequirement> shapeless_a_requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
      gsl::make_span(shapeless_a), shapeless_a_requirements));
  EXPECT_TRUE(shapeless_a_requirements.empty());

  const std::array<WorkspaceInputShape, 1> missing_a{WorkspaceInputShape{}};
  InlinedVector<WorkspaceRequirement> missing_a_requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
      gsl::make_span(missing_a), missing_a_requirements));
  EXPECT_TRUE(missing_a_requirements.empty());

  // ---- Runtime: run once and read the workspace size the CUTLASS runner actually requested. ----
  std::vector<MLFloat16> a_data(static_cast<size_t>(kE2eM * kE2eK), MLFloat16(0.25f));
  OrtValue a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{kE2eM, kE2eK}, a_data.data(), OrtMemoryInfo(), &a_value);

  NameMLValMap feeds;
  feeds.emplace("A", a_value);
  const std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  EXPECT_FALSE(GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(op_kernel))
      << "The first run should trace workspace lifetime and use the dynamic fallback.";

  fetches.clear();
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  const size_t runtime = GetMatMulNBitsLastComputeWorkspaceBytes(op_kernel);
  EXPECT_TRUE(GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(op_kernel))
      << "The cached memory pattern did not supply the Level-2 workspace slot.";

  std::cout << "[ WORKSPACE ] Level1(runtime)=" << level1_runtime_workspace
            << " bytes, Level1(persistent prepack)=" << level1->persistent_prepack_bytes
            << " bytes, Level1(temporary prepack)=" << level1->temporary_prepack_bytes
            << " bytes, Level2(declare)=" << level2
            << " bytes, runtime(request)=" << runtime << " bytes" << std::endl;

  // Guard against a trivially-satisfied 0 == 0 == 0: a real CUTLASS GEMM workspace for this config is
  // strictly positive (ceil(M/16)*ceil(N/64)*SPLIT_K_LIMIT*sizeof(float) on SM80). A zero here would
  // mean the GEMV path was taken (runtime never recorded) or the estimate degenerated.
  EXPECT_GT(runtime, static_cast<size_t>(0))
      << "Runtime workspace request was 0 - the CUTLASS GEMM branch did not run (GEMV path?).";

  // The whole point of the pilot: all three must be exactly equal.
  EXPECT_EQ(level1_runtime_workspace, level2)
      << "Level 1 runtime workspace (" << level1_runtime_workspace << ") != Level 2 (" << level2 << ")";
  EXPECT_EQ(level2, runtime)
      << "Level 2 (" << level2 << ") != runtime request (" << runtime << "). A runtime value of 0 "
      << "usually means the GEMV path was taken instead of the CUTLASS GEMM branch.";
  // ---- Empty-output parity on the same dynamic-shape session and kernel. ----
  // Level 1 knows that m == 0 and must return a known zero, including for native SM90.
  const std::optional<Level1MemoryEstimate> zero_m_level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(
          *mm_node, zero_m_a_shape.GetDims(), cuda_ep->GetDeviceProp());
  ASSERT_TRUE(zero_m_level1.has_value());
  ASSERT_TRUE(zero_m_level1->runtime_workspace_bytes.has_value());
  EXPECT_EQ(*zero_m_level1->runtime_workspace_bytes, 0u);

  // Resolve the concrete empty shape through the same production shadow-inference and positional
  // resolver used for shape hints. The two optional holes and the bias must retain their positions.
  const auto zero_m_input_shapes =
      ResolveNodeInputShapes(*mm_node, &graph, zero_m_inferred_shapes);
  ASSERT_NE(zero_m_input_shapes[0].GetShape(), nullptr);
  EXPECT_EQ(*zero_m_input_shapes[0].GetShape(), zero_m_a_shape);
  ASSERT_EQ(zero_m_input_shapes.size(), 6u);
  EXPECT_EQ(zero_m_input_shapes[3].GetState(), WorkspaceInputShapeState::Missing);
  EXPECT_EQ(zero_m_input_shapes[4].GetState(), WorkspaceInputShapeState::Missing);
#if USE_COMPACT_FPA_INTB_GEMM
  EXPECT_EQ(zero_m_input_shapes[5].GetState(), WorkspaceInputShapeState::Missing);
#else
  EXPECT_EQ(zero_m_input_shapes[5].GetState(), WorkspaceInputShapeState::PresentWithShape);
#endif

  InlinedVector<WorkspaceRequirement> zero_m_requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(
      gsl::make_span(zero_m_input_shapes), zero_m_requirements));
  EXPECT_TRUE(zero_m_requirements.empty());

  MLFloat16 unused_zero_m_storage{};
  OrtValue zero_m_a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{0, kE2eK}, &unused_zero_m_storage,
                           OrtMemoryInfo(), &zero_m_a_value);
  NameMLValMap zero_m_feeds;
  zero_m_feeds.emplace("A", zero_m_a_value);
  std::vector<OrtValue> zero_m_fetches;
  ASSERT_STATUS_OK(session.Run(zero_m_feeds, output_names, &zero_m_fetches));
  ASSERT_EQ(zero_m_fetches.size(), 1u);
  ASSERT_TRUE(zero_m_fetches[0].IsTensor());
  EXPECT_EQ(zero_m_fetches[0].Get<Tensor>().Shape(), TensorShape({0, kE2eN}));

  const size_t zero_m_runtime = GetMatMulNBitsLastComputeWorkspaceBytes(op_kernel);
  std::cout << "[ WORKSPACE ZERO-M ] Level1=" << *zero_m_level1->runtime_workspace_bytes
            << " bytes, Level2=empty, runtime(request)=" << zero_m_runtime << " bytes" << std::endl;
  EXPECT_EQ(zero_m_runtime, 0u)
      << "The same kernel must replace its prior positive capture with zero for an m == 0 run.";
}

TEST(MatMulNBitsWorkspace, SequentialChainUsesSharedPlannedWorkspace) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping multi-node workspace test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible.";
  }

  constexpr size_t kNodeCount = 3;
  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});
  const std::string model_bytes = BuildMatMulNBitsChainModelBytes(kNodeCount);

  SessionOptions so;
  so.session_logid = "MatMulNBitsWorkspaceChain";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableStaticWorkspacePreallocation, "1"));
  InferenceSessionWrapper session(so, GetEnvironment());
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const SessionState& session_state = session.GetSessionState();
  const auto matmul_nodes = FindNodesByOpType(session.GetGraph(), "MatMulNBits");
  ASSERT_EQ(matmul_nodes.size(), kNodeCount);

  const SequentialExecutionPlan* execution_plan = session_state.GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);
  InlinedHashSet<int> workspace_pattern_ids;
  std::vector<const OpKernel*> kernels;
  kernels.reserve(kNodeCount);
  for (const Node* node : matmul_nodes) {
    ASSERT_EQ(node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider);
    const OpKernel* kernel = session_state.GetKernel(node->Index());
    ASSERT_NE(kernel, nullptr);
    kernels.push_back(kernel);

    const auto workspace_plan_it = execution_plan->workspace_allocation_plan.find(node->Index());
    ASSERT_NE(workspace_plan_it, execution_plan->workspace_allocation_plan.end());
    ASSERT_EQ(workspace_plan_it->second.size(), static_cast<size_t>(1));
    EXPECT_TRUE(workspace_pattern_ids.insert(workspace_plan_it->second.front().pattern_id).second)
        << "Each MatMulNBits node must have a distinct synthetic pattern ID.";
  }

  std::vector<MLFloat16> a_data(static_cast<size_t>(kE2eM * kE2eK), MLFloat16(0.0f));
  OrtValue a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{kE2eM, kE2eK}, a_data.data(), OrtMemoryInfo(), &a_value);
  NameMLValMap feeds;
  feeds.emplace("A", a_value);
  const std::vector<std::string> output_names{"Y" + std::to_string(kNodeCount - 1)};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  for (const OpKernel* kernel : kernels) {
    EXPECT_GT(GetMatMulNBitsLastComputeWorkspaceBytes(kernel), static_cast<size_t>(0));
    EXPECT_FALSE(GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(kernel))
        << "Every node should use the dynamic fallback while the first run traces lifetimes.";
  }
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const Tensor& dynamic_output = fetches.front().Get<Tensor>();
  ASSERT_EQ(dynamic_output.Shape(), TensorShape({kE2eM, kE2eN}));
  const auto dynamic_output_data = dynamic_output.DataAsSpan<MLFloat16>();
  ASSERT_TRUE(std::any_of(dynamic_output_data.begin(), dynamic_output_data.end(),
                          [](MLFloat16 value) { return value.val != 0; }))
      << "The reference output must be nonzero so data corruption cannot be masked.";
  std::vector<MLFloat16> expected_output(dynamic_output_data.begin(), dynamic_output_data.end());

  int input_index = -1;
  ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx("A", input_index));
  const InlinedHashMap<int, TensorShape>* inferred_shapes = nullptr;
  const MemoryPatternGroup* pattern_group = session_state.GetMemoryPatternGroup(
      gsl::make_span(&a_value, 1), gsl::make_span(&input_index, 1), inferred_shapes);
  ASSERT_NE(pattern_group, nullptr) << "The first run did not cache a memory pattern.";

  std::optional<size_t> shared_workspace_offset;
  for (const Node* node : matmul_nodes) {
    const auto& workspace_plan = execution_plan->workspace_allocation_plan.at(node->Index()).front();
    const MemoryPattern* pattern = pattern_group->GetPatterns(workspace_plan.location);
    ASSERT_NE(pattern, nullptr);
    const MemoryBlock* block = pattern->GetBlock(workspace_plan.pattern_id);
    ASSERT_NE(block, nullptr) << "Synthetic workspace was not recorded in the cached pattern.";
    EXPECT_EQ(block->size_, workspace_plan.allocation_bytes);
    if (shared_workspace_offset.has_value()) {
      EXPECT_EQ(block->offset_, *shared_workspace_offset)
          << "Sequential MatMulNBits workspaces should reuse one non-overlapping region.";
    } else {
      shared_workspace_offset = block->offset_;
    }
  }

  fetches.clear();
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  for (const OpKernel* kernel : kernels) {
    EXPECT_TRUE(GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(kernel))
        << "Every node should resolve its workspace from the cached memory pattern.";
  }
  ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
  const auto planned_output_data = fetches.front().Get<Tensor>().DataAsSpan<MLFloat16>();
  ASSERT_EQ(planned_output_data.size(), expected_output.size());
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(planned_output_data[i].val, expected_output[i].val)
        << "Planned workspace changed output element " << i << ".";
  }
}

// ---------------------------------------------------------------------------
// Test C - fixed-shape specialization via free-dimension override.
// Input A is declared with a symbolic leading dim ("seq"); SessionOptions overrides "seq" -> 512
// before Initialize(). The override rewrites the NodeArg shape into a concrete value that flows into
// BOTH Level 1 (reads the node shape) and Level 2 (given the same static shape), and matches the
// real runtime request. This is *fixed-shape* specialization (exactly 512, not an upper bound).
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, FixedShapeViaFreeDimensionOverride) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping free-dimension-override workspace test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible, so the eligible-path assertions "
                    "cannot hold. Skipping.";
  }

  constexpr int64_t kOverrideM = 512;
  constexpr const char* kSeqDim = "seq";

  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});

  // A declared as ["seq", K] -- genuinely dynamic until the override is applied.
  const std::string model_bytes = BuildMatMulNBitsModelBytes(kSeqDim);

  SessionOptions so;
  so.session_logid = "MatMulNBitsWorkspaceFixedOverride";
  // Fixed-shape specialization: bind the symbolic "seq" dim to exactly kOverrideM at session init.
  so.free_dimension_overrides.push_back(
      FreeDimensionOverride{kSeqDim, FreeDimensionOverrideType::Name, kOverrideM});

  InferenceSessionWrapper session(so, GetEnvironment());
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* mm_node = FindNodeByOpType(graph, "MatMulNBits");
  ASSERT_NE(mm_node, nullptr) << "MatMulNBits node not found in the graph.";
  ASSERT_EQ(mm_node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
      << "MatMulNBits node was not assigned to the CUDA EP.";

  // Confirm the override actually rewrote the symbolic leading dim into the concrete value: this is
  // what makes Level 1 estimable (StaticLeadingDimProduct requires a dim_value).
  const NodeArg* a_arg = mm_node->InputDefs()[0];
  ASSERT_NE(a_arg, nullptr);
  const ONNX_NAMESPACE::TensorShapeProto* a_shape = a_arg->Shape();
  ASSERT_NE(a_shape, nullptr);
  ASSERT_EQ(a_shape->dim_size(), 2);
  ASSERT_TRUE(a_shape->dim(0).has_dim_value())
      << "Free-dimension override did not rewrite the symbolic 'seq' dim into a concrete value.";
  ASSERT_EQ(a_shape->dim(0).dim_value(), kOverrideM);

  // ---- Level 1: estimator reads the (now-overridden) node shape directly. ----
  const std::optional<Level1MemoryEstimate> level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(*mm_node, cuda_ep->GetDeviceProp());
  ASSERT_TRUE(level1.has_value())
      << "Level-1 estimate returned nullopt after the fixed override made the shape static.";
  ASSERT_TRUE(level1->runtime_workspace_bytes.has_value());
  const size_t level1_runtime_workspace = *level1->runtime_workspace_bytes;

  // ---- Level 2: constructed-kernel estimate for the same fixed shape, using the production
  //      positional shape resolver.
  const OpKernel* op_kernel = session.GetSessionState().GetKernel(mm_node->Index());
  ASSERT_NE(op_kernel, nullptr) << "No kernel constructed for the MatMulNBits node.";
  MaxShapeInferenceResult inferred_shapes;
  const auto input_shapes = ResolveNodeInputShapes(*mm_node, &graph, inferred_shapes);
  ASSERT_FALSE(input_shapes.empty());
  const TensorShape* resolved_a_shape = input_shapes[0].GetShape();
  ASSERT_NE(resolved_a_shape, nullptr);
  ASSERT_EQ(resolved_a_shape->NumDimensions(), static_cast<size_t>(2));
  ASSERT_EQ((*resolved_a_shape)[0], kOverrideM)
      << "Overridden NodeArg leading dim did not convert to the fixed value " << kOverrideM << ".";
  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(gsl::make_span(input_shapes), requirements));
  ASSERT_EQ(requirements.size(), static_cast<size_t>(1))
      << "Level-2 DeclareWorkspaceRequirements did not report exactly one workspace slot.";
  // MatMulNBits::DeclareWorkspaceRequirements assigns the single split-K workspace slot_id 0.
  EXPECT_EQ(requirements[0].slot_id, 0) << "Unexpected workspace slot_id.";
  const size_t level2 = requirements[0].size_bytes;

  // ---- Runtime: run once at the fixed M and read the workspace the CUTLASS runner requested. ----
  std::vector<MLFloat16> a_data(static_cast<size_t>(kOverrideM * kE2eK), MLFloat16(0.0f));
  OrtValue a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{kOverrideM, kE2eK}, a_data.data(), OrtMemoryInfo(),
                           &a_value);
  NameMLValMap feeds;
  feeds.emplace("A", a_value);
  const std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  const size_t runtime = GetMatMulNBitsLastComputeWorkspaceBytes(op_kernel);

  std::cout << "[ WORKSPACE ] (fixed override seq=" << kOverrideM
            << ") Level1 runtime=" << level1_runtime_workspace
            << " bytes, Level2=" << level2 << " bytes, runtime=" << runtime << " bytes" << std::endl;

  EXPECT_GT(runtime, static_cast<size_t>(0))
      << "Runtime workspace request was 0 - the CUTLASS GEMM branch did not run (GEMV path?).";
  EXPECT_EQ(level1_runtime_workspace, level2)
      << "Level 1 runtime workspace (" << level1_runtime_workspace << ") != Level 2 (" << level2 << ")";
  EXPECT_EQ(level2, runtime) << "Level 2 (" << level2 << ") != runtime request (" << runtime << ")";
  EXPECT_EQ(level1_runtime_workspace, runtime)
      << "Level 1 runtime workspace (" << level1_runtime_workspace
      << ") != runtime request (" << runtime << ")";
}

// ---------------------------------------------------------------------------
// Test D - dynamic, no override -> graceful fallback.
// Input A declared ["seq", K] with NO override. Level 1 returns nullopt (leading dim is symbolic);
// Level 2 returns empty requirements (symbolic dim -> negative extent -> fallback); and the kernel
// still runs correctly using its live GetScratchBuffer path.
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, DynamicShapeNoOverrideFallsBack) {
  const int device_sm = CudaDeviceComputeCapabilityOrNegative();
  if (device_sm < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping dynamic-shape fallback workspace test.";
  }
  if (device_sm < kMinFpAIntBSm) {
    GTEST_SKIP() << "Device compute capability " << device_sm << " < " << kMinFpAIntBSm
                 << "; MatMulNBits fpA_intB path is not eligible, so the runtime>0 assertion cannot "
                    "hold. Skipping.";
  }

  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});

  // A declared as ["seq", K] and NO free-dimension override -> stays dynamic.
  const std::string model_bytes = BuildMatMulNBitsModelBytes("seq");

  SessionOptions so;
  so.session_logid = "MatMulNBitsWorkspaceDynamicFallback";
  InferenceSessionWrapper session(so, GetEnvironment());
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* mm_node = FindNodeByOpType(graph, "MatMulNBits");
  ASSERT_NE(mm_node, nullptr) << "MatMulNBits node not found in the graph.";
  ASSERT_EQ(mm_node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
      << "MatMulNBits node was not assigned to the CUDA EP.";

  // Sanity: the leading dim really is still symbolic (no dim_value), i.e. genuinely dynamic.
  const NodeArg* a_arg = mm_node->InputDefs()[0];
  ASSERT_NE(a_arg, nullptr);
  const ONNX_NAMESPACE::TensorShapeProto* a_shape = a_arg->Shape();
  ASSERT_NE(a_shape, nullptr);
  ASSERT_EQ(a_shape->dim_size(), 2);
  ASSERT_FALSE(a_shape->dim(0).has_dim_value())
      << "Leading dim unexpectedly became static; the dynamic-fallback path would not be exercised.";

  // ---- Level 1: dynamic leading dim leaves runtime workspace unknown, while shape-independent
  //      prepack memory remains estimable. ----
  const std::optional<Level1MemoryEstimate> level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(*mm_node, cuda_ep->GetDeviceProp());
  ASSERT_TRUE(level1.has_value());
  EXPECT_FALSE(level1->runtime_workspace_bytes.has_value())
      << "Level-1 runtime workspace must be unknown for a dynamic (symbolic) leading dim.";
  EXPECT_GT(level1->persistent_prepack_bytes, size_t{0});
  EXPECT_GT(level1->temporary_prepack_bytes, size_t{0});

  // A separately inferred maximum shape makes the same dynamic node estimable without
  // modifying its canonical shape metadata.
  const TensorShape max_input_shape({256, kE2eK});
  const std::optional<Level1MemoryEstimate> bounded_level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsMemory(
          *mm_node, max_input_shape.GetDims(), cuda_ep->GetDeviceProp());
  ASSERT_TRUE(bounded_level1.has_value());
  EXPECT_TRUE(bounded_level1->runtime_workspace_bytes.has_value());

  // ---- Level 2: the production resolver preserves rank and converts the symbolic leading
  //      dimension to -1, which drives the dynamic fallback -> empty requirements. ----
  const OpKernel* op_kernel = session.GetSessionState().GetKernel(mm_node->Index());
  ASSERT_NE(op_kernel, nullptr) << "No kernel constructed for the MatMulNBits node.";
  MaxShapeInferenceResult inferred_shapes;
  const auto input_shapes = ResolveNodeInputShapes(*mm_node, &graph, inferred_shapes);
  ASSERT_FALSE(input_shapes.empty());
  EXPECT_EQ(input_shapes[0].GetState(), WorkspaceInputShapeState::PresentWithShape);
  const TensorShape* resolved_a_shape = input_shapes[0].GetShape();
  ASSERT_NE(resolved_a_shape, nullptr);
  ASSERT_EQ(resolved_a_shape->NumDimensions(), static_cast<size_t>(2));
  ASSERT_LT((*resolved_a_shape)[0], 0)
      << "Symbolic NodeArg leading dim did not convert to a negative (unknown) extent.";
  EXPECT_EQ((*resolved_a_shape)[1], kE2eK);
  InlinedVector<WorkspaceRequirement> requirements;
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(gsl::make_span(input_shapes), requirements));
  EXPECT_TRUE(requirements.empty())
      << "Level-2 DeclareWorkspaceRequirements must be empty for an unknown (symbolic) leading dim.";

  // ---- Runtime: with a concrete feed the kernel still runs via the live GetScratchBuffer path. ----
  constexpr int64_t kRunM = 256;
  std::vector<MLFloat16> a_data(static_cast<size_t>(kRunM * kE2eK), MLFloat16(0.0f));
  OrtValue a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{kRunM, kE2eK}, a_data.data(), OrtMemoryInfo(), &a_value);
  NameMLValMap feeds;
  feeds.emplace("A", a_value);
  const std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  // The runtime still allocated its scratch dynamically (behavior unchanged); it must be > 0 for the
  // CUTLASS GEMM branch, proving Run() works even though both static levels declined to pre-size it.
  const size_t runtime = GetMatMulNBitsLastComputeWorkspaceBytes(op_kernel);
  std::cout << "[ WORKSPACE ] (dynamic, no override) Level1=nullopt, Level2=empty, runtime=" << runtime
            << " bytes (live GetScratchBuffer)" << std::endl;
  EXPECT_GT(runtime, static_cast<size_t>(0))
      << "Runtime workspace request was 0 - the CUTLASS GEMM branch did not run (GEMV path?).";
  EXPECT_FALSE(GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(op_kernel))
      << "Dynamic-shape execution unexpectedly used a static workspace buffer.";
}

// ---------------------------------------------------------------------------
// Test E - regression safety / control.
// A non-MatMulNBits kernel (Add) does not override DeclareWorkspaceRequirements, so calling it must
// hit the OpKernel base-class default: Status::OK() with an empty requirements vector. This confirms
// the base default is a true no-op and that the new virtual does not affect unrelated kernels.
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, NonMatMulNBitsKernelDeclaresNoWorkspace) {
  // Add is a plain elementwise kernel available on every CUDA GPU, so this control test only needs a
  // device to be present -- it deliberately does NOT apply the SM>=7.5 fpA_intB guard used by the
  // MatMulNBits tests, because Add has no compute-capability requirement.
  if (CudaDeviceComputeCapabilityOrNegative() < 0) {
    GTEST_SKIP() << "No CUDA device available; skipping Add control workspace test.";
  }

  const std::string model_bytes = BuildAddModelBytes();

  SessionOptions so;
  so.session_logid = "AddWorkspaceControl";
  InferenceSessionWrapper session(so, GetEnvironment());
  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const Graph& graph = session.GetGraph();
  const Node* add_node = FindNodeByOpType(graph, "Add");
  ASSERT_NE(add_node, nullptr) << "Add node not found in the graph.";
  ASSERT_EQ(add_node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
      << "Add node was not assigned to the CUDA EP.";

  const OpKernel* op_kernel = session.GetSessionState().GetKernel(add_node->Index());
  ASSERT_NE(op_kernel, nullptr) << "No kernel constructed for the Add node.";

  // Add does not override DeclareWorkspaceRequirements -> base-class no-op: OK() + empty output.
  MaxShapeInferenceResult inferred_shapes;
  const auto input_shapes = ResolveNodeInputShapes(*add_node, &graph, inferred_shapes);
  ASSERT_EQ(input_shapes.size(), 2u);
  InlinedVector<WorkspaceRequirement> requirements;
  // Pre-populate to prove the no-op default clears rather than appends.
  requirements.push_back(WorkspaceRequirement{123, /*slot_id=*/7, /*alignment_bytes=*/0});
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(gsl::make_span(input_shapes), requirements));
  EXPECT_TRUE(requirements.empty())
      << "A kernel that does not override DeclareWorkspaceRequirements must report no workspace.";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM
