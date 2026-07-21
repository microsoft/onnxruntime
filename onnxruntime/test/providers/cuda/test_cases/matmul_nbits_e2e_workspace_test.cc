// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Cross-level agreement test (Major 1, Phase-A memory roadmap, issue microsoft/onnxruntime#29775)
// for the two-level MatMulNBits workspace-estimation pilot: it proves that the workspace *estimator
// function* agrees with the kernel-instance estimate and with the real runtime workspace request.
//
//   Level 1  : EstimateMatMulNBitsWorkspace(node, device_prop)   -- the same estimator function that
//                                                                    CUDAExecutionProvider::GetCapability()
//                                                                    calls at partition time; here it is
//                                                                    invoked DIRECTLY (no kernel).
//   Level 2  : OpKernel::DeclareWorkspaceRequirements(shapes)     -- constructed kernel instance
//                                                                    (virtual dispatch into MatMulNBits).
//   Runtime  : MatMulNBits<MLFloat16>::LastComputeWorkspaceBytes  -- recorded inside the CUTLASS GEMM
//                                                                    branch of the real Compute()
//                                                                    (read through the provider-world
//                                                                    probe in the .cc companion TU).
//
// Scope / what this does NOT prove: this test does not drive a full GetCapability()-based partition-time
// run with a real IResourceAccountant. It exercises the estimator that GetCapability() delegates to, not
// GetCapability()'s own partition-time wiring (device_prop plumbing, resource_accountant invocation),
// which is covered elsewhere / is out of scope for this test's specific claim. In short, it proves the
// estimator itself is consistent with Level 2 and with the real runtime allocation.
//
// This translation unit runs a real InferenceSession, so it includes the core framework headers.
// Those cannot coexist with the CUDA-provider (shared-provider bridge) headers in one TU, so the two
// provider-world pieces it needs (the Level-1 estimate and the runtime probe) are reached through
// slim, bridge-free declarations. It lives in the CUDA-only unit-test module because that is the only
// place these provider-internal symbols are linkable. Requires a real CUDA device; skips otherwise.
// Those cannot coexist with the CUDA-provider (shared-provider bridge) headers in one TU, so the two
// provider-world pieces it needs (the Level-1 estimate and the runtime probe) are reached through
// slim, bridge-free declarations. It lives in the CUDA-only unit-test module because that is the only
// place these provider-internal symbols are linkable. Requires a real CUDA device; skips otherwise.

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <array>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/common/inlined_containers.h"
#include "core/common/span_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/workspace_requirement.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

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
constexpr uint16_t kHalfOne = 0x3C00;  // 1.0 in IEEE-754 half precision.

// Builds a minimal single-node MatMulNBits model (A[M,K] fp16, B int4 initializer, scales fp16
// initializer, Y[M,N] fp16) and returns its serialized ModelProto bytes.
std::string BuildMatMulNBitsModelBytes() {
  const int64_t k_blocks = (kE2eK + kE2eBlockSize - 1) / kE2eBlockSize;  // 32
  const int64_t blob_size = (kE2eBlockSize * kE2eBits + 7) / 8;          // 16

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

  auto set_fp16_shape = [](ONNX_NAMESPACE::ValueInfoProto* vi, int64_t d0, int64_t d1) {
    auto* tt = vi->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(d0);
    shape->add_dim()->set_dim_value(d1);
  };

  // Graph input A and output Y (both fp16, static shapes).
  auto* a = graph->add_input();
  a->set_name("A");
  set_fp16_shape(a, kE2eM, kE2eK);
  auto* y = graph->add_output();
  y->set_name("Y");
  set_fp16_shape(y, kE2eM, kE2eN);

  // B initializer: uint8 {N, k_blocks, blob_size}, zero-filled.
  auto* b = graph->add_initializer();
  b->set_name("B");
  b->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  b->add_dims(kE2eN);
  b->add_dims(k_blocks);
  b->add_dims(blob_size);
  b->mutable_raw_data()->assign(static_cast<size_t>(kE2eN * k_blocks * blob_size), '\0');

  // scales initializer: fp16 {N, k_blocks}, all 1.0.
  auto* scales = graph->add_initializer();
  scales->set_name("scales");
  scales->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  scales->add_dims(kE2eN);
  scales->add_dims(k_blocks);
  const size_t n_scales = static_cast<size_t>(kE2eN * k_blocks);
  std::string scale_raw(n_scales * sizeof(uint16_t), '\0');
  for (size_t i = 0; i < n_scales; ++i) {
    std::memcpy(&scale_raw[i * sizeof(uint16_t)], &kHalfOne, sizeof(uint16_t));
  }
  *scales->mutable_raw_data() = std::move(scale_raw);

  // MatMulNBits node.
  auto* node = graph->add_node();
  node->set_op_type("MatMulNBits");
  node->set_domain("com.microsoft");
  node->set_name("matmul_nbits");
  node->add_input("A");
  node->add_input("B");
  node->add_input("scales");
  node->add_output("Y");
  auto add_int_attr = [node](const char* name, int64_t v) {
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(v);
  };
  add_int_attr("K", kE2eK);
  add_int_attr("N", kE2eN);
  add_int_attr("block_size", kE2eBlockSize);
  add_int_attr("bits", kE2eBits);
  add_int_attr("accuracy_level", 0);

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

}  // namespace

TEST(MatMulNBitsWorkspace, EndToEndWorkspaceAgreement) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available; skipping end-to-end workspace test.";
  }

  // Enable the fpA_intB path via the ENV var (not the session config) so that BOTH Level 1 - which
  // can only read the env var (see the Major-2 known limitation in EstimateMatMulNBitsWorkspace) -
  // and the kernel constructor observe it enabled, keeping the two eligibility decisions in sync.
  ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_FPA_INTB_GEMM", optional<std::string>{"1"}}});

  const std::string model_bytes = BuildMatMulNBitsModelBytes();

  SessionOptions so;
  so.session_logid = "MatMulNBitsWorkspaceE2E";
  InferenceSessionWrapper session(so, GetEnvironment());

  auto cuda_ep = std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{});
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  // Locate the MatMulNBits node and confirm it was assigned to the CUDA EP (fpA_intB eligible).
  const Graph& graph = session.GetGraph();
  const Node* mm_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBits") {
      mm_node = &node;
      break;
    }
  }
  ASSERT_NE(mm_node, nullptr) << "MatMulNBits node not found in the graph.";
  ASSERT_EQ(mm_node->GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
      << "MatMulNBits node was not assigned to the CUDA EP.";

  // ---- Level 1: the estimator function GetCapability() uses, invoked directly on the node + device
  //      properties (this is not a full GetCapability()-driven partition-time run). ----
  const std::optional<size_t> level1 =
      onnxruntime::contrib::cuda::EstimateMatMulNBitsWorkspace(*mm_node, cuda_ep->GetDeviceProp());
  ASSERT_TRUE(level1.has_value()) << "Level-1 estimate returned nullopt for an eligible node.";

  // ---- Level 2: instance-level estimate from the constructed kernel + static input shape. ----
  const OpKernel* op_kernel = session.GetSessionState().GetKernel(mm_node->Index());
  ASSERT_NE(op_kernel, nullptr) << "No kernel constructed for the MatMulNBits node.";

  const std::vector<TensorShape> input_shapes{TensorShape({kE2eM, kE2eK})};
  InlinedVector<WorkspaceRequirement> requirements;
  // DeclareWorkspaceRequirements is virtual on OpKernel; this dispatches into the MatMulNBits override.
  ASSERT_STATUS_OK(op_kernel->DeclareWorkspaceRequirements(AsSpan(input_shapes), requirements));
  ASSERT_EQ(requirements.size(), static_cast<size_t>(1))
      << "Level-2 DeclareWorkspaceRequirements did not report exactly one workspace slot.";
  const size_t level2 = requirements[0].size_bytes;

  // ---- Runtime: run once and read the workspace size the CUTLASS runner actually requested. ----
  std::vector<MLFloat16> a_data(static_cast<size_t>(kE2eM * kE2eK), MLFloat16(0.0f));
  OrtValue a_value;
  CreateMLValue<MLFloat16>(std::array<int64_t, 2>{kE2eM, kE2eK}, a_data.data(), OrtMemoryInfo(), &a_value);

  NameMLValMap feeds;
  feeds.emplace("A", a_value);
  const std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  const size_t runtime = GetMatMulNBitsLastComputeWorkspaceBytes(op_kernel);

  std::cout << "[ WORKSPACE ] Level1(estimate)=" << *level1 << " bytes, Level2(declare)=" << level2
            << " bytes, runtime(request)=" << runtime << " bytes" << std::endl;

  // Guard against a trivially-satisfied 0 == 0 == 0: a real CUTLASS GEMM workspace for this config is
  // strictly positive (ceil(M/16)*ceil(N/64)*SPLIT_K_LIMIT*sizeof(float) on SM80). A zero here would
  // mean the GEMV path was taken (runtime never recorded) or the estimate degenerated.
  EXPECT_GT(runtime, static_cast<size_t>(0))
      << "Runtime workspace request was 0 - the CUTLASS GEMM branch did not run (GEMV path?).";

  // The whole point of the pilot: all three must be exactly equal.
  EXPECT_EQ(*level1, level2) << "Level 1 (" << *level1 << ") != Level 2 (" << level2 << ")";
  EXPECT_EQ(level2, runtime)
      << "Level 2 (" << level2 << ") != runtime request (" << runtime << "). A runtime value of 0 "
      << "usually means the GEMV path was taken instead of the CUTLASS GEMM branch.";
  EXPECT_EQ(*level1, runtime) << "Level 1 (" << *level1 << ") != runtime request (" << runtime << ")";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && USE_FPA_INTB_GEMM
