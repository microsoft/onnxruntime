// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Verifies that the CUTLASS weight-only (fpA_intB) path reproduces the dequantized weight matrix
// exactly, element for element, rather than only approximately.
//
// The trick is to run MatMulNBits with A = I and M = K, so Y[k][n] == dequant(B)[n][k]. Every
// quantity is chosen to be exact in fp16 (unit scales, no zero points), so any difference is a
// layout error, not rounding. This is the only check that pins down the offline weight transform
// -- the LDSM row permutation, the sub-byte transpose, the column interleave and the pair
// interleave -- because a GEMM that silently reuses one K tile still produces plausible numbers
// and passes a tolerance-based comparison on some shapes.
#if !defined(ORT_MINIMAL_BUILD) && defined(USE_CUDA)

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

// K spans four 64-element threadblock K tiles, which is what caught the B pipeline stalling on
// the first tile. N is a multiple of 128 so 2-bit weights are fpA_intB-eligible.
constexpr int64_t kN = 128, kK = 256;
// A = I makes M = K, so 512 reaches the large-M INT8 path (dequantize + one cuBLAS GEMM, see
// ep.cuda.fpa_intb_dequant_gemm_min_m).
constexpr int64_t kLargeMK = 512;

// dequant_gemm_min_m is the ep.cuda.fpa_intb_dequant_gemm_min_m session option; nullptr leaves it unset. A non-null
// expected_error expects session creation to fail with that message.
void ExpectExactDequantizedWeights(int64_t bits, bool deterministic = false, int64_t k_dim = kK,
                                   const char* dequant_gemm_min_m = "512", const char* expected_error = nullptr) {
  SCOPED_TRACE("bits=" + std::to_string(bits) + " K=" + std::to_string(k_dim) + " dequant_gemm_min_m=" +
               (dequant_gemm_min_m != nullptr ? dequant_gemm_min_m : "<unset>"));
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  const int64_t block_size = bits == 2 ? 64 : 32;
  const int64_t k_blocks = k_dim / block_size;
  const int64_t blob_size = block_size * bits / 8;
  const int per_byte = static_cast<int>(8 / bits);
  const int mask = (1 << bits) - 1;
  const int zero_point = 1 << (bits - 1);

  std::vector<int> code(static_cast<size_t>(kN * k_dim));
  std::vector<uint8_t> b(static_cast<size_t>(kN * k_dim * bits / 8), 0);
  unsigned seed = 7;
  for (int64_t i = 0; i < kN * k_dim; ++i) {
    seed = seed * 1103515245u + 12345u;
    int c = static_cast<int>((seed >> 16) & mask);
    code[static_cast<size_t>(i)] = c;
    b[static_cast<size_t>(i / per_byte)] |= static_cast<uint8_t>(c << (bits * (i % per_byte)));
  }

  std::vector<MLFloat16> scales(static_cast<size_t>(kN * k_blocks), MLFloat16(1.0f));
  std::vector<MLFloat16> a(static_cast<size_t>(k_dim * k_dim), MLFloat16(0.0f));
  for (int64_t i = 0; i < k_dim; ++i) a[static_cast<size_t>(i * k_dim + i)] = MLFloat16(1.0f);

  std::vector<MLFloat16> expected(static_cast<size_t>(k_dim * kN));
  for (int64_t k = 0; k < k_dim; ++k) {
    for (int64_t n = 0; n < kN; ++n) {
      expected[static_cast<size_t>(k * kN + n)] =
          MLFloat16(static_cast<float>(code[static_cast<size_t>(n * k_dim + k)] - zero_point));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", k_dim);
  test.AddAttribute<int64_t>("N", kN);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("accuracy_level", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("A", {k_dim, k_dim}, a, false);
  test.AddInput<uint8_t>("B", {kN, k_blocks, blob_size}, b, true);
  test.AddInput<MLFloat16>("scales", {kN, k_blocks}, scales, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<MLFloat16>();
  test.AddOutput<MLFloat16>("Y", {k_dim, kN}, expected);
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.emplace_back(DefaultCudaExecutionProvider());
  SessionOptions session_options;
  session_options.use_per_session_threads = false;
  session_options.use_deterministic_compute = deterministic;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  if (dequant_gemm_min_m != nullptr) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsCudaFpAIntBDequantGemmMinM,
                                                                   dequant_gemm_min_m));
  }
  test.Config(session_options);
  test.ConfigEps(std::move(eps));
  if (expected_error != nullptr) {
    test.Config(OpTester::ExpectResult::kExpectFailure, expected_error);
  }
  test.RunWithConfig();
}

#if USE_FPA_INTB_GEMM
// Only the CUDA graph test below uses these helpers. Builds without the fpA_intB kernels skip that test, and an
// unused helper would fail them under -Werror=unused-function.
class ScopedCudaStream {
 public:
  ScopedCudaStream() { ORT_ENFORCE(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess); }
  ~ScopedCudaStream() { (void)cudaStreamDestroy(stream_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedCudaStream);
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// MatMulNBits(A[M, K] fp16, B, scales) -> Y[M, N] with 8-bit constant weights and block size 32.
std::string BuildInt8MatMulNBitsModel(int64_t n, int64_t k, const std::vector<uint8_t>& codes,
                                      const std::vector<MLFloat16>& scales) {
  constexpr int64_t kBlockSize = 32;
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(13);
  auto* contrib_opset = model.add_opset_import();
  contrib_opset->set_domain(kMSDomain);
  contrib_opset->set_version(1);
  auto& graph = *model.mutable_graph();
  graph.set_name("int8_large_m_cuda_graph");

  auto set_value = [](ONNX_NAMESPACE::ValueInfoProto* value, const char* name, int64_t columns) {
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    type->mutable_shape()->add_dim()->set_dim_param("M");
    type->mutable_shape()->add_dim()->set_dim_value(columns);
  };
  set_value(graph.add_input(), "A", k);
  set_value(graph.add_output(), "Y", n);

  auto* weights = graph.add_initializer();
  weights->set_name("B");
  weights->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  for (int64_t dim : {n, k / kBlockSize, kBlockSize}) weights->add_dims(dim);
  weights->set_raw_data(codes.data(), codes.size());
  auto* scale_tensor = graph.add_initializer();
  scale_tensor->set_name("scales");
  scale_tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t dim : {n, k / kBlockSize}) scale_tensor->add_dims(dim);
  scale_tensor->set_raw_data(scales.data(), scales.size() * sizeof(MLFloat16));

  auto* node = graph.add_node();
  node->set_op_type("MatMulNBits");
  node->set_domain(kMSDomain);
  for (const char* input : {"A", "B", "scales"}) node->add_input(input);
  node->add_output("Y");
  for (const auto& [name, value] :
       {std::pair<const char*, int64_t>{"K", k}, {"N", n}, {"bits", 8}, {"block_size", kBlockSize}}) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  }
  return model.SerializeAsString();
}
#endif  // USE_FPA_INTB_GEMM

}  // namespace

TEST(MatMulNBitsFpAIntBLayout, TensorCoreSessionOptions) {
  if (!HasCudaEnvironment(900)) GTEST_SKIP() << "Tensor-core INT8 GEMV requires SM90";
  for (const char* disable : {"0", "1"}) {
    ScopedEnvironmentVariables env{EnvVarMap{{"ORT_DISABLE_FPA_INTB_GEMV_TC", "invalid"}}};
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", kK);
    test.AddAttribute<int64_t>("N", kN);
    test.AddAttribute<int64_t>("block_size", 32);
    test.AddAttribute<int64_t>("bits", 8);
    std::vector<BFloat16> a(8 * kK, BFloat16(0.0f));
    for (int i = 0; i < 8; ++i) a[i * kK + i] = BFloat16(1.0f);
    test.AddInput<BFloat16>("A", {8, kK}, a);
    test.AddInput<uint8_t>("B", {kN, kK / 32, 32}, std::vector<uint8_t>(kN * kK, 129), true);
    test.AddInput<BFloat16>("scales", {kN, kK / 32}, std::vector<BFloat16>(kN * kK / 32, BFloat16(1.0f)), true);
    test.AddOutput<BFloat16>("Y", {8, kN}, std::vector<BFloat16>(8 * kN, BFloat16(1.0f)));
    test.SetOutputTolerance(0.0f, 0.0f);
    SessionOptions options;
    options.use_deterministic_compute = true;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.fpa_intb_gemm", "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.disable_fpa_intb_gemv_tc", disable));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.emplace_back(DefaultCudaExecutionProvider());
    test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(MatMulNBitsFpAIntBLayout, Int4WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(4);
}

TEST(MatMulNBitsFpAIntBLayout, Int2WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(2);
}

TEST(MatMulNBitsFpAIntBLayout, DeterministicWeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  for (int64_t bits : {2, 4, 8}) {
    ExpectExactDequantizedWeights(bits, true);
  }
}

// K = M = 512: eight 64-wide K tiles through the dequantize + cuBLAS path, and through the row-chunked CUTLASS path
// (threshold 0 turns the large-M path off). The threshold is set explicitly so that the path is also covered on
// architectures where it is off by default.
TEST(MatMulNBitsFpAIntBLayout, Int8LargeMDequantGemmRoundTripsExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(8, false, kLargeMK, "512");
  ExpectExactDequantizedWeights(8, false, kLargeMK, "0");
}

// The session option takes precedence over the environment, which is consulted only when no entry is set; an invalid
// value from either source fails kernel construction. Only the fpA_intB implementation parses the option, so a build
// without it has nothing to fail on.
TEST(MatMulNBitsFpAIntBLayout, Int8LargeMDequantGemmSessionOption) {
#if !USE_FPA_INTB_GEMM
  GTEST_SKIP() << "Requires fpA_intB kernels";
#else
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  constexpr const char* kEnv = "ORT_FPA_INTB_DEQUANT_GEMM_MIN_M";
  {
    ScopedEnvironmentVariables env{EnvVarMap{{kEnv, "invalid"}}};
    ExpectExactDequantizedWeights(8, false, kLargeMK, "512");
    ExpectExactDequantizedWeights(8, false, kLargeMK, "0");
    ExpectExactDequantizedWeights(8, false, kLargeMK, nullptr, kEnv);
  }
  for (const char* env_value : {"0", "512"}) {
    ScopedEnvironmentVariables env{EnvVarMap{{kEnv, env_value}}};
    ExpectExactDequantizedWeights(8, false, kLargeMK, nullptr);
  }
  for (const char* invalid : {"-1", "", "1x", "many", "2147483648"}) {
    SCOPED_TRACE(std::string("value='") + invalid + "'");
    ExpectExactDequantizedWeights(8, false, kLargeMK, invalid, kOrtSessionOptionsCudaFpAIntBDequantGemmMinM);
  }
#endif
}

// A CUDA-graph-enabled session captures M above the profiled range on the CUTLASS path, which cannot profile while
// capturing. The warmup runs take the dequantize route for this M, so the route has to tune the CUTLASS bucket itself.
// The default profiled range ends at M = 2048, so M = 4096 is outside it. One-hot rows of A keep every result exact
// on both paths, and each run scales A so that a replay that read stale input would be caught.
TEST(MatMulNBitsFpAIntBLayout, Int8LargeMDequantGemmCudaGraphCapture) {
#if !USE_FPA_INTB_GEMM
  GTEST_SKIP() << "Requires fpA_intB kernels";
#else
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";

  constexpr int64_t kM = 4096, kGraphN = 256, kGraphK = 512, kBlockSize = 32, kZeroPoint = 128;
  std::vector<uint8_t> codes(static_cast<size_t>(kGraphN * kGraphK));
  unsigned seed = 11;
  for (uint8_t& code : codes) {
    seed = seed * 1103515245u + 12345u;
    code = static_cast<uint8_t>(seed >> 16);
  }
  const std::vector<MLFloat16> scales(static_cast<size_t>(kGraphN * kGraphK / kBlockSize), MLFloat16(1.0f));
  const std::string model = BuildInt8MatMulNBitsModel(kGraphN, kGraphK, codes, scales);

  ScopedCudaStream stream;
  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.AddConfigEntry(kOrtSessionOptionsCudaFpAIntBGemm, "1");
  options.AddConfigEntry(kOrtSessionOptionsCudaFpAIntBDequantGemmMinM, "512");
  options.SetIntraOpNumThreads(1);
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
  // The plugin library does not export the in-tree provider bridge that AppendExecutionProvider_CUDA_V2 loads, so
  // select the device that unit test main registered.
  Ort::ConstEpDevice cuda_device;
  for (const auto& ep_device : ort_env->GetEpDevices()) {
    if (std::string_view{ep_device.EpName()} == kCudaExecutionProvider) {
      cuda_device = ep_device;
      break;
    }
  }
  if (cuda_device == nullptr) GTEST_SKIP() << "CUDA plugin EP is not registered";
  const std::unordered_map<std::string, std::string> provider_options{
      {"enable_cuda_graph", "1"},
      {"user_compute_stream", std::to_string(reinterpret_cast<uintptr_t>(stream.get()))}};
  options.AppendExecutionProvider_V2(*ort_env, {cuda_device}, provider_options);
#else
  Ort::CUDAProviderOptions cuda_options;
  cuda_options.Update({{"enable_cuda_graph", "1"}});
  cuda_options.UpdateWithValue("user_compute_stream", stream.get());
  options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#endif
  Ort::Session session(*ort_env, model.data(), model.size(), options);

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
  auto allocator = ort_env->GetSharedAllocator(cuda_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
#else
  Ort::MemoryInfo memory_info("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session, memory_info);
#endif
  ASSERT_NE(allocator, nullptr);
  const std::array<int64_t, 2> a_shape{kM, kGraphK}, y_shape{kM, kGraphN};
  auto a_tensor = Ort::Value::CreateTensor(allocator, a_shape.data(), a_shape.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto y_tensor = Ort::Value::CreateTensor(allocator, y_shape.data(), y_shape.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  Ort::IoBinding binding(session);
  binding.BindInput("A", a_tensor);
  binding.BindOutput("Y", y_tensor);

  auto run_and_verify = [&](const char* graph_id, float alpha) {
    SCOPED_TRACE(std::string("gpu_graph_id=") + graph_id + " alpha=" + std::to_string(alpha));
    std::vector<MLFloat16> a(static_cast<size_t>(kM * kGraphK), MLFloat16(0.0f));
    for (int64_t row = 0; row < kM; ++row) a[static_cast<size_t>(row * kGraphK + row % kGraphK)] = MLFloat16(alpha);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(a_tensor.GetTensorMutableRawData(), a.data(), a.size() * sizeof(MLFloat16),
                                           cudaMemcpyHostToDevice, stream.get()));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(y_tensor.GetTensorMutableRawData(), 0xff,
                                           static_cast<size_t>(kM * kGraphN) * sizeof(MLFloat16), stream.get()));
    // Capture must start with no pending host-issued copies on the user stream.
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("gpu_graph_id", graph_id);
    ASSERT_NO_THROW(session.Run(run_options, binding));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));
    std::vector<MLFloat16> y(static_cast<size_t>(kM * kGraphN));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(y.data(), y_tensor.GetTensorMutableRawData(), y.size() * sizeof(MLFloat16),
                                      cudaMemcpyDeviceToHost));
    for (int64_t row = 0; row < kM; ++row) {
      for (int64_t col = 0; col < kGraphN; ++col) {
        const int code = codes[static_cast<size_t>(col * kGraphK + row % kGraphK)];
        ASSERT_EQ(y[static_cast<size_t>(row * kGraphN + col)].ToFloat(), alpha * static_cast<float>(code - kZeroPoint))
            << "row=" << row << " column=" << col;
      }
    }
  };

  // Two regular warmup runs, then a capture run, then replays.
  for (float alpha : {1.0f, 2.0f, 4.0f, -1.0f, 0.5f}) {
    ASSERT_NO_FATAL_FAILURE(run_and_verify("1", alpha));
  }
#endif
}

}  // namespace test
}  // namespace onnxruntime

#endif
