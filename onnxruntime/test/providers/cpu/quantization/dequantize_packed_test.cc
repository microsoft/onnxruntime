// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/customregistry.h"
#include "core/optimizer/constant_folding.h"
#include "core/providers/cpu/quantization/dequantize_packed.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

struct Oracle {
  const char* name;
  uint64_t hash;
};

// FNV-1a of little-endian FLOAT output for ONNX's deterministic 64-block corpus.
// Generated independently with ONNX 94d0e6056931a8c1dad84104c2e567e0011d04f2's
// reference decoder; its SHA256 results were checked against that revision's
// tests/python/dequantize_packed_test.py GGML da8e2a54 oracle digests.
constexpr Oracle kOracles[] = {
    {"gguf:q4_0", 0x347df38fe6b197c8ULL},
    {"gguf:q4_1", 0x402503a4032dd771ULL},
    {"gguf:q5_0", 0xe63f7ce1222a5832ULL},
    {"gguf:q5_1", 0xc55ad0ec27813b7eULL},
    {"gguf:q8_0", 0xc66b383211b9acd9ULL},
    {"gguf:q8_1", 0x53a6fda8ae6e3cc0ULL},
    {"gguf:q2_k", 0x7a5dff012509ed0cULL},
    {"gguf:q3_k", 0x3358ccb8d9cbcf70ULL},
    {"gguf:q4_k", 0x302bdf61e35eda82ULL},
    {"gguf:q5_k", 0x5b48080e8e112110ULL},
    {"gguf:q6_k", 0xf797c553c90eaadfULL},
    {"gguf:q8_k", 0xc18061311fd7cff0ULL},
    {"gguf:iq2_xxs", 0x18c414c17f6a6e45ULL},
    {"gguf:iq2_xs", 0xe41ddc4d9f0e7907ULL},
    {"gguf:iq3_xxs", 0x2146ae7f760d7dffULL},
    {"gguf:iq1_s", 0x6a03a2146d7c1174ULL},
    {"gguf:iq4_nl", 0x52bad54854a68148ULL},
    {"gguf:iq3_s", 0x28da03729d0f4b0fULL},
    {"gguf:iq2_s", 0x7e3cff214f8e6461ULL},
    {"gguf:iq4_xs", 0x7ed7e2a104676269ULL},
    {"gguf:iq1_m", 0xe3762fec5e716cc4ULL},
    {"gguf:tq1_0", 0x4ae879376d08d00aULL},
    {"gguf:tq2_0", 0xe8d21cc3d67cbbcaULL},
    {"gguf:mxfp4", 0x4121096600239e0fULL},
    {"gguf:nvfp4", 0xcababe2463b6289aULL},
    {"gguf:q1_0", 0x731d5623f5ed41a5ULL},
    {"gguf:q2_0", 0xaf234791af5b326cULL},
};

void Write16(uint8_t* b, uint16_t value) {
  b[0] = static_cast<uint8_t>(value);
  b[1] = static_cast<uint8_t>(value >> 8);
}

std::vector<uint8_t> Corpus(dequantize_packed::Format format, int count) {
  constexpr uint16_t scales[] = {0x3500, 0xb900, 0, 0x3c80, 0x2e66, 0xc280, 1, 0x7bff};
  constexpr uint32_t float_scales[] = {
      0x3dd55555, 0xbe555555, 0, 0x3ec00000, 0x3d088889, 0xbf8aaaab, 0x32aaaaab, 0x46aa9555};
  std::vector<uint8_t> result(static_cast<size_t>(format.bytes) * count);
  for (int i = 0; i < count; ++i) {
    auto* b = result.data() + static_cast<size_t>(i) * format.bytes;
    for (int j = 0; j < format.bytes; ++j) {
      b[j] = static_cast<uint8_t>(j * 73 + i * 29 + (j / 7) * 17 + (i / 3) * 11);
    }
    if (format.id == 15) {
      const uint32_t bits = float_scales[i % 8];
      Write16(b, static_cast<uint16_t>(bits));
      Write16(b + 2, static_cast<uint16_t>(bits >> 16));
    } else if (format.id == 29) {
      for (int k = 0; k < 4; ++k) {
        const uint16_t word = dequantize_packed::U16(b + 48 + 2 * k) & 0x0fff;
        Write16(b + 48 + 2 * k, static_cast<uint16_t>(word | (((scales[i % 8] >> (4 * k)) & 15) << 12)));
      }
    } else if (format.id == 39) {
      b[0] = static_cast<uint8_t>(i * 17);
    } else if (format.id != 40) {
      const int offset = format.id == 10 ? 80 : format.id == 11 ? 108
                                            : format.id == 14   ? 208
                                            : format.id == 34   ? 52
                                            : format.id == 35   ? 64
                                                                : 0;
      Write16(b + offset, scales[i % 8]);
      if (format.id == 3 || format.id == 7 || format.id == 10 || format.id == 12 || format.id == 13) {
        Write16(b + offset + 2, scales[(i + 1) % 8]);
      }
    }
  }
  return result;
}

std::vector<float> DecodeCorpus(dequantize_packed::Format format, const uint8_t* bytes, int count) {
  std::vector<float> output(static_cast<size_t>(format.elements) * count);
  dequantize_packed::Dispatch(format.id, [&](auto kind, auto elements, auto block_bytes) {
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < elements; ++j) {
        output[i * elements + j] = dequantize_packed::Decode<decltype(kind)::value>(bytes + i * block_bytes, j);
      }
    }
  });
  return output;
}

TEST(DequantizePackedTest, AllFormatsIndependentOracleAndUnalignedPayload) {
  for (const auto& oracle : kOracles) {
    SCOPED_TRACE(oracle.name);
    dequantize_packed::Format format{};
    ASSERT_TRUE(dequantize_packed::FindFormat(oracle.name, format));
    const auto corpus = Corpus(format, 64);
    // Every format must work even when its first half/float field is unaligned.
    std::vector<uint8_t> unaligned(corpus.size() + 1);
    std::copy(corpus.begin(), corpus.end(), unaligned.begin() + 1);
    const auto values = DecodeCorpus(format, unaligned.data() + 1, 64);
    std::vector<float> optimized(values.size());
    dequantize_packed::Dispatch(format.id, [&](auto kind, auto elements, auto block_bytes) {
      for (int i = 0; i < 64; ++i) {
        dequantize_packed::DecodeBlockCpu<decltype(kind)::value, decltype(elements)::value>(
            unaligned.data() + 1 + i * block_bytes, optimized.data() + i * elements);
      }
    });
    EXPECT_EQ(std::memcmp(values.data(), optimized.data(), values.size() * sizeof(float)), 0);
    uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int j = 0; j < 4; ++j) {
        hash = (hash ^ ((bits >> (j * 8)) & 255)) * 1099511628211ULL;
      }
    }
    EXPECT_EQ(hash, oracle.hash);
  }
}

std::shared_ptr<CustomRegistry> TestSchema() {
  // Session-local only: neither install unreleased production schemas nor
  // change ORT's released-opset limit while ONNX 29 is not pinned.
  auto registry = std::make_shared<CustomRegistry>();
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("DequantizePacked")
      .SetDomain(kOnnxDomain)
      .SinceVersion(29)
      .Input(0, "x", "", "tensor(uint8)")
      .Output(0, "y", "", "T")
      .Attr("format", "", ONNX_NAMESPACE::AttributeProto::STRING, true)
      .Attr("shape", "", ONNX_NAMESPACE::AttributeProto::INTS, true)
      .Attr("output_dtype", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{1})
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(double)"}, "");
  std::vector<ONNX_NAMESPACE::OpSchema> schemas;
  schemas.push_back(std::move(schema));
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 29, 29));
  return registry;
}

template <typename T>
void RunFormat(const Oracle& oracle, int dtype, bool cuda) {
  dequantize_packed::Format format{};
  ASSERT_TRUE(dequantize_packed::FindFormat(oracle.name, format));
  const auto bytes = Corpus(format, 64);
  // The decoder is independently oracle-checked above. This comparison checks
  // provider execution, tensor ordering and the one final output conversion.
  const auto floats = DecodeCorpus(format, bytes.data(), 64);
  std::vector<T> expected;
  expected.reserve(floats.size());
  for (float value : floats) expected.push_back(T(value));
  OpTester test("DequantizePacked", 29);
  test.SetAllowUnreleasedOnnxOpset();
  test.AddCustomOpRegistry(TestSchema());
  test.AddAttribute("format", std::string(oracle.name));
  test.AddAttribute("shape", std::vector<int64_t>{8, 8, format.elements});
  test.AddAttribute("output_dtype", int64_t{dtype});
  test.AddInput<uint8_t>("x", {static_cast<int64_t>(bytes.size())}, bytes);
  test.AddOutput<T>("y", {8, 8, format.elements}, expected);
  test.SetCustomOutputVerifier([expected, format](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), 1u);
    const auto& tensor = fetches[0].Get<Tensor>();
    ASSERT_EQ(tensor.Shape(), TensorShape({8, 8, format.elements}));
    const auto* actual = tensor.Data<T>();
    for (size_t j = 0; j < expected.size(); ++j) {
      EXPECT_EQ(std::memcmp(actual + j, expected.data() + j, sizeof(T)), 0) << "element " << j;
    }
  });
  SessionOptions options;
  if (cuda) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider());
  test.Config(options).ConfigEps(std::move(providers)).RunWithConfig();
}

void RunAllFormats(bool cuda) {
  for (const auto& oracle : kOracles) {
    SCOPED_TRACE(oracle.name);
    RunFormat<float>(oracle, 1, cuda);
    RunFormat<MLFloat16>(oracle, 10, cuda);
    RunFormat<BFloat16>(oracle, 16, cuda);
    RunFormat<double>(oracle, 11, cuda);
  }
}

TEST(DequantizePackedTest, CpuAllFormatsAllOutputTypes) { RunAllFormats(false); }

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
TEST(DequantizePackedTest, CudaAllFormatsAllOutputTypes) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  RunAllFormats(true);
}
#endif

struct AttributeInfo {
  std::string name = "gguf:q4_0";
  TensorShapeVector shape{32};
  int64_t dtype = 1;
  Status GetAttr(const char*, std::string* out) const {
    *out = name;
    return Status::OK();
  }
  Status GetAttrs(const char*, TensorShapeVector& out) const {
    out = shape;
    return Status::OK();
  }
  template <typename T>
  T GetAttrOrDefault(const char*, T) const { return static_cast<T>(dtype); }
};

TEST(DequantizePackedTest, MetadataAndExactPayloadValidation) {
  AttributeInfo info;
  for (const char* name : {"q4_0", "gguf:Q4_0", "gguf:q4_0:32", "gguf:f32", ""}) {
    info.name = name;
    EXPECT_THROW(dequantize_packed::Attributes attributes(info), OnnxRuntimeException);
  }
  info.name = "gguf:q4_0";
  for (const TensorShapeVector& shape : {TensorShapeVector{}, {-1, 32}, {0, 31}, {33}, {std::numeric_limits<int64_t>::max(), 32}}) {
    info.shape = shape;
    EXPECT_THROW(dequantize_packed::Attributes attributes(info), OnnxRuntimeException);
  }
  info.shape = {32};
  for (int64_t dtype : {int64_t{0}, int64_t{2}, int64_t{7}, int64_t{9}, int64_t{12}, (int64_t{1} << 32) + 1}) {
    info.dtype = dtype;
    EXPECT_THROW(dequantize_packed::Attributes attributes(info), OnnxRuntimeException);
  }
  info.dtype = 1;
  const dequantize_packed::Attributes attributes(info);
  EXPECT_TRUE(attributes.ValidateInput(TensorShape({18})).IsOK());
  EXPECT_FALSE(attributes.ValidateInput(TensorShape({17})).IsOK());
  EXPECT_FALSE(attributes.ValidateInput(TensorShape({19})).IsOK());
  EXPECT_FALSE(attributes.ValidateInput(TensorShape({1, 18})).IsOK());
  info.shape = {0, 32};
  const dequantize_packed::Attributes empty(info);
  EXPECT_EQ(empty.elements, 0);
  EXPECT_TRUE(empty.ValidateInput(TensorShape({0})).IsOK());
  EXPECT_FALSE(empty.ValidateInput(TensorShape({18})).IsOK());
}

TEST(DequantizePackedTest, RoundingAndSpecialValues) {
  using namespace dequantize_packed;
  std::array<uint8_t, 36> b{};
  // Decode with FLOAT arithmetic even when the eventual output is DOUBLE.
  Write16(b.data(), 0x2e66);      // binary16 0.1
  Write16(b.data() + 2, 0xbc00);  // -1
  b[4] = 10;
  EXPECT_EQ(Decode<3>(b.data(), 0), -0x1p-12f);
  // NVFP4 masks bit 7, but only the complete byte 0x7f is the zero sentinel.
  b.fill(0);
  b[0] = 0xff;
  b[4] = 1;
  EXPECT_EQ(Decode<40>(b.data(), 0), 240.0f);
  b[0] = 0x7f;
  EXPECT_EQ(Decode<40>(b.data(), 0), 0.0f);
  // MXFP4 exponent 255 must use the finite half-scale before multiplication.
  b.fill(0);
  b[0] = 255;
  b[1] = 1;
  EXPECT_TRUE(std::isfinite(Decode<39>(b.data(), 0)));
  EXPECT_EQ(Decode<39>(b.data(), 0), FromBits(0x7f000000));
  Write16(b.data(), 0x7c00);
  EXPECT_TRUE(std::isnan(Decode<8>(b.data(), 0)));
}

TEST(DequantizePackedTest, PackedInitializerSurvivesOptimization) {
  std::vector<uint8_t> bytes(18, 0x98);
  Write16(bytes.data(), 0x3c00);
  std::vector<float> expected(32);
  std::fill(expected.begin() + 16, expected.end(), 1.0f);
  OpTester test("DequantizePacked", 29);
  test.SetAllowUnreleasedOnnxOpset();
  test.AddCustomOpRegistry(TestSchema());
  test.AddAttribute("format", std::string("gguf:q4_0"));
  test.AddAttribute("shape", std::vector<int64_t>{32});
  test.AddInput<uint8_t>("x", {18}, bytes, true);
  test.AddOutput<float>("y", {32}, expected);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Level2;
  auto provider = DefaultCpuExecutionProvider();
  auto& graph = test.BuildModel({}, ModelOptions(false, true)).MainGraph();
  ORT_THROW_IF_ERROR(graph.Resolve());
  ConstantFolding folding(*provider, false, options.config_options);
  bool modified = false;
  ORT_THROW_IF_ERROR(folding.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_FALSE(modified);
  EXPECT_EQ(graph.NumberOfNodes(), 1);
  EXPECT_NE(graph.GetConstantInitializer("x", true), nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.Config(options).ConfigEps(std::move(providers)).RunWithConfig();
}

TEST(DequantizePackedTest, EmptyTensor) {
  OpTester test("DequantizePacked", 29);
  test.SetAllowUnreleasedOnnxOpset();
  test.AddCustomOpRegistry(TestSchema());
  test.AddAttribute("format", std::string("gguf:q4_0"));
  test.AddAttribute("shape", std::vector<int64_t>{2, 0, 32});
  test.AddInput<uint8_t>("x", {0}, {});
  test.AddOutput<float>("y", {2, 0, 32}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(providers)).RunWithConfig();
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
