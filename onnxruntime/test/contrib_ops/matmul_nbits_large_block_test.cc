// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CPU MatMulNBits with authored power-of-two block sizes larger than 256. The kernel executes these as 256-element
// sub-blocks that repeat each logical block's scale and zero point. Each case compares
//   1. an independent float decode of the authored weights,
//   2. an independently serialized block-256 equivalent model, and
//   3. the unchanged large-block model,
// requiring (3) to match (2) bit-for-bit in the same compute mode.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/float16.h"
#include "core/framework/allocator.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

enum class ZeroPoint {
  kAbsent,    // implicit midpoint 2^(bits-1)
  kMidpoint,  // explicit uint8 zero points equal to the midpoint
  kRandom,    // explicit uint8 zero points
  kFloat,     // float zero points (T1), 4-bit only
};

// Authored quantization, kept as unpacked values so every serialized layout is derived independently.
struct Weights {
  int64_t n{};
  int64_t k{};
  int64_t bits{};
  int64_t block_size{};  // authored (logical) block size
  int64_t k_blocks{};
  ZeroPoint zp_mode{};
  std::vector<uint8_t> q;       // [n, k_blocks * block_size]; padding holds the block's integer zero point
  std::vector<float> scales;    // [n, k_blocks]
  std::vector<uint8_t> zp_int;  // [n, k_blocks], used for kAbsent/kMidpoint/kRandom
  std::vector<float> zp_float;  // [n, k_blocks], used for kFloat
};

Weights MakeWeights(int64_t n, int64_t k, int64_t bits, int64_t block_size, ZeroPoint zp_mode, uint32_t seed) {
  Weights w;
  w.n = n;
  w.k = k;
  w.bits = bits;
  w.block_size = block_size;
  w.k_blocks = (k + block_size - 1) / block_size;
  w.zp_mode = zp_mode;

  std::mt19937 rng(seed);
  const int qmax = (1 << bits) - 1;
  const int midpoint = 1 << (bits - 1);
  std::uniform_int_distribution<int> q_dist(0, qmax);
  std::uniform_real_distribution<float> scale_dist(0.002f, 0.02f);
  std::uniform_real_distribution<float> zp_float_dist(0.0f, static_cast<float>(qmax));

  const int64_t row = w.k_blocks * block_size;
  w.scales.resize(static_cast<size_t>(n * w.k_blocks));
  w.zp_int.resize(static_cast<size_t>(n * w.k_blocks));
  w.zp_float.resize(static_cast<size_t>(n * w.k_blocks));
  for (int64_t i = 0; i < n * w.k_blocks; ++i) {
    w.scales[i] = scale_dist(rng);
    w.zp_int[i] = static_cast<uint8_t>(zp_mode == ZeroPoint::kRandom ? q_dist(rng) : midpoint);
    w.zp_float[i] = zp_float_dist(rng);
  }

  w.q.resize(static_cast<size_t>(n * row));
  for (int64_t c = 0; c < n; ++c) {
    for (int64_t kk = 0; kk < row; ++kk) {
      const int64_t g = kk / block_size;
      w.q[c * row + kk] = kk < k ? static_cast<uint8_t>(q_dist(rng)) : w.zp_int[c * w.k_blocks + g];
    }
  }
  return w;
}

float Decode(const Weights& w, int64_t c, int64_t kk) {
  const int64_t g = kk / w.block_size;
  const float q = static_cast<float>(w.q[c * w.k_blocks * w.block_size + kk]);
  const float zp = w.zp_mode == ZeroPoint::kFloat ? w.zp_float[c * w.k_blocks + g]
                                                  : static_cast<float>(w.zp_int[c * w.k_blocks + g]);
  return (q - zp) * w.scales[c * w.k_blocks + g];
}

// MatMulNBits inputs for `block_size`, which must evenly divide the authored block size (or equal it).
struct NBitsTensors {
  int64_t block_size{};
  int64_t k_blocks{};
  std::vector<uint8_t> b;          // [n, k_blocks, block_size * bits / 8]
  std::vector<float> scales;       // [n, k_blocks]
  std::vector<uint8_t> zp_packed;  // [n, ceil(k_blocks * bits / 8)]
  std::vector<float> zp_float;     // [n, k_blocks]
};

NBitsTensors Serialize(const Weights& w, int64_t block_size) {
  NBitsTensors t;
  t.block_size = block_size;
  t.k_blocks = (w.k + block_size - 1) / block_size;
  const int64_t per_byte = 8 / w.bits;
  const int64_t blob = block_size * w.bits / 8;
  const int64_t zp_row = (t.k_blocks * w.bits + 7) / 8;
  const int64_t src_row = w.k_blocks * w.block_size;
  t.b.assign(static_cast<size_t>(w.n * t.k_blocks * blob), 0);
  t.scales.resize(static_cast<size_t>(w.n * t.k_blocks));
  t.zp_float.resize(static_cast<size_t>(w.n * t.k_blocks));
  t.zp_packed.assign(static_cast<size_t>(w.n * zp_row), 0);
  for (int64_t c = 0; c < w.n; ++c) {
    for (int64_t kk = 0; kk < t.k_blocks * block_size; ++kk) {
      const uint8_t q = w.q[c * src_row + kk];
      t.b[c * t.k_blocks * blob + kk / per_byte] |= static_cast<uint8_t>(q << ((kk % per_byte) * w.bits));
    }
    for (int64_t h = 0; h < t.k_blocks; ++h) {
      const int64_t g = h * block_size / w.block_size;
      t.scales[c * t.k_blocks + h] = w.scales[c * w.k_blocks + g];
      t.zp_float[c * t.k_blocks + h] = w.zp_float[c * w.k_blocks + g];
      t.zp_packed[c * zp_row + h / per_byte] |=
          static_cast<uint8_t>(w.zp_int[c * w.k_blocks + g] << ((h % per_byte) * w.bits));
    }
  }
  return t;
}

struct RunConfig {
  int64_t accuracy_level{1};
  bool fp16{false};
  bool disable_prepacking{false};
  bool quant_params_as_inputs{false};  // scales / zero_points provided at run time
  bool external_data{false};
  bool has_bias{false};
  int64_t batch{1};
  int64_t m{3};
};

void SetValueInfo(ONNX_NAMESPACE::ValueInfoProto& info, const std::string& name, int32_t type,
                  const std::vector<int64_t>& dims) {
  info.set_name(name);
  auto* tensor = info.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  auto* shape = tensor->mutable_shape();
  for (int64_t d : dims) {
    shape->add_dim()->set_dim_value(d);
  }
}

template <typename T>
std::string ToBytes(const std::vector<T>& data) {
  return std::string(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

std::vector<MLFloat16> ToFp16(const std::vector<float>& v) {
  std::vector<MLFloat16> out;
  out.reserve(v.size());
  for (float f : v) {
    out.emplace_back(f);
  }
  return out;
}

struct Tensors {
  std::vector<float> a;
  std::vector<float> bias;
};

// Builds a single-node MatMulNBits model. Returns the model and, for external data, the data file contents.
ONNX_NAMESPACE::ModelProto BuildModel(const Weights& w, const NBitsTensors& t, const RunConfig& cfg,
                                      const Tensors& io, const std::string& data_file, std::string* data_blob,
                                      int64_t block_size_attr = -1, bool add_g_idx = false) {
  const int32_t float_type = cfg.fp16 ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                      : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(21);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto& graph = *model.mutable_graph();
  graph.set_name("matmul_nbits_large_block");
  SetValueInfo(*graph.add_input(), "A", float_type, {cfg.batch, cfg.m, w.k});
  SetValueInfo(*graph.add_output(), "Y", float_type, {cfg.batch, cfg.m, w.n});

  auto add_initializer = [&](const std::string& name, int32_t type, const std::vector<int64_t>& dims,
                             const std::string& bytes) {
    auto* init = graph.add_initializer();
    init->set_name(name);
    init->set_data_type(type);
    for (int64_t d : dims) {
      init->add_dims(d);
    }
    if (cfg.external_data && data_blob != nullptr) {
      init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
      auto add_entry = [&](const char* key, const std::string& value) {
        auto* entry = init->add_external_data();
        entry->set_key(key);
        entry->set_value(value);
      };
      add_entry("location", data_file);
      add_entry("offset", std::to_string(data_blob->size()));
      add_entry("length", std::to_string(bytes.size()));
      data_blob->append(bytes);
      // Keep offsets aligned for memory mapping.
      data_blob->append((4096 - data_blob->size() % 4096) % 4096, '\0');
    } else {
      init->set_raw_data(bytes);
    }
  };

  auto float_bytes = [&](const std::vector<float>& v) { return cfg.fp16 ? ToBytes(ToFp16(v)) : ToBytes(v); };

  const int64_t blob = t.block_size * w.bits / 8;
  const int64_t zp_row = (t.k_blocks * w.bits + 7) / 8;
  add_initializer("B", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {w.n, t.k_blocks, blob}, ToBytes(t.b));

  const bool has_zp = w.zp_mode != ZeroPoint::kAbsent;
  const bool float_zp = w.zp_mode == ZeroPoint::kFloat;
  if (cfg.quant_params_as_inputs) {
    SetValueInfo(*graph.add_input(), "scales", float_type, {w.n, t.k_blocks});
    if (has_zp) {
      SetValueInfo(*graph.add_input(), "zero_points",
                   float_zp ? float_type : ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                   float_zp ? std::vector<int64_t>{w.n, t.k_blocks} : std::vector<int64_t>{w.n, zp_row});
    }
  } else {
    add_initializer("scales", float_type, {w.n, t.k_blocks}, float_bytes(t.scales));
    if (has_zp) {
      if (float_zp) {
        add_initializer("zero_points", float_type, {w.n, t.k_blocks}, float_bytes(t.zp_float));
      } else {
        add_initializer("zero_points", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {w.n, zp_row},
                        ToBytes(t.zp_packed));
      }
    }
  }
  if (cfg.has_bias) {
    add_initializer("bias", float_type, {w.n}, float_bytes(io.bias));
  }
  if (add_g_idx) {
    std::vector<int32_t> g_idx(static_cast<size_t>(w.k));
    for (int64_t i = 0; i < w.k; ++i) {
      g_idx[i] = static_cast<int32_t>(i / t.block_size);
    }
    add_initializer("g_idx", ONNX_NAMESPACE::TensorProto_DataType_INT32, {w.k}, ToBytes(g_idx));
  }

  auto* node = graph.add_node();
  node->set_op_type("MatMulNBits");
  node->set_domain(kMSDomain);
  node->set_name("nbits");
  for (const char* input : {"A", "B", "scales"}) {
    node->add_input(input);
  }
  node->add_input(has_zp ? "zero_points" : "");
  node->add_input(add_g_idx ? "g_idx" : "");
  node->add_input(cfg.has_bias ? "bias" : "");
  node->add_output("Y");
  auto add_int_attr = [&](const char* name, int64_t value) {
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(value);
  };
  add_int_attr("K", w.k);
  add_int_attr("N", w.n);
  add_int_attr("bits", w.bits);
  add_int_attr("block_size", block_size_attr > 0 ? block_size_attr : t.block_size);
  add_int_attr("accuracy_level", cfg.accuracy_level);
  return model;
}

template <typename T>
OrtValue MakeValue(const std::vector<int64_t>& dims, const std::vector<T>& data) {
  OrtValue value;
  CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, data, &value);
  return value;
}

struct RunResult {
  Status status;
  std::vector<float> y;
  size_t used_shared_prepacked{0};
  size_t prepacked{0};
};

RunResult RunModel(const Weights& w, const NBitsTensors& t, const RunConfig& cfg, const Tensors& io,
                   PrepackedWeightsContainer* container = nullptr, const OrtValue* shared_b = nullptr,
                   int64_t block_size_attr = -1, bool add_g_idx = false) {
  RunResult result;
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Default;
  so.intra_op_param.thread_pool_size = 4;
  if (cfg.disable_prepacking) {
    EXPECT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
  }
  if (shared_b != nullptr) {
    EXPECT_STATUS_OK(so.AddInitializer("B", shared_b));
  }

  InferenceSessionWrapper session{so, GetEnvironment()};
  if (container != nullptr) {
    EXPECT_STATUS_OK(session.AddPrePackedWeightsContainer(container));
  }

  std::filesystem::path dir;
  if (cfg.external_data) {
    dir = std::filesystem::temp_directory_path() /
          ("ort_nbits_large_block_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    std::string blob;
    const auto model = BuildModel(w, t, cfg, io, "weights.bin", &blob, block_size_attr, add_g_idx);
    std::ofstream(dir / "weights.bin", std::ios::binary).write(blob.data(), static_cast<std::streamsize>(blob.size()));
    std::ofstream model_file(dir / "model.onnx", std::ios::binary);
    model.SerializeToOstream(&model_file);
    model_file.close();
    result.status = session.Load((dir / "model.onnx").native());
  } else {
    std::string bytes;
    BuildModel(w, t, cfg, io, "", nullptr, block_size_attr, add_g_idx).SerializeToString(&bytes);
    result.status = session.Load(bytes.data(), static_cast<int>(bytes.size()));
  }

  if (result.status.IsOK()) {
    result.status = session.Initialize();
  }

  if (result.status.IsOK()) {
    NameMLValMap feeds;
    const std::vector<int64_t> a_dims{cfg.batch, cfg.m, w.k};
    feeds["A"] = cfg.fp16 ? MakeValue(a_dims, ToFp16(io.a)) : MakeValue(a_dims, io.a);
    if (cfg.quant_params_as_inputs) {
      const std::vector<int64_t> s_dims{w.n, t.k_blocks};
      feeds["scales"] = cfg.fp16 ? MakeValue(s_dims, ToFp16(t.scales)) : MakeValue(s_dims, t.scales);
      if (w.zp_mode == ZeroPoint::kFloat) {
        feeds["zero_points"] = cfg.fp16 ? MakeValue(s_dims, ToFp16(t.zp_float)) : MakeValue(s_dims, t.zp_float);
      } else if (w.zp_mode != ZeroPoint::kAbsent) {
        feeds["zero_points"] = MakeValue({w.n, (t.k_blocks * w.bits + 7) / 8}, t.zp_packed);
      }
    }
    std::vector<OrtValue> fetches;
    const std::vector<std::string> output_names{"Y"};
    RunOptions run_options;
    result.status = session.Run(run_options, feeds, output_names, &fetches);
    if (result.status.IsOK()) {
      const Tensor& y = fetches[0].Get<Tensor>();
      if (cfg.fp16) {
        for (const MLFloat16& v : y.DataAsSpan<MLFloat16>()) {
          result.y.push_back(v.ToFloat());
        }
      } else {
        const auto span = y.DataAsSpan<float>();
        result.y.assign(span.begin(), span.end());
      }
    }
    result.used_shared_prepacked = session.GetSessionState().GetUsedSharedPrePackedWeightCounter();
    result.prepacked = session.GetSessionState().GetNumberOfPrepacksCounter();
  }

  if (!dir.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  return result;
}

Tensors MakeIo(const Weights& w, const RunConfig& cfg, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Tensors io;
  io.a.resize(static_cast<size_t>(cfg.batch * cfg.m * w.k));
  for (auto& v : io.a) {
    v = dist(rng);
  }
  if (cfg.has_bias) {
    io.bias.resize(static_cast<size_t>(w.n));
    for (auto& v : io.bias) {
      v = dist(rng);
    }
  }
  return io;
}

std::vector<float> Reference(const Weights& w, const RunConfig& cfg, const Tensors& io) {
  std::vector<float> y(static_cast<size_t>(cfg.batch * cfg.m * w.n));
  for (int64_t b = 0; b < cfg.batch; ++b) {
    for (int64_t m = 0; m < cfg.m; ++m) {
      for (int64_t c = 0; c < w.n; ++c) {
        double sum = cfg.has_bias ? io.bias[c] : 0.0;
        for (int64_t kk = 0; kk < w.k; ++kk) {
          sum += static_cast<double>(io.a[(b * cfg.m + m) * w.k + kk]) * Decode(w, c, kk);
        }
        y[(b * cfg.m + m) * w.n + c] = static_cast<float>(sum);
      }
    }
  }
  return y;
}

void ExpectBitwiseEqual(const std::vector<float>& actual, const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i], expected[i]) << "index " << i;
  }
}

void ExpectClose(const std::vector<float>& actual, const std::vector<float>& expected, float rel_tol) {
  ASSERT_EQ(actual.size(), expected.size());
  float max_abs = 0.0f;
  for (float v : expected) {
    max_abs = std::max(max_abs, std::abs(v));
  }
  const float abs_tol = rel_tol * std::max(max_abs, 1.0f);
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], abs_tol) << "index " << i;
  }
}

float ReferenceTolerance(const RunConfig& cfg) {
  if (cfg.fp16) {
    return 1e-2f;
  }
  return cfg.accuracy_level == 4 ? 2e-2f : 1e-4f;
}

// Runs the authored large-block model and its block-256 twin and checks both against the reference.
void CheckAgainstTwin(const Weights& w, const RunConfig& cfg, uint32_t seed) {
  SCOPED_TRACE(::testing::Message() << "N=" << w.n << " K=" << w.k << " bits=" << w.bits
                                    << " block=" << w.block_size << " zp=" << static_cast<int>(w.zp_mode)
                                    << " acc=" << cfg.accuracy_level << " fp16=" << cfg.fp16
                                    << " no_prepack=" << cfg.disable_prepacking
                                    << " runtime_qparams=" << cfg.quant_params_as_inputs
                                    << " external=" << cfg.external_data << " bias=" << cfg.has_bias
                                    << " batch=" << cfg.batch);
  const Tensors io = MakeIo(w, cfg, seed);
  const RunResult authored = RunModel(w, Serialize(w, w.block_size), cfg, io);
  ASSERT_STATUS_OK(authored.status);
  const RunResult twin = RunModel(w, Serialize(w, 256), cfg, io);
  ASSERT_STATUS_OK(twin.status);
  // The adapted node takes the same (pre)packed execution path as the twin.
  EXPECT_EQ(authored.prepacked, twin.prepacked);
  if (cfg.disable_prepacking) {
    EXPECT_EQ(authored.prepacked, 0u);
  }
  ExpectBitwiseEqual(authored.y, twin.y);
  ExpectClose(authored.y, Reference(w, cfg, io), ReferenceTolerance(cfg));
}

}  // namespace

TEST(MatMulNBitsLargeBlock, SerializedTwinsDecodeIdentically) {
  // The independently serialized block-256 layout must represent exactly the same weights.
  for (int64_t block : {512, 1024, 2048}) {
    for (int64_t k : {block, block / 2 + 37, 3 * block, 2 * block + 300}) {
      const Weights w = MakeWeights(3, k, 4, block, ZeroPoint::kRandom, 11);
      const NBitsTensors t = Serialize(w, 256);
      const int64_t blob = 256 * w.bits / 8;
      const int64_t zp_row = (t.k_blocks * w.bits + 7) / 8;
      for (int64_t c = 0; c < w.n; ++c) {
        for (int64_t kk = 0; kk < k; ++kk) {
          const int64_t h = kk / 256;
          const uint8_t q = (t.b[c * t.k_blocks * blob + kk / 2] >> ((kk % 2) * 4)) & 0xF;
          const uint8_t zp = (t.zp_packed[c * zp_row + h / 2] >> ((h % 2) * 4)) & 0xF;
          const float value = (static_cast<float>(q) - static_cast<float>(zp)) * t.scales[c * t.k_blocks + h];
          ASSERT_EQ(value, Decode(w, c, kk)) << "block=" << block << " k=" << k << " c=" << c << " kk=" << kk;
        }
      }
    }
  }
}

TEST(MatMulNBitsLargeBlock, Int4_MatchesBlock256Twin) {
  uint32_t seed = 1;
  for (int64_t block : {512, 1024, 2048}) {
    // K equal to, smaller than and a multiple of the block, plus partial tails not divisible by 256.
    for (int64_t k : {block, block / 2 + 37, 3 * block, 2 * block + 300}) {
      for (ZeroPoint zp : {ZeroPoint::kAbsent, ZeroPoint::kMidpoint, ZeroPoint::kRandom}) {
        for (int64_t accuracy_level : {1, 4}) {
          RunConfig cfg;
          cfg.accuracy_level = accuracy_level;
          CheckAgainstTwin(MakeWeights(13, k, 4, block, zp, seed++), cfg, seed);
        }
      }
    }
  }
}

TEST(MatMulNBitsLargeBlock, Int8_MatchesBlock256Twin) {
  uint32_t seed = 100;
  for (int64_t block : {512, 1024, 2048}) {
    for (int64_t k : {block, block / 2 + 37, 2 * block + 300}) {
      for (ZeroPoint zp : {ZeroPoint::kAbsent, ZeroPoint::kRandom}) {
        for (int64_t accuracy_level : {1, 4}) {
          RunConfig cfg;
          cfg.accuracy_level = accuracy_level;
          CheckAgainstTwin(MakeWeights(7, k, 8, block, zp, seed++), cfg, seed);
        }
      }
    }
  }
}

TEST(MatMulNBitsLargeBlock, OddLogicalGroupCounts) {
  // 1, 3 and 5 logical groups: exposes zero-point nibble padding errors in both the logical and repeated layouts.
  uint32_t seed = 200;
  for (int64_t groups : {1, 3, 5}) {
    for (int64_t tail : {0, 100}) {
      const int64_t k = (groups - 1) * 1024 + (tail == 0 ? 1024 : tail);
      RunConfig cfg;
      CheckAgainstTwin(MakeWeights(9, k, 4, 1024, ZeroPoint::kRandom, seed++), cfg, seed);
    }
  }
}

TEST(MatMulNBitsLargeBlock, FloatZeroPoints) {
  uint32_t seed = 300;
  for (int64_t k : {1024, 1500}) {
    RunConfig cfg;
    CheckAgainstTwin(MakeWeights(6, k, 4, 1024, ZeroPoint::kFloat, seed++), cfg, seed);
  }
}

TEST(MatMulNBitsLargeBlock, BiasAndBatchedA) {
  RunConfig cfg;
  cfg.has_bias = true;
  cfg.batch = 3;
  cfg.m = 5;
  for (int64_t accuracy_level : {1, 4}) {
    cfg.accuracy_level = accuracy_level;
    CheckAgainstTwin(MakeWeights(16, 2048 + 300, 4, 1024, ZeroPoint::kRandom, 400), cfg, 401);
    CheckAgainstTwin(MakeWeights(16, 2048 + 300, 8, 1024, ZeroPoint::kRandom, 402), cfg, 403);
  }
}

TEST(MatMulNBitsLargeBlock, Fp16Input) {
  RunConfig cfg;
  cfg.fp16 = true;
  for (int64_t accuracy_level : {1, 4}) {
    cfg.accuracy_level = accuracy_level;
    CheckAgainstTwin(MakeWeights(8, 1024, 4, 1024, ZeroPoint::kAbsent, 500), cfg, 501);
    CheckAgainstTwin(MakeWeights(8, 1300, 4, 1024, ZeroPoint::kRandom, 502), cfg, 503);
    CheckAgainstTwin(MakeWeights(8, 1300, 8, 1024, ZeroPoint::kRandom, 504), cfg, 505);
  }
}

TEST(MatMulNBitsLargeBlock, PrepackingDisabled) {
  RunConfig cfg;
  cfg.disable_prepacking = true;
  CheckAgainstTwin(MakeWeights(10, 1024, 4, 1024, ZeroPoint::kRandom, 600), cfg, 601);
  CheckAgainstTwin(MakeWeights(10, 1500, 4, 1024, ZeroPoint::kRandom, 602), cfg, 603);
  CheckAgainstTwin(MakeWeights(10, 1500, 8, 2048, ZeroPoint::kAbsent, 604), cfg, 605);
}

TEST(MatMulNBitsLargeBlock, RuntimeScalesAndZeroPoints) {
  RunConfig cfg;
  cfg.quant_params_as_inputs = true;
  // Accuracy level 1 only: runtime scales with x64 CompInt8 already produce zeros for small blocks (v1.27.0 too).
  cfg.accuracy_level = 1;
  CheckAgainstTwin(MakeWeights(10, 1500, 4, 1024, ZeroPoint::kRandom, 700), cfg, 701);
  CheckAgainstTwin(MakeWeights(10, 1024, 4, 1024, ZeroPoint::kAbsent, 702), cfg, 703);
  CheckAgainstTwin(MakeWeights(10, 1500, 8, 512, ZeroPoint::kRandom, 704), cfg, 705);
}

TEST(MatMulNBitsLargeBlock, ExternalDataInitializers) {
  RunConfig cfg;
  cfg.external_data = true;
  cfg.has_bias = true;
  for (int64_t accuracy_level : {1, 4}) {
    cfg.accuracy_level = accuracy_level;
    CheckAgainstTwin(MakeWeights(12, 1024, 4, 1024, ZeroPoint::kRandom, 800), cfg, 801);
    CheckAgainstTwin(MakeWeights(12, 1300, 4, 1024, ZeroPoint::kAbsent, 802), cfg, 803);
  }

  // External and embedded initializers give identical results.
  const Weights w = MakeWeights(12, 1300, 4, 1024, ZeroPoint::kRandom, 804);
  const Tensors io = MakeIo(w, cfg, 805);
  const RunResult external = RunModel(w, Serialize(w, 1024), cfg, io);
  RunConfig embedded_cfg = cfg;
  embedded_cfg.external_data = false;
  const RunResult embedded = RunModel(w, Serialize(w, 1024), embedded_cfg, io);
  ASSERT_STATUS_OK(external.status);
  ASSERT_STATUS_OK(embedded.status);
  ExpectBitwiseEqual(external.y, embedded.y);
}

TEST(MatMulNBitsLargeBlock, WeightReconstructionWithIdentityInput) {
  // A = I makes Y[m][n] the dequantized weight W[k=m][n].
  for (int64_t bits : {4, 8}) {
    const Weights w = MakeWeights(5, 1300, bits, 1024, ZeroPoint::kRandom, 900 + static_cast<uint32_t>(bits));
    RunConfig cfg;
    cfg.m = w.k;
    Tensors io;
    io.a.assign(static_cast<size_t>(w.k * w.k), 0.0f);
    for (int64_t i = 0; i < w.k; ++i) {
      io.a[i * w.k + i] = 1.0f;
    }
    const RunResult authored = RunModel(w, Serialize(w, 1024), cfg, io);
    const RunResult twin = RunModel(w, Serialize(w, 256), cfg, io);
    ASSERT_STATUS_OK(authored.status);
    ASSERT_STATUS_OK(twin.status);
    ExpectBitwiseEqual(authored.y, twin.y);
    for (int64_t kk = 0; kk < w.k; ++kk) {
      for (int64_t c = 0; c < w.n; ++c) {
        // Allow the rounding of a q * scale - zp * scale dequantization formula.
        const float scale = w.scales[c * w.k_blocks + kk / w.block_size];
        ASSERT_NEAR(authored.y[kk * w.n + c], Decode(w, c, kk), scale * static_cast<float>(1 << bits) * 1e-6f)
            << "bits=" << bits << " k=" << kk << " n=" << c;
      }
    }
  }
}

TEST(MatMulNBitsLargeBlock, SharedPrepackedWeightsRespectScalesAndZeroPoints) {
  // Two nodes share the identical quantized B through a pre-packed weights container but use different scales and
  // zero points. The second session must not reuse packed data that embeds the first session's metadata.
  const Weights w1 = MakeWeights(16, 1500, 4, 1024, ZeroPoint::kRandom, 1000);
  Weights w2 = w1;
  for (size_t i = 0; i < w2.scales.size(); ++i) {
    w2.scales[i] *= 1.5f;
    w2.zp_int[i] = static_cast<uint8_t>((w2.zp_int[i] + 5) % 16);
  }
  for (int64_t accuracy_level : {1, 4}) {
    RunConfig cfg;
    cfg.accuracy_level = accuracy_level;
    const Tensors io = MakeIo(w1, cfg, 1001);
    const NBitsTensors t1 = Serialize(w1, 1024);
    NBitsTensors t2 = Serialize(w2, 1024);
    t2.b = t1.b;  // identical B initializer bytes

    std::vector<uint8_t> b_bytes = t1.b;
    OrtValue shared_b;
    Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(),
                         TensorShape({w1.n, t1.k_blocks, 1024 * 4 / 8}), b_bytes.data(),
                         OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), shared_b);

    PrepackedWeightsContainer container;
    const RunResult r1 = RunModel(w1, t1, cfg, io, &container, &shared_b);
    ASSERT_STATUS_OK(r1.status);
    const RunResult r1_again = RunModel(w1, t1, cfg, io, &container, &shared_b);
    ASSERT_STATUS_OK(r1_again.status);
    ExpectBitwiseEqual(r1_again.y, r1.y);
    if (r1.prepacked > 0) {
      EXPECT_GT(r1_again.used_shared_prepacked, 0u);
    }

    const RunResult r2 = RunModel(w2, t2, cfg, io, &container, &shared_b);
    ASSERT_STATUS_OK(r2.status);
    // Reference for w2 must use the shared (w1) quantized values.
    Weights w2_effective = w2;
    w2_effective.q = w1.q;
    const RunResult r2_unshared = RunModel(w2_effective, Serialize(w2_effective, 256), cfg, io);
    ASSERT_STATUS_OK(r2_unshared.status);
    ExpectBitwiseEqual(r2.y, r2_unshared.y);
    ExpectClose(r2.y, Reference(w2_effective, cfg, io), ReferenceTolerance(cfg));
  }
}

TEST(MatMulNBitsLargeBlock, Invalid_NonPowerOfTwoBlock) {
  const Weights w = MakeWeights(4, 768, 4, 256, ZeroPoint::kAbsent, 1100);
  const RunResult r = RunModel(w, Serialize(w, 256), RunConfig{}, MakeIo(w, RunConfig{}, 1101), nullptr, nullptr,
                               /*block_size_attr*/ 384);
  ASSERT_FALSE(r.status.IsOK());
  EXPECT_NE(r.status.ErrorMessage().find("Only power-of-two block sizes"), std::string::npos) << r.status;
}

TEST(MatMulNBitsLargeBlock, Invalid_TwoBitLargeBlock) {
  Weights w = MakeWeights(4, 1024, 2, 1024, ZeroPoint::kAbsent, 1200);
  const RunResult r = RunModel(w, Serialize(w, 1024), RunConfig{}, MakeIo(w, RunConfig{}, 1201));
  ASSERT_FALSE(r.status.IsOK());
  EXPECT_NE(r.status.ErrorMessage().find("2-bit MatMulNBits block_size must not exceed 256"), std::string::npos)
      << r.status;
}

TEST(MatMulNBitsLargeBlock, Invalid_GroupIndexWithLargeBlock) {
  const Weights w = MakeWeights(4, 1024, 4, 1024, ZeroPoint::kRandom, 1300);
  const RunResult r = RunModel(w, Serialize(w, 1024), RunConfig{}, MakeIo(w, RunConfig{}, 1301), nullptr, nullptr,
                               -1, /*add_g_idx*/ true);
  ASSERT_FALSE(r.status.IsOK());
  EXPECT_NE(r.status.ErrorMessage().find("g_idx does not support block_size > 256"), std::string::npos) << r.status;
}

TEST(MatMulNBitsLargeBlock, Invalid_UndersizedTensors) {
  // Tensors laid out for block 256 while the attribute claims 1024. With K = 1300 the logical layout is
  // 2 x 512 bytes per row and the 256 layout is 6 x 128 bytes, so B, scales and zero points are all mismatched.
  // Both the PrePack validation and the Compute-time validation must reject them.
  const Weights w = MakeWeights(4, 1300, 4, 1024, ZeroPoint::kRandom, 1400);
  for (bool disable_prepacking : {false, true}) {
    RunConfig cfg;
    cfg.disable_prepacking = disable_prepacking;
    const RunResult r = RunModel(w, Serialize(w, 256), cfg, MakeIo(w, cfg, 1401), nullptr, nullptr,
                                 /*block_size_attr*/ 1024);
    ASSERT_FALSE(r.status.IsOK()) << "disable_prepacking=" << disable_prepacking;
    if (!disable_prepacking) {
      EXPECT_NE(r.status.ErrorMessage().find("MatMulNBits PrePack: B initializer shape"), std::string::npos)
          << r.status;
    }
  }
}

TEST(MatMulNBitsLargeBlock, Invalid_OverflowingBlockSize) {
  // A huge power-of-two block must be rejected cleanly, never read out of bounds.
  const Weights w = MakeWeights(2, 256, 8, 256, ZeroPoint::kAbsent, 1500);
  for (int64_t block : {int64_t{1} << 40, int64_t{1} << 61, int64_t{1} << 62}) {
    const RunResult r = RunModel(w, Serialize(w, 256), RunConfig{}, MakeIo(w, RunConfig{}, 1501), nullptr, nullptr,
                                 block);
    EXPECT_FALSE(r.status.IsOK()) << "block=" << block;
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
