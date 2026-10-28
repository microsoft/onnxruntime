// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/common/make_string.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

// GRU and LSTM on WebGPU in float32 and float16. The CPU kernels are float32-only, so expected values
// come from a host reference. The float32 runs use a tight tolerance and so also validate the reference.

namespace onnxruntime {
namespace test {
namespace {

struct RnnShape {
  int64_t seq;
  int64_t batch;
  int64_t input;
  int64_t hidden;
};

struct GruCase {
  RnnShape s;
  std::string direction;
  std::vector<std::string> activations;
  bool linear_before_reset;
  std::vector<float> X, W, R, B, h0;
};

// An empty P or seq_lens omits that input.
struct LstmCase {
  RnnShape s;
  std::string direction;
  std::vector<std::string> activations;
  std::vector<float> X, W, R, B, h0, c0, P;
  std::vector<int32_t> seq_lens;
};

// Deterministic values in [-scale, scale], rounded to float16 so both precisions see the same data.
std::vector<float> MakeData(size_t count, int seed, float scale) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    const int v = static_cast<int>((i * 37 + static_cast<size_t>(seed) * 11) % 23) - 11;
    values[i] = static_cast<float>(MLFloat16(scale * static_cast<float>(v) / 11.0f));
  }
  return values;
}

double Activate(const std::string& name, double v) {
  if (name == "Sigmoid") return 1.0 / (1.0 + std::exp(-v));
  if (name == "Tanh") return std::tanh(v);
  return std::max(0.0, v);  // Relu
}

std::vector<bool> ReverseFlags(const std::string& direction) {
  if (direction == "bidirectional") return {false, true};
  return {direction == "reverse"};
}

int64_t NumDirections(const std::string& direction) {
  return direction == "bidirectional" ? 2 : 1;
}

// ONNX GRU, layout 0. W/R gate order is z, r, h; B is [Wb(z, r, h), Rb(z, r, h)].
void ReferenceGru(const GruCase& cfg, std::vector<float>& Y, std::vector<float>& Y_h) {
  const RnnShape& s = cfg.s;
  const auto& X = cfg.X;
  const auto& W = cfg.W;
  const auto& R = cfg.R;
  const auto& B = cfg.B;
  const bool linear_before_reset = cfg.linear_before_reset;
  const auto reverse = ReverseFlags(cfg.direction);
  const int64_t dirs = static_cast<int64_t>(reverse.size());
  const int64_t H = s.hidden;
  Y.assign(static_cast<size_t>(s.seq * dirs * s.batch * H), 0.0f);
  Y_h.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  for (int64_t d = 0; d < dirs; ++d) {
    const std::string& f = cfg.activations[d * 2];
    const std::string& g = cfg.activations[d * 2 + 1];
    std::vector<double> h(cfg.h0.begin() + d * s.batch * H, cfg.h0.begin() + (d + 1) * s.batch * H);
    for (int64_t step = 0; step < s.seq; ++step) {
      const int64_t t = reverse[d] ? s.seq - 1 - step : step;
      const auto gate = [&](int64_t b, int64_t row) {
        double acc = B[d * 6 * H + row] + B[d * 6 * H + 3 * H + row];
        for (int64_t i = 0; i < s.input; ++i) {
          acc += X[(t * s.batch + b) * s.input + i] * W[(d * 3 * H + row) * s.input + i];
        }
        for (int64_t k = 0; k < H; ++k) acc += h[b * H + k] * R[(d * 3 * H + row) * H + k];
        return acc;
      };
      std::vector<double> next(h.size());
      for (int64_t b = 0; b < s.batch; ++b) {
        std::vector<double> r(H);
        for (int64_t k = 0; k < H; ++k) r[k] = Activate(f, gate(b, H + k));
        for (int64_t j = 0; j < H; ++j) {
          const double z = Activate(f, gate(b, j));
          double rec = B[d * 6 * H + 5 * H + j];
          for (int64_t k = 0; k < H; ++k) {
            rec += (linear_before_reset ? 1.0 : r[k]) * h[b * H + k] * R[(d * 3 * H + 2 * H + j) * H + k];
          }
          double xh = B[d * 6 * H + 2 * H + j];
          for (int64_t i = 0; i < s.input; ++i) {
            xh += X[(t * s.batch + b) * s.input + i] * W[(d * 3 * H + 2 * H + j) * s.input + i];
          }
          const double hh = Activate(g, xh + (linear_before_reset ? r[j] * rec : rec));
          next[b * H + j] = (1.0 - z) * hh + z * h[b * H + j];
          Y[((t * dirs + d) * s.batch + b) * H + j] = static_cast<float>(next[b * H + j]);
        }
      }
      h = next;
    }
    for (size_t k = 0; k < h.size(); ++k) Y_h[d * s.batch * H + k] = static_cast<float>(h[k]);
  }
}

// ONNX LSTM, layout 0. W/R gate order is i, o, f, c; B is [Wb, Rb]; P is [Pi, Po, Pf]. Past a batch's
// sequence length the state carries over and Y stays zero.
void ReferenceLstm(const LstmCase& cfg, std::vector<float>& Y, std::vector<float>& Y_h, std::vector<float>& Y_c) {
  const RnnShape& s = cfg.s;
  const auto& X = cfg.X;
  const auto& W = cfg.W;
  const auto& R = cfg.R;
  const auto& B = cfg.B;
  const auto reverse = ReverseFlags(cfg.direction);
  const int64_t dirs = static_cast<int64_t>(reverse.size());
  const int64_t H = s.hidden;
  const std::vector<float> P = cfg.P.empty() ? std::vector<float>(static_cast<size_t>(dirs * 3 * H), 0.0f) : cfg.P;
  Y.assign(static_cast<size_t>(s.seq * dirs * s.batch * H), 0.0f);
  Y_h.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  Y_c.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  for (int64_t d = 0; d < dirs; ++d) {
    const std::string& f = cfg.activations[d * 3];
    const std::string& g = cfg.activations[d * 3 + 1];
    const std::string& hact = cfg.activations[d * 3 + 2];
    std::vector<double> h(cfg.h0.begin() + d * s.batch * H, cfg.h0.begin() + (d + 1) * s.batch * H);
    std::vector<double> c(cfg.c0.begin() + d * s.batch * H, cfg.c0.begin() + (d + 1) * s.batch * H);
    for (int64_t step = 0; step < s.seq; ++step) {
      const int64_t t = reverse[d] ? s.seq - 1 - step : step;
      const auto gate = [&](int64_t b, int64_t row) {
        double acc = B[d * 8 * H + row] + B[d * 8 * H + 4 * H + row];
        for (int64_t i = 0; i < s.input; ++i) {
          acc += X[(t * s.batch + b) * s.input + i] * W[(d * 4 * H + row) * s.input + i];
        }
        for (int64_t k = 0; k < H; ++k) acc += h[b * H + k] * R[(d * 4 * H + row) * H + k];
        return acc;
      };
      std::vector<double> next_h = h;
      std::vector<double> next_c = c;
      for (int64_t b = 0; b < s.batch; ++b) {
        if (!cfg.seq_lens.empty() && t >= cfg.seq_lens[b]) continue;
        for (int64_t j = 0; j < H; ++j) {
          const double c_prev = c[b * H + j];
          const double i_gate = Activate(f, gate(b, j) + P[d * 3 * H + j] * c_prev);
          const double f_gate = Activate(f, gate(b, 2 * H + j) + P[d * 3 * H + 2 * H + j] * c_prev);
          const double c_hat = Activate(g, gate(b, 3 * H + j));
          const double c_new = f_gate * c_prev + i_gate * c_hat;
          const double o_gate = Activate(f, gate(b, H + j) + P[d * 3 * H + H + j] * c_new);
          next_c[b * H + j] = c_new;
          next_h[b * H + j] = o_gate * Activate(hact, c_new);
          Y[((t * dirs + d) * s.batch + b) * H + j] = static_cast<float>(next_h[b * H + j]);
        }
      }
      h = next_h;
      c = next_c;
    }
    for (size_t k = 0; k < h.size(); ++k) {
      Y_h[d * s.batch * H + k] = static_cast<float>(h[k]);
      Y_c[d * s.batch * H + k] = static_cast<float>(c[k]);
    }
  }
}

template <typename T>
std::vector<T> Convert(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (float v : values) result.push_back(T(v));
  return result;
}

SessionOptions WebGpuOnly() {
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return options;
}

template <typename T>
float Tolerance() {
  return std::is_same_v<T, MLFloat16> ? 4e-3f : 1e-4f;
}

// The state-precision cases keep every output in [64, 128), where two float16 ULPs are 0.125.
template <typename T>
float StatePrecisionAbsTolerance() {
  return std::is_same_v<T, MLFloat16> ? 0.125f : Tolerance<T>();
}

template <typename T>
float StatePrecisionRelTolerance() {
  return std::is_same_v<T, MLFloat16> ? 0.0f : Tolerance<T>();
}

GruCase MakeGruCase(const RnnShape& s, const std::string& direction, const std::vector<std::string>& activations,
                    bool linear_before_reset) {
  const int64_t dirs = NumDirections(direction);
  return GruCase{s,
                 direction,
                 activations,
                 linear_before_reset,
                 MakeData(s.seq * s.batch * s.input, 1, 1.0f),
                 MakeData(dirs * 3 * s.hidden * s.input, 2, 0.5f),
                 MakeData(dirs * 3 * s.hidden * s.hidden, 3, 0.5f),
                 MakeData(dirs * 6 * s.hidden, 4, 0.25f),
                 MakeData(dirs * s.batch * s.hidden, 5, 0.5f)};
}

LstmCase MakeLstmCase(const RnnShape& s, const std::string& direction, const std::vector<std::string>& activations) {
  const int64_t dirs = NumDirections(direction);
  return LstmCase{s,
                  direction,
                  activations,
                  MakeData(s.seq * s.batch * s.input, 1, 1.0f),
                  MakeData(dirs * 4 * s.hidden * s.input, 2, 0.5f),
                  MakeData(dirs * 4 * s.hidden * s.hidden, 3, 0.5f),
                  MakeData(dirs * 8 * s.hidden, 4, 0.25f),
                  MakeData(dirs * s.batch * s.hidden, 5, 0.5f),
                  MakeData(dirs * s.batch * s.hidden, 6, 0.5f),
                  MakeData(dirs * 3 * s.hidden, 7, 0.5f),
                  {}};
}

template <typename T>
void RunGru(const GruCase& cfg, float abs_tol, float rel_tol) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  const RnnShape& s = cfg.s;
  const int64_t dirs = NumDirections(cfg.direction);
  std::vector<float> Y, Y_h;
  ReferenceGru(cfg, Y, Y_h);

  OpTester test("GRU", 14);
  test.AddAttribute("activations", cfg.activations);
  test.AddAttribute("direction", cfg.direction);
  test.AddAttribute<int64_t>("hidden_size", s.hidden);
  test.AddAttribute<int64_t>("linear_before_reset", cfg.linear_before_reset ? 1 : 0);
  test.AddInput<T>("X", {s.seq, s.batch, s.input}, Convert<T>(cfg.X));
  test.AddInput<T>("W", {dirs, 3 * s.hidden, s.input}, Convert<T>(cfg.W), true);
  test.AddInput<T>("R", {dirs, 3 * s.hidden, s.hidden}, Convert<T>(cfg.R), true);
  test.AddInput<T>("B", {dirs, 6 * s.hidden}, Convert<T>(cfg.B), true);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<T>("initial_h", {dirs, s.batch, s.hidden}, Convert<T>(cfg.h0));
  test.AddOutput<T>("Y", {s.seq, dirs, s.batch, s.hidden}, Convert<T>(Y));
  test.AddOutput<T>("Y_h", {dirs, s.batch, s.hidden}, Convert<T>(Y_h));
  test.SetOutputTolerance(abs_tol, rel_tol);
  test.Config(WebGpuOnly()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

void AddLstmAttributes(OpTester& test, const LstmCase& cfg) {
  test.AddAttribute("activations", cfg.activations);
  test.AddAttribute("direction", cfg.direction);
  test.AddAttribute<int64_t>("hidden_size", cfg.s.hidden);
}

// Adds one LSTM node's inputs and expected outputs, with `suffix` appended to every name.
template <typename T>
void AddLstmInputsAndOutputs(OpTester& test, const LstmCase& cfg, const std::string& suffix, float abs_tol,
                             float rel_tol) {
  const RnnShape& s = cfg.s;
  const int64_t dirs = NumDirections(cfg.direction);
  std::vector<float> Y, Y_h, Y_c;
  ReferenceLstm(cfg, Y, Y_h, Y_c);

  const auto name = [&suffix](const char* base) { return base + suffix; };
  test.AddInput<T>(name("X").c_str(), {s.seq, s.batch, s.input}, Convert<T>(cfg.X));
  test.AddInput<T>(name("W").c_str(), {dirs, 4 * s.hidden, s.input}, Convert<T>(cfg.W), true);
  test.AddInput<T>(name("R").c_str(), {dirs, 4 * s.hidden, s.hidden}, Convert<T>(cfg.R), true);
  test.AddInput<T>(name("B").c_str(), {dirs, 8 * s.hidden}, Convert<T>(cfg.B), true);
  if (cfg.seq_lens.empty()) {
    test.AddOptionalInputEdge<int32_t>();
  } else {
    test.AddInput<int32_t>(name("sequence_lens").c_str(), {s.batch}, cfg.seq_lens);
  }
  test.AddInput<T>(name("initial_h").c_str(), {dirs, s.batch, s.hidden}, Convert<T>(cfg.h0));
  test.AddInput<T>(name("initial_c").c_str(), {dirs, s.batch, s.hidden}, Convert<T>(cfg.c0));
  if (!cfg.P.empty()) {
    test.AddInput<T>(name("P").c_str(), {dirs, 3 * s.hidden}, Convert<T>(cfg.P), true);
  }
  test.AddOutput<T>(name("Y").c_str(), {s.seq, dirs, s.batch, s.hidden}, Convert<T>(Y));
  test.AddOutput<T>(name("Y_h").c_str(), {dirs, s.batch, s.hidden}, Convert<T>(Y_h));
  test.AddOutput<T>(name("Y_c").c_str(), {dirs, s.batch, s.hidden}, Convert<T>(Y_c));
  for (const char* output : {"Y", "Y_h", "Y_c"}) {
    test.SetOutputAbsErr(name(output).c_str(), abs_tol);
    test.SetOutputRelErr(name(output).c_str(), rel_tol);
  }
}

template <typename T>
void RunLstm(const LstmCase& cfg, float abs_tol, float rel_tol) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  OpTester test("LSTM", 14);
  AddLstmAttributes(test, cfg);
  AddLstmInputsAndOutputs<T>(test, cfg, "", abs_tol, rel_tol);
  test.Config(WebGpuOnly()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <typename T>
void TestGruVariants() {
  const RnnShape s{3, 2, 3, 4};
  for (bool lbr : {false, true}) {
    SCOPED_TRACE(lbr);
    RunGru<T>(MakeGruCase(s, "forward", {"Sigmoid", "Tanh"}, lbr), Tolerance<T>(), Tolerance<T>());
    RunGru<T>(MakeGruCase(s, "reverse", {"Sigmoid", "Relu"}, lbr), Tolerance<T>(), Tolerance<T>());
    RunGru<T>(MakeGruCase(s, "bidirectional", {"Sigmoid", "Tanh", "Relu", "Relu"}, lbr), Tolerance<T>(),
              Tolerance<T>());
  }
}

template <typename T>
void TestLstmVariants() {
  const RnnShape s{3, 2, 3, 4};
  RunLstm<T>(MakeLstmCase(s, "forward", {"Sigmoid", "Tanh", "Tanh"}), Tolerance<T>(), Tolerance<T>());
  RunLstm<T>(MakeLstmCase(s, "reverse", {"Sigmoid", "Relu", "Relu"}), Tolerance<T>(), Tolerance<T>());
  RunLstm<T>(MakeLstmCase(s, "bidirectional", {"Sigmoid", "Tanh", "Tanh", "Relu", "Relu", "Relu"}), Tolerance<T>(),
             Tolerance<T>());
}

// X for input_size 1, in [0.5, 1.0] and exact in float16.
std::vector<float> RampInput(const RnnShape& s) {
  std::vector<float> x;
  x.reserve(static_cast<size_t>(s.seq * s.batch));
  for (int64_t t = 0; t < s.seq; ++t) {
    for (int64_t b = 0; b < s.batch; ++b) {
      x.push_back(static_cast<float>(MLFloat16(0.5f + 0.05f * static_cast<float>((t * 7 + b * 3) % 11))));
    }
  }
  return x;
}

// [batch, hidden] state 64, 64 + step, ..., inside float16's [64, 128) binade where the ULP is 0.0625.
std::vector<float> StateRamp(const RnnShape& s, float step) {
  std::vector<float> state(static_cast<size_t>(s.batch * s.hidden));
  for (size_t i = 0; i < state.size(); ++i) state[i] = 64.0f + step * static_cast<float>(i);
  return state;
}

// z = sigmoid(8.5) or sigmoid(9) (1 - z = 2.0e-4 or 1.2e-4) rounds to exactly 1 in float16, and each step
// moves h by under 0.02. With float32 state and z scratch h drifts toward the Relu candidate by 1.2-2.4
// over 200 steps; float16 state or z scratch freezes h at initial_h.
template <typename T>
void TestGruStatePrecision() {
  const RnnShape s{200, 2, 1, 2};
  const GruCase cfg{s,
                    "forward",
                    {"Sigmoid", "Relu"},
                    false,
                    RampInput(s),
                    {0.0f, 0.0f, 0.0f, 0.0f, 32.0f, 48.0f},  // W: z, r, h
                    std::vector<float>(static_cast<size_t>(3 * s.hidden * s.hidden), 0.0f),
                    {4.25f, 4.5f, 0.0f, 0.0f, 100.0f, 100.0f,  // Wb: z, r, h
                     4.25f, 4.5f, 0.0f, 0.0f, 0.0f, 0.0f},     // Rb: z, r, h
                    StateRamp(s, 8.0f)};
  RunGru<T>(cfg, StatePrecisionAbsTolerance<T>(), StatePrecisionRelTolerance<T>());
}

// The i, o and f gates saturate at 1 and R is zero, so C (and H = o * Relu(C)) grows by tanh(Wc * x), in
// [0.01, 0.025], each step. With float32 state C grows by 1.9-2.4 over 128 steps; with float16 state every
// increment is under half a ULP and C stays at initial_c.
template <typename T>
void TestLstmStatePrecision() {
  const RnnShape s{128, 2, 1, 2};
  const float wc0 = static_cast<float>(MLFloat16(0.02f));
  const float wc1 = static_cast<float>(MLFloat16(0.025f));
  const LstmCase cfg{s,
                     "forward",
                     {"Sigmoid", "Tanh", "Relu"},
                     RampInput(s),
                     {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, wc0, wc1},  // W: i, o, f, c
                     std::vector<float>(static_cast<size_t>(4 * s.hidden * s.hidden), 0.0f),
                     {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 0.0f, 0.0f,   // Wb: i, o, f, c
                      10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 0.0f, 0.0f},  // Rb: i, o, f, c
                     std::vector<float>(static_cast<size_t>(s.batch * s.hidden), 0.0f),
                     StateRamp(s, 16.0f),
                     {},
                     {}};
  RunLstm<T>(cfg, StatePrecisionAbsTolerance<T>(), StatePrecisionRelTolerance<T>());
}

// Adds one LSTM node per variant so all variants run in one session and share its WebGPU program cache.
class LstmVariantsTester final : public OpTester {
 public:
  explicit LstmVariantsTester(size_t variants) : OpTester("LSTM", 14), variants_{variants} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LstmVariantsTester);

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>& add_attribute_funcs) override {
    constexpr size_t kInputs = 8;
    constexpr size_t kOutputs = 3;
    ORT_ENFORCE(inputs.size() == variants_ * kInputs && outputs.size() == variants_ * kOutputs);
    for (size_t v = 0; v < variants_; ++v) {
      std::vector<NodeArg*> node_inputs(inputs.begin() + v * kInputs, inputs.begin() + (v + 1) * kInputs);
      std::vector<NodeArg*> node_outputs(outputs.begin() + v * kOutputs, outputs.begin() + (v + 1) * kOutputs);
      auto& node = graph.AddNode(MakeString("lstm", v), "LSTM", "", node_inputs, node_outputs);
      for (auto& add_attribute : add_attribute_funcs) {
        add_attribute(node);
      }
    }
  }

 private:
  const size_t variants_;
};

}  // namespace

TEST(GRUTest, WebGpu_Float_Reference) {
  TestGruVariants<float>();
}

TEST(GRUTest, WebGpu_Float16) {
  TestGruVariants<MLFloat16>();
}

TEST(GRUTest, WebGpu_Float_StatePrecision) {
  TestGruStatePrecision<float>();
}

TEST(GRUTest, WebGpu_Float16_StatePrecision) {
  TestGruStatePrecision<MLFloat16>();
}

TEST(LSTMTest, WebGpu_Float_Reference) {
  TestLstmVariants<float>();
}

TEST(LSTMTest, WebGpu_Float16) {
  TestLstmVariants<MLFloat16>();
}

TEST(LSTMTest, WebGpu_Float_StatePrecision) {
  TestLstmStatePrecision<float>();
}

TEST(LSTMTest, WebGpu_Float16_StatePrecision) {
  TestLstmStatePrecision<MLFloat16>();
}

// With B, P and sequence_lens the cell program has no storage binding left for Y, so LstmWriteYProgram copies
// each step's float32 state into Y. The float32, float16 and float32 nodes share one session's program cache,
// so that program must key on Y's element type.
TEST(LSTMTest, WebGpu_SplitYOutputTypes) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  LstmCase cfg = MakeLstmCase({4, 3, 3, 4}, "bidirectional", {"Sigmoid", "Tanh", "Tanh", "Sigmoid", "Relu", "Relu"});
  cfg.seq_lens = {4, 1, 3};

  LstmVariantsTester test(3);
  AddLstmAttributes(test, cfg);
  AddLstmInputsAndOutputs<float>(test, cfg, "0", Tolerance<float>(), Tolerance<float>());
  AddLstmInputsAndOutputs<MLFloat16>(test, cfg, "1", Tolerance<MLFloat16>(), Tolerance<MLFloat16>());
  AddLstmInputsAndOutputs<float>(test, cfg, "2", Tolerance<float>(), Tolerance<float>());
  test.Config(WebGpuOnly()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
