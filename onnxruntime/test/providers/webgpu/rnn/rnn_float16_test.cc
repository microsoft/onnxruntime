// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
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

// ONNX GRU, layout 0. W/R gate order is z, r, h; B is [Wb(z, r, h), Rb(z, r, h)].
void ReferenceGru(const RnnShape& s, const std::string& direction, const std::vector<std::string>& activations,
                  bool linear_before_reset, const std::vector<float>& X, const std::vector<float>& W,
                  const std::vector<float>& R, const std::vector<float>& B, const std::vector<float>& h0,
                  std::vector<float>& Y, std::vector<float>& Y_h) {
  const auto reverse = ReverseFlags(direction);
  const int64_t dirs = static_cast<int64_t>(reverse.size());
  const int64_t H = s.hidden;
  Y.assign(static_cast<size_t>(s.seq * dirs * s.batch * H), 0.0f);
  Y_h.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  for (int64_t d = 0; d < dirs; ++d) {
    const std::string& f = activations[d * 2];
    const std::string& g = activations[d * 2 + 1];
    std::vector<double> h(h0.begin() + d * s.batch * H, h0.begin() + (d + 1) * s.batch * H);
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

// ONNX LSTM, layout 0. W/R gate order is i, o, f, c; B is [Wb, Rb]; P is [Pi, Po, Pf].
void ReferenceLstm(const RnnShape& s, const std::string& direction, const std::vector<std::string>& activations,
                   const std::vector<float>& X, const std::vector<float>& W, const std::vector<float>& R,
                   const std::vector<float>& B, const std::vector<float>& h0, const std::vector<float>& c0,
                   const std::vector<float>& P, std::vector<float>& Y, std::vector<float>& Y_h,
                   std::vector<float>& Y_c) {
  const auto reverse = ReverseFlags(direction);
  const int64_t dirs = static_cast<int64_t>(reverse.size());
  const int64_t H = s.hidden;
  Y.assign(static_cast<size_t>(s.seq * dirs * s.batch * H), 0.0f);
  Y_h.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  Y_c.assign(static_cast<size_t>(dirs * s.batch * H), 0.0f);
  for (int64_t d = 0; d < dirs; ++d) {
    const std::string& f = activations[d * 3];
    const std::string& g = activations[d * 3 + 1];
    const std::string& hact = activations[d * 3 + 2];
    std::vector<double> h(h0.begin() + d * s.batch * H, h0.begin() + (d + 1) * s.batch * H);
    std::vector<double> c(c0.begin() + d * s.batch * H, c0.begin() + (d + 1) * s.batch * H);
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
      std::vector<double> next_h(h.size());
      std::vector<double> next_c(c.size());
      for (int64_t b = 0; b < s.batch; ++b) {
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

template <typename T>
void TestGru(const std::string& direction, const std::vector<std::string>& activations, bool linear_before_reset) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  const RnnShape s{3, 2, 3, 4};
  const int64_t dirs = direction == "bidirectional" ? 2 : 1;
  const auto X = MakeData(s.seq * s.batch * s.input, 1, 1.0f);
  const auto W = MakeData(dirs * 3 * s.hidden * s.input, 2, 0.5f);
  const auto R = MakeData(dirs * 3 * s.hidden * s.hidden, 3, 0.5f);
  const auto B = MakeData(dirs * 6 * s.hidden, 4, 0.25f);
  const auto h0 = MakeData(dirs * s.batch * s.hidden, 5, 0.5f);
  std::vector<float> Y, Y_h;
  ReferenceGru(s, direction, activations, linear_before_reset, X, W, R, B, h0, Y, Y_h);

  OpTester test("GRU", 14);
  test.AddAttribute("activations", activations);
  test.AddAttribute("direction", direction);
  test.AddAttribute<int64_t>("hidden_size", s.hidden);
  test.AddAttribute<int64_t>("linear_before_reset", linear_before_reset ? 1 : 0);
  test.AddInput<T>("X", {s.seq, s.batch, s.input}, Convert<T>(X));
  test.AddInput<T>("W", {dirs, 3 * s.hidden, s.input}, Convert<T>(W), true);
  test.AddInput<T>("R", {dirs, 3 * s.hidden, s.hidden}, Convert<T>(R), true);
  test.AddInput<T>("B", {dirs, 6 * s.hidden}, Convert<T>(B), true);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<T>("initial_h", {dirs, s.batch, s.hidden}, Convert<T>(h0));
  test.AddOutput<T>("Y", {s.seq, dirs, s.batch, s.hidden}, Convert<T>(Y));
  test.AddOutput<T>("Y_h", {dirs, s.batch, s.hidden}, Convert<T>(Y_h));
  test.SetOutputTolerance(Tolerance<T>(), Tolerance<T>());
  test.Config(WebGpuOnly()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <typename T>
void TestLstm(const std::string& direction, const std::vector<std::string>& activations) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  const RnnShape s{3, 2, 3, 4};
  const int64_t dirs = direction == "bidirectional" ? 2 : 1;
  const auto X = MakeData(s.seq * s.batch * s.input, 1, 1.0f);
  const auto W = MakeData(dirs * 4 * s.hidden * s.input, 2, 0.5f);
  const auto R = MakeData(dirs * 4 * s.hidden * s.hidden, 3, 0.5f);
  const auto B = MakeData(dirs * 8 * s.hidden, 4, 0.25f);
  const auto h0 = MakeData(dirs * s.batch * s.hidden, 5, 0.5f);
  const auto c0 = MakeData(dirs * s.batch * s.hidden, 6, 0.5f);
  const auto P = MakeData(dirs * 3 * s.hidden, 7, 0.5f);
  std::vector<float> Y, Y_h, Y_c;
  ReferenceLstm(s, direction, activations, X, W, R, B, h0, c0, P, Y, Y_h, Y_c);

  OpTester test("LSTM", 14);
  test.AddAttribute("activations", activations);
  test.AddAttribute("direction", direction);
  test.AddAttribute<int64_t>("hidden_size", s.hidden);
  test.AddInput<T>("X", {s.seq, s.batch, s.input}, Convert<T>(X));
  test.AddInput<T>("W", {dirs, 4 * s.hidden, s.input}, Convert<T>(W), true);
  test.AddInput<T>("R", {dirs, 4 * s.hidden, s.hidden}, Convert<T>(R), true);
  test.AddInput<T>("B", {dirs, 8 * s.hidden}, Convert<T>(B), true);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<T>("initial_h", {dirs, s.batch, s.hidden}, Convert<T>(h0));
  test.AddInput<T>("initial_c", {dirs, s.batch, s.hidden}, Convert<T>(c0));
  test.AddInput<T>("P", {dirs, 3 * s.hidden}, Convert<T>(P), true);
  test.AddOutput<T>("Y", {s.seq, dirs, s.batch, s.hidden}, Convert<T>(Y));
  test.AddOutput<T>("Y_h", {dirs, s.batch, s.hidden}, Convert<T>(Y_h));
  test.AddOutput<T>("Y_c", {dirs, s.batch, s.hidden}, Convert<T>(Y_c));
  test.SetOutputTolerance(Tolerance<T>(), Tolerance<T>());
  test.Config(WebGpuOnly()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <typename T>
void TestGruVariants() {
  for (bool lbr : {false, true}) {
    SCOPED_TRACE(lbr);
    TestGru<T>("forward", {"Sigmoid", "Tanh"}, lbr);
    TestGru<T>("reverse", {"Sigmoid", "Relu"}, lbr);
    TestGru<T>("bidirectional", {"Sigmoid", "Tanh", "Relu", "Relu"}, lbr);
  }
}

template <typename T>
void TestLstmVariants() {
  TestLstm<T>("forward", {"Sigmoid", "Tanh", "Tanh"});
  TestLstm<T>("reverse", {"Sigmoid", "Relu", "Relu"});
  TestLstm<T>("bidirectional", {"Sigmoid", "Tanh", "Tanh", "Relu", "Relu", "Relu"});
}

}  // namespace

TEST(GRUTest, WebGpu_Float_Reference) {
  TestGruVariants<float>();
}

TEST(GRUTest, WebGpu_Float16) {
  TestGruVariants<MLFloat16>();
}

TEST(LSTMTest, WebGpu_Float_Reference) {
  TestLstmVariants<float>();
}

TEST(LSTMTest, WebGpu_Float16) {
  TestLstmVariants<MLFloat16>();
}

}  // namespace test
}  // namespace onnxruntime
