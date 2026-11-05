// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_FUSED_STATE_SPACE_NORM_TEST)
// Standalone kernel-test mode: link fused_state_space_impl.cu, cudart and gtest_main, not ORT.
#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace onnxruntime::test {
TEST(FusedStateSpaceKernelTest, MambaNorm65536Groups) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  // Only normalization runs: the full operator retains its fixed 64/128 head/state dimensions.
  constexpr int rows = 1, slots = 1, heads = 65536, groups = heads, head_dim = 1, state_dim = 1;
  constexpr int width = heads * head_dim, channels = width + 2 * groups * state_dim;
  constexpr int stride = width + channels + heads;
  static_assert(rows == slots);
  constexpr float epsilon = 1e-5f;
  // BF16 values below are dyadic and exactly representable; no ORT type or allocator is needed.
  std::vector<uint16_t> projection(stride), output(width);
  std::vector<float> readout(width), gamma(width), expected(width);
  for (int j = 0; j < width; ++j) {
    const int gate_steps = j % 4 + 1;
    constexpr uint16_t gate_bits[] = {0x3e80, 0x3f00, 0x3f40, 0x3f80};  // 0.25, 0.5, 0.75, 1
    projection[j] = gate_bits[gate_steps - 1];
    const float gate = gate_steps * 0.25f;
    readout[j] = (j % 11 - 5) * 0.125f;
    gamma[j] = (j % 9 + 1) * 0.125f;
    const float v = readout[j] * (gate / (1.f + std::exp(-gate)));
    expected[j] = v / std::sqrt(v * v + epsilon) * gamma[j];
  }
  auto allocate = [](size_t bytes) {
    void* ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
    return std::unique_ptr<void, decltype(&cudaFree)>(ptr, cudaFree);
  };
  auto device_proj = allocate(projection.size() * sizeof(uint16_t));
  auto device_readout = allocate(readout.size() * sizeof(float));
  auto device_gamma = allocate(gamma.size() * sizeof(float));
  auto device_output = allocate(output.size() * sizeof(uint16_t));
  ASSERT_NE(device_proj, nullptr);
  ASSERT_NE(device_readout, nullptr);
  ASSERT_NE(device_gamma, nullptr);
  ASSERT_NE(device_output, nullptr);
  ASSERT_EQ(
      cudaMemcpy(device_proj.get(), projection.data(), projection.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_readout.get(), readout.data(), readout.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_gamma.get(), gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output.get(), 0xff, output.size() * sizeof(uint16_t)), cudaSuccess);
  // Reverting this production launch to dim3(rows, groups) exceeds CUDA's grid-y limit.
  ASSERT_EQ(contrib::cuda::LaunchMamba2VerifyNormalization(
                nullptr, rows, width, groups, stride, epsilon, device_proj.get(),
                static_cast<const float*>(device_readout.get()), static_cast<const float*>(device_gamma.get()),
                device_output.get()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int j = 0; j < width; ++j) {
    const uint32_t bits = static_cast<uint32_t>(output[j]) << 16;
    float actual;
    std::memcpy(&actual, &bits, sizeof(actual));
    ASSERT_TRUE(std::isfinite(actual)) << j;
    EXPECT_NEAR(actual, expected[j], 0.004f * std::abs(expected[j]) + 1e-6f) << j;
  }
}
}  // namespace onnxruntime::test
#else
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(USE_CUDA)
#include "core/framework/data_transfer.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"
#endif

namespace onnxruntime::test {
#if defined(USE_CUDA)
namespace {
void RunCuda(OpTester& tester) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
float Rounded(float x) { return BFloat16(x).ToFloat(); }

class MambaSession : public InferenceSession {
 public:
  using InferenceSession::InferenceSession;
  Status LoadModel(Model& model) {
    // The OpTester owns the model and outlives this session.
    return LoadWithLoader(
        [&model](std::shared_ptr<Model>& loaded) {
          loaded = std::shared_ptr<Model>(&model, [](Model*) {});
          return Status::OK();
        },
        "mamba_alias_test");
  }
};

class MambaTester : public OpTester {
 public:
  MambaTester() : OpTester("Mamba2Verify", 1, kMSDomain) {
    SetCustomOutputVerifier([this](const std::vector<OrtValue>& actual, const std::string& provider) {
      auto& expected = GetOutputData();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        CheckOrtValuesAreEqual(expected[i].def.Name(), expected[i].data, actual[i], expected[i].validation_params,
                               provider);
        const auto& want = expected[i].data.Get<Tensor>();
        if (!want.IsDataType<BFloat16>()) continue;
        const auto& got = actual[i].Get<Tensor>();
        ASSERT_EQ(want.Shape(), got.Shape());
        ASSERT_TRUE(got.IsDataType<BFloat16>());
        // The base BF16 checker checks the expected, not actual, NaN.
        for (int64_t j = 0; j < want.Shape().Size(); ++j) {
          if (std::isnan(want.Data<BFloat16>()[j].ToFloat())) {
            EXPECT_TRUE(std::isnan(got.Data<BFloat16>()[j].ToFloat())) << expected[i].def.Name() << ", " << j;
          }
        }
      }
    });
  }

  void CheckDeviceAliasing(int rows) {
    SetTestFunctionCalled();
    auto& model = BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    SessionOptions options;
    MambaSession session(options, GetEnvironment());
    auto provider = DefaultCudaExecutionProvider();
    ASSERT_NE(provider, nullptr);
    auto* ep = provider.get();
    const auto transfer = ep->GetDataTransfer();
    ASSERT_NE(transfer, nullptr);
    const auto allocators = ep->CreatePreferredAllocators();
    const OrtMemoryInfo* gpu_info = nullptr;
    for (const auto& allocator : allocators)
      if (allocator->Info().device.Type() == OrtDevice::GPU && allocator->Info().mem_type == OrtMemTypeDefault)
        gpu_info = &allocator->Info();
    ASSERT_NE(gpu_info, nullptr);
    const OrtMemoryInfo info = *gpu_info;
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    ASSERT_STATUS_OK(session.LoadModel(model));
    ASSERT_STATUS_OK(session.Initialize());
    const auto gpu_allocator = session.GetAllocator(info);
    ASSERT_NE(gpu_allocator, nullptr);
    const auto cpu_allocator = DefaultCpuExecutionProvider()->CreatePreferredAllocators()[0];
    std::unordered_map<std::string, OrtValue> cpu_feeds, gpu_feeds;
    std::vector<std::string> names;
    FillFeedsAndOutputNames(cpu_feeds, names);
    for (const auto& [name, value] : cpu_feeds) {
      const auto& tensor = value.Get<Tensor>();
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(), gpu_allocator, gpu_feeds[name]);
    }
    std::vector<OrtValue> baseline;
    // Test convolution-only, SSM-only and simultaneous input/output device aliasing.
    for (int mask = 0; mask < 4; ++mask) {
      for (const auto& [name, value] : cpu_feeds)
        ASSERT_STATUS_OK(transfer->CopyTensor(value.Get<Tensor>(), *gpu_feeds[name].GetMutable<Tensor>()));
      std::unique_ptr<IOBinding> binding;
      ASSERT_STATUS_OK(session.NewIOBinding(&binding));
      for (const auto& [name, value] : gpu_feeds) ASSERT_STATUS_OK(binding->BindInput(name, value));
      for (const auto& name : names) {
        if (name == "conv_snapshots" && (mask & 1)) {
          ASSERT_STATUS_OK(binding->BindOutput(name, gpu_feeds.at("conv_state")));
        } else if (name == "ssm_snapshots" && (mask & 2)) {
          ASSERT_STATUS_OK(binding->BindOutput(name, gpu_feeds.at("ssm_state")));
        } else {
          ASSERT_STATUS_OK(binding->BindOutput(name, info.device));
        }
      }
      ASSERT_STATUS_OK(binding->SynchronizeInputs());
      ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
      ASSERT_STATUS_OK(binding->SynchronizeOutputs());
      const auto& outputs = binding->GetOutputs();
      ASSERT_EQ(outputs.size(), names.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& tensor = outputs[i].Get<Tensor>();
        OrtValue actual;
        Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(), cpu_allocator, actual);
        ASSERT_STATUS_OK(transfer->CopyTensor(tensor, *actual.GetMutable<Tensor>()));
        if (mask == 0) {
          baseline.push_back(actual);
        } else {
          EXPECT_EQ(
              std::memcmp(actual.Get<Tensor>().DataRaw(), baseline[i].Get<Tensor>().DataRaw(), tensor.SizeInBytes()), 0)
              << names[i] << ", alias mask " << mask;
        }
        const char* input_name = names[i] == "conv_snapshots"  ? "conv_state"
                                 : names[i] == "ssm_snapshots" ? "ssm_state"
                                                               : nullptr;
        if (input_name) {
          const auto& input = cpu_feeds.at(input_name).Get<Tensor>();
          const size_t offset = input.SizeInBytes() / static_cast<size_t>(input.Shape()[0]) * rows;
          EXPECT_EQ(std::memcmp(static_cast<const uint8_t*>(actual.Get<Tensor>().DataRaw()) + offset,
                                static_cast<const uint8_t*>(input.DataRaw()) + offset, input.SizeInBytes() - offset),
                    0);
          const int bit = names[i] == "conv_snapshots" ? 1 : 2;
          if (mask & bit) {
            EXPECT_EQ(tensor.DataRaw(), gpu_feeds.at(input_name).Get<Tensor>().DataRaw());
          } else {
            EXPECT_NE(tensor.DataRaw(), gpu_feeds.at(input_name).Get<Tensor>().DataRaw());
          }
        }
      }
    }
  }
};

void TestMambaNonuniform(int rows, int selected, bool alias = false) {
  constexpr int heads = 4, groups = 2, di = heads * 64, channels = di + groups * 256;
  constexpr int stride = di + channels + heads, state_stride = heads * 128 * 64;
  const int slots = std::max(rows + 2, 4);
  const bool valid = selected >= 0 && selected < slots;
  std::vector<BFloat16> proj(rows * stride), conv(slots * channels * 3), cw(channels * 4), cb(channels);
  std::vector<float> state(slots * state_stride), decay(heads), skip(heads), dt_bias(heads), gamma(di);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < di; ++j) proj[r * stride + j] = BFloat16((j % 11 - 5 + r) / 16.f);
    for (int j = 0; j < channels; ++j) proj[r * stride + di + j] = BFloat16((j % 19 - 7 + r * 2 + j / 128) / 64.f);
    for (int h = 0; h < heads; ++h) proj[r * stride + di + channels + h] = BFloat16((r - h - 2) / 16.f);
  }
  for (int s = 0; s < slots; ++s) {
    for (int j = 0; j < channels; ++j)
      for (int k = 0; k < 3; ++k) conv[(s * channels + j) * 3 + k] = BFloat16((s * 3 + j % 13 - k * 2) / 128.f);
    for (int h = 0; h < heads; ++h)
      for (int n = 0; n < 128; ++n)
        for (int p = 0; p < 64; ++p)
          state[s * state_stride + (h * 128 + n) * 64 + p] = (s * 2 + h * 3 + n % 17 - p % 7) / 1024.f;
  }
  for (int j = 0; j < channels; ++j) {
    for (int k = 0; k < 4; ++k) cw[j * 4 + k] = BFloat16((k + 1 + j % 3) / 16.f);
    cb[j] = BFloat16((j % 7 - 3) / 128.f);
  }
  for (int h = 0; h < heads; ++h) {
    decay[h] = -(h + 1) / 16.f;
    skip[h] = (h + 1) / 8.f;
    dt_bias[h] = (h - 1) / 32.f;
  }
  for (int j = 0; j < di; ++j) gamma[j] = (j % 9 + 1) / 8.f;
  auto conv_expected = conv;
  auto state_expected = state;
  std::vector<BFloat16> xbc(rows * channels), y(rows * di);
  std::vector<float> readout(rows * di), st(state_stride), history(channels * 3);
  if (valid) {
    std::copy_n(state.begin() + selected * state_stride, state_stride, st.begin());
    for (int j = 0; j < channels * 3; ++j) history[j] = conv[selected * channels * 3 + j].ToFloat();
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < channels; ++j) {
      const float input = proj[r * stride + di + j].ToFloat();
      float acc = cb[j].ToFloat();
      for (int k = 0; k < 3; ++k) acc = std::fma(cw[j * 4 + k].ToFloat(), history[j * 3 + k], acc);
      acc = std::fma(cw[j * 4 + 3].ToFloat(), input, acc);
      xbc[r * channels + j] = BFloat16(valid ? acc / (1.f + std::exp(-acc)) : nan);
      history[j * 3] = history[j * 3 + 1];
      history[j * 3 + 1] = history[j * 3 + 2];
      history[j * 3 + 2] = input;
      for (int k = 0; k < 3; ++k)
        conv_expected[(r * channels + j) * 3 + k] = BFloat16(valid ? history[j * 3 + k] : nan);
    }
    for (int h = 0; h < heads; ++h) {
      const int g = h / (heads / groups);
      const float raw = proj[r * stride + di + channels + h].ToFloat() + dt_bias[h];
      const float dt = std::max(std::log1p(std::exp(raw)), 0.01f);
      const float dA = std::exp(decay[h] * dt);
      for (int p = 0; p < 64; ++p) {
        const float x = xbc[r * channels + h * 64 + p].ToFloat();
        const int parts = rows == 1 ? 2 : 4;
        float result = skip[h] * x;
        for (int q = 0; q < parts; ++q) {
          float partial = 0.f;
          for (int n = q * (128 / parts); n < (q + 1) * (128 / parts); ++n) {
            const int pos = (h * 128 + n) * 64 + p;
            const float b = xbc[r * channels + di + g * 128 + n].ToFloat();
            const float c = xbc[r * channels + di + groups * 128 + g * 128 + n].ToFloat();
            st[pos] = std::fma(st[pos], dA, (dt * x) * b);
            state_expected[r * state_stride + pos] = valid ? st[pos] : nan;
            partial = std::fma(c, st[pos], partial);
          }
          result += partial;
        }
        readout[r * di + h * 64 + p] = valid ? result : nan;
      }
    }
    for (int g = 0; g < groups; ++g) {
      float square = 0.f;
      std::vector<float> gated(di / groups);
      for (int j = 0; j < di / groups; ++j) {
        const int col = g * (di / groups) + j;
        const float gate = proj[r * stride + col].ToFloat();
        gated[j] = readout[r * di + col] * (gate / (1.f + std::exp(-gate)));
        square += gated[j] * gated[j];
      }
      for (int j = 0; j < di / groups; ++j) {
        const int col = g * (di / groups) + j;
        y[r * di + col] = BFloat16(gated[j] / std::sqrt(square / (di / groups) + 1e-5f) * gamma[col]);
      }
    }
  }
  MambaTester tester;
  tester.AddAttribute("heads", int64_t{heads});
  tester.AddAttribute("n_groups", int64_t{groups});
  tester.AddAttribute("time_step_min", 0.01f);
  tester.AddInput<BFloat16>("projection", {rows, stride}, proj);
  tester.AddInput<BFloat16>("conv_state", {slots, channels, 3}, conv);
  tester.AddInput<float>("ssm_state", {slots, heads, 128, 64}, state);
  tester.AddInput<BFloat16>("conv_weight", {channels, 4}, cw);
  tester.AddInput<BFloat16>("conv_bias", {channels}, cb);
  tester.AddInput<float>("dt_bias", {heads}, dt_bias);
  tester.AddInput<float>("A_negative", {heads}, decay);
  tester.AddInput<float>("D", {heads}, skip);
  tester.AddInput<float>("norm_weight", {di}, gamma);
  tester.AddInput<int32_t>("state_row", {1}, {selected});
  tester.AddOutput<BFloat16>("Y", {rows, di}, y);
  tester.AddOutput<BFloat16>("conv_snapshots", {slots, channels, 3}, conv_expected);
  tester.AddOutput<float>("ssm_snapshots", {slots, heads, 128, 64}, state_expected);
  tester.AddOutput<BFloat16>("xbc", {rows, channels}, xbc);
  tester.AddOutput<float>("readout", {rows, di}, readout);
  tester.SetOutputAbsErr("ssm_snapshots", 2e-5f);
  tester.SetOutputAbsErr("readout", 2e-4f);
  RunCuda(tester);
  if (alias) tester.CheckDeviceAliasing(rows);
}

void TestMambaSnapshots(int rows, int selected, int heads = 2) {
  const int di = heads * 64, channels = di + 256, stride = di + channels + heads;
  const int slots = std::max(rows, selected + 1) + 1;
  std::vector<BFloat16> proj(rows * stride), conv(slots * channels * 3), cw(channels * 4);
  std::vector<float> state(static_cast<size_t>(slots) * heads * 128 * 64);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < di; ++j) proj[r * stride + j] = BFloat16(0.5f);
    for (int j = 0; j < channels; ++j) proj[r * stride + di + j] = BFloat16(0.25f + r * 0.03125f);
    for (int h = 0; h < heads; ++h) proj[r * stride + di + channels + h] = BFloat16(-0.25f + r * 0.03125f);
  }
  for (int s = 0; s < slots; ++s) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < 3; ++k) conv[(s * channels + j) * 3 + k] = BFloat16((s + 1) * (k + 1) * 0.03125f);
    }
    std::fill_n(state.begin() + static_cast<size_t>(s) * heads * 128 * 64, heads * 128 * 64, (s + 1) * 0.015625f);
  }
  for (int j = 0; j < channels; ++j) {
    cw[j * 4] = BFloat16(0.125f);
    cw[j * 4 + 1] = BFloat16(0.25f);
    cw[j * 4 + 2] = BFloat16(0.5f);
    cw[j * 4 + 3] = BFloat16(1.f);
  }
  auto conv_expected = conv;
  auto state_expected = state;
  std::vector<BFloat16> xbc(rows * channels), y(rows * di);
  std::vector<float> readout(rows * di), st(heads, (selected + 1) * 0.015625f);
  float history[3] = {(selected + 1) * 0.03125f, (selected + 1) * 0.0625f, (selected + 1) * 0.09375f};
  for (int r = 0; r < rows; ++r) {
    const float input = Rounded(0.25f + r * 0.03125f);
    const float value = -0.125f + 0.125f * history[0] + 0.25f * history[1] + 0.5f * history[2] + input;
    const float x = Rounded(value / (1.f + std::exp(-value)));
    std::fill_n(xbc.begin() + r * channels, channels, BFloat16(x));
    history[0] = history[1];
    history[1] = history[2];
    history[2] = input;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < 3; ++k) conv_expected[(r * channels + j) * 3 + k] = BFloat16(history[k]);
    }
    const float dt = std::max(std::log1p(std::exp(-0.125f + r * 0.03125f)), 0.01f);
    float mean_square = 0.f;
    for (int h = 0; h < heads; ++h) {
      st[h] = std::fma(st[h], std::exp((-0.125f - h * 0.0078125f) * dt), (dt * x) * x);
      std::fill_n(state_expected.begin() + (static_cast<size_t>(r) * heads + h) * 128 * 64, 128 * 64, st[h]);
      float partial = 0.f;
      const int parts = rows == 1 ? 2 : 4;
      for (int n = 0; n < 128 / parts; ++n) partial = std::fma(x, st[h], partial);
      float result = 0.25f * x;
      for (int q = 0; q < parts; ++q) result += partial;
      std::fill_n(readout.begin() + r * di + h * 64, 64, result);
      const float gated = result * (0.5f / (1.f + std::exp(-0.5f)));
      mean_square += gated * gated / heads;
    }
    for (int j = 0; j < di; ++j) {
      const float gated = readout[r * di + j] * (0.5f / (1.f + std::exp(-0.5f)));
      y[r * di + j] = BFloat16(gated / std::sqrt(mean_square + 1e-5f) * 1.25f);
    }
  }
  OpTester tester("Mamba2Verify", 1, kMSDomain);
  tester.AddAttribute("heads", static_cast<int64_t>(heads));
  tester.AddAttribute("n_groups", int64_t{1});
  tester.AddAttribute("time_step_min", 0.01f);
  tester.AddInput<BFloat16>("projection", {rows, stride}, proj);
  tester.AddInput<BFloat16>("conv_state", {slots, channels, 3}, conv);
  tester.AddInput<float>("ssm_state", {slots, heads, 128, 64}, state);
  tester.AddInput<BFloat16>("conv_weight", {channels, 1, 4}, cw);
  tester.AddInput<BFloat16>("conv_bias", {channels}, std::vector<BFloat16>(channels, BFloat16(-0.125f)));
  tester.AddInput<float>("dt_bias", {heads}, std::vector<float>(heads, 0.125f));
  std::vector<float> decay(heads);
  for (int h = 0; h < heads; ++h) decay[h] = -0.125f - h * 0.0078125f;
  tester.AddInput<float>("A_negative", {heads}, decay);
  tester.AddInput<float>("D", {heads}, std::vector<float>(heads, 0.25f));
  tester.AddInput<float>("norm_weight", {di}, std::vector<float>(di, 1.25f));
  tester.AddInput<int32_t>("state_row", {1}, {selected});
  tester.AddOutput<BFloat16>("Y", {rows, di}, y);
  tester.AddOutput<BFloat16>("conv_snapshots", {slots, channels, 3}, conv_expected);
  tester.AddOutput<float>("ssm_snapshots", {slots, heads, 128, 64}, state_expected);
  tester.AddOutput<BFloat16>("xbc", {rows, channels}, xbc);
  tester.AddOutput<float>("readout", {rows, di}, readout);
  tester.SetOutputAbsErr("ssm_snapshots", 2e-5f);
  tester.SetOutputAbsErr("readout", 2e-4f);
  RunCuda(tester);
}
}  // namespace

TEST(FusedStateSpaceTest, MambaRecurrenceAndAcceptedSnapshot) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaSnapshots(1, 3);
  TestMambaSnapshots(3, 1);
  TestMambaSnapshots(8, 7);
  TestMambaSnapshots(16, 5);
}

TEST(FusedStateSpaceTest, MambaEightRows1024WideNorm) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaSnapshots(8, 3, 16);
}

TEST(FusedStateSpaceTest, MambaMultigroupNonuniformStateAndRankTwoWeights) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaNonuniform(1, 3);
  TestMambaNonuniform(3, 1);
  TestMambaNonuniform(8, 3);
}

TEST(FusedStateSpaceTest, MambaEightRowsDeviceStateAliasing) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaNonuniform(8, 3, true);
}

TEST(FusedStateSpaceTest, MambaInvalidSelectorsProduceNaNWithoutStateReads) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8}) {
    TestMambaNonuniform(rows, -1);
    TestMambaNonuniform(rows, std::max(rows + 2, 4));
  }
}

TEST(FusedStateSpaceTest, RouterBiasOnlySelectionAndNonunitScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("BiasCorrectedMoERoute", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{2});
  tester.AddAttribute("scale", 5.f);
  tester.AddInput<BFloat16>("X", {2, 256}, std::vector<BFloat16>(512, BFloat16(1.f)));
  std::vector<BFloat16> weight(4 * 256, BFloat16(0.f));
  weight[256] = BFloat16(1.f);
  weight[512] = BFloat16(-1.f);
  tester.AddInput<BFloat16>("W", {4, 256}, weight);
  tester.AddInput<float>("bias", {4}, {0.f, 0.f, 2.f, 0.f});
  const float high = 1.f / (1.f + std::exp(-1.f)), low = 1.f / (1.f + std::exp(1.f));
  tester.AddOutput<int32_t>("indices", {2, 2}, {2, 1, 2, 1});
  tester.AddOutput<float>("weights", {2, 2}, {low * 5.f, high * 5.f, low * 5.f, high * 5.f});
  tester.AddOutput<float>("scores", {2, 4}, {0.5f, high, low, 0.5f, 0.5f, high, low, 0.5f});
  RunCuda(tester);
}

TEST(FusedStateSpaceTest, RouterTiesPreferLowerExpert) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("BiasCorrectedMoERoute", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{3});
  tester.AddInput<BFloat16>("X", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
  tester.AddInput<BFloat16>("W", {40, 256}, std::vector<BFloat16>(40 * 256, BFloat16(0.f)));
  tester.AddInput<float>("bias", {40}, std::vector<float>(40, 0.f));
  tester.AddOutput<int32_t>("indices", {1, 3}, {0, 1, 2});
  tester.AddOutput<float>("weights", {1, 3}, {1.f / 3, 1.f / 3, 1.f / 3});
  tester.AddOutput<float>("scores", {1, 40}, std::vector<float>(40, 0.5f));
  RunCuda(tester);
}

TEST(FusedStateSpaceTest, RouterNegativeInfinityIsAnEligibleScore) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("BiasCorrectedMoERoute", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{2});
  tester.AddInput<BFloat16>("X", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
  tester.AddInput<BFloat16>("W", {2, 256}, std::vector<BFloat16>(512, BFloat16(0.f)));
  tester.AddInput<float>("bias", {2}, {0.f, -std::numeric_limits<float>::infinity()});
  tester.AddOutput<int32_t>("indices", {1, 2}, {0, 1});
  tester.AddOutput<float>("weights", {1, 2}, {0.5f, 0.5f});
  tester.AddOutput<float>("scores", {1, 2}, {0.5f, 0.5f});
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester);
}

TEST(FusedStateSpaceTest, ReluSquaredSingleRoundingAndEmpty) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("ReluSquared", 1, kMSDomain);
  tester.AddInput<BFloat16>("X", {5}, {BFloat16(-2.f), BFloat16(0.f), BFloat16(1.125f), BFloat16(2.5f), BFloat16(std::numeric_limits<float>::quiet_NaN())});
  tester.AddOutput<BFloat16>("Y", {5}, {BFloat16(0.f), BFloat16(0.f), BFloat16(1.125f * 1.125f), BFloat16(6.25f), BFloat16(0.f)});
  RunCuda(tester);
  OpTester empty("ReluSquared", 1, kMSDomain);
  empty.AddInput<BFloat16>("X", {0}, {});
  empty.AddOutput<BFloat16>("Y", {0}, {});
  RunCuda(empty);
}
#endif
}  // namespace onnxruntime::test
#endif  // ORT_FUSED_STATE_SPACE_NORM_TEST
