// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
#include "core/framework/data_transfer.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"
#endif

namespace onnxruntime::test {
namespace {
class MambaSchemaTester : public OpTester {
 public:
  MambaSchemaTester(bool known_shapes, const char* op) : OpTester(op, 1, kMSDomain, known_shapes) {}
  Status Resolve() {
    SetTestFunctionCalled();
    return BuildModel().MainGraph().Resolve();
  }
};

using MambaInputShapes = std::array<std::vector<int64_t>, 10>;
MambaInputShapes MambaShapes() {
  return {{{3, 514}, {4, 384, 3}, {4, 2, 128, 64}, {384, 4}, {384}, {2}, {2}, {2}, {128}, {1}}};
}

void TestMambaSchema(const MambaInputShapes& shapes, const std::string& error = "", bool known_shapes = true,
                     int64_t heads = 2, int64_t groups = 1, int intermediates = 3, const char* op = "Mamba2Verify") {
  MambaSchemaTester tester(known_shapes, op);
  tester.AddAttribute("heads", heads);
  tester.AddAttribute("n_groups", groups);
  constexpr std::array<const char*, 10> names = {
      "projection", "conv_state", "ssm_state", "conv_weight", "conv_bias",
      "dt_bias", "A_negative", "D", "norm_weight", "state_row"};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto size = static_cast<size_t>(TensorShape(shapes[i]).Size());
    if (i == 9) {
      tester.AddInput<int32_t>(names[i], shapes[i], std::vector<int32_t>(size));
    } else if (i == 2 || (i >= 5 && i <= 8)) {
      tester.AddInput<float>(names[i], shapes[i], std::vector<float>(size));
    } else {
      tester.AddInput<BFloat16>(names[i], shapes[i], std::vector<BFloat16>(size));
    }
  }
  const int64_t rows = shapes[0][0];
  const auto size_of = [](const std::vector<int64_t>& shape) { return static_cast<size_t>(TensorShape(shape).Size()); };
  tester.AddOutput<BFloat16>("Y", {rows, 128}, std::vector<BFloat16>(rows * 128));
  tester.AddOutput<BFloat16>("conv_snapshots", shapes[1], std::vector<BFloat16>(size_of(shapes[1])));
  tester.AddOutput<float>("ssm_snapshots", shapes[2], std::vector<float>(size_of(shapes[2])));
  if (intermediates & 1) {
    tester.AddOutput<BFloat16>("xbc", {rows, 384}, std::vector<BFloat16>(rows * 384));
  } else if (intermediates & 2) {
    tester.AddOptionalOutputEdge<BFloat16>();
  }
  if (intermediates & 2) tester.AddOutput<float>("readout", {rows, 128}, std::vector<float>(rows * 128));
  const auto status = tester.Resolve();
  if (error.empty()) {
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  } else {
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(error), std::string::npos) << status.ErrorMessage();
  }
}
}  // namespace

TEST(FusedStateSpaceSchemaTest, MambaKnownAndUnknownGeometry) {
  auto shapes = MambaShapes();
  TestMambaSchema(shapes);
  shapes[3] = {384, 1, 4};
  TestMambaSchema(shapes);
  TestMambaSchema(shapes, "", false);
}

TEST(FusedStateSpaceSchemaTest, MambaOptionalIntermediates) {
  for (int intermediates = 0; intermediates < 4; ++intermediates) {
    SCOPED_TRACE(intermediates);
    for (bool known_shapes : {false, true}) {
      TestMambaSchema(MambaShapes(), "", known_shapes, 2, 1, intermediates);
    }
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(FusedStateSpaceSchemaTest, MambaInvalidHeadGroupDiagnostic) {
  for (const auto& attributes : {std::pair{0, 1}, {2, 0}, {3, 2}, {std::numeric_limits<int>::max() / 1024 + 1, 1}}) {
    const auto [heads, groups] = attributes;
    TestMambaSchema(MambaShapes(), "got heads=" + std::to_string(heads) + ", n_groups=" + std::to_string(groups),
                    true, heads, groups);
  }
}

TEST(FusedStateSpaceSchemaTest, MambaRejectsRankOneConvolutionWeights) {
  auto shapes = MambaShapes();
  shapes[3] = {384 * 4};
  TestMambaSchema(shapes, "Invalid input rank at 3");
}

TEST(FusedStateSpaceSchemaTest, MambaRejectsKnownInvalidGeometry) {
  for (const auto& entry : {std::pair{0, 1}, {1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 0}, {3, 1}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}}) {
    SCOPED_TRACE("input " + std::to_string(entry.first) + " axis " + std::to_string(entry.second));
    auto shapes = MambaShapes();
    ++shapes[entry.first][entry.second];
    TestMambaSchema(shapes, "Invalid input dimension");
  }
  auto shapes = MambaShapes();
  shapes[3] = {384, 2, 4};
  TestMambaSchema(shapes, "Invalid input dimension at 3 axis 1");
  for (size_t input : {size_t{1}, size_t{2}}) {
    shapes = MambaShapes();
    shapes[input][0] = 2;
    TestMambaSchema(shapes, "Snapshot slots must be >= R");
  }
  shapes = MambaShapes();
  shapes[2][0] = 5;
  TestMambaSchema(shapes, "Snapshot slot dimensions must match");
}

TEST(FusedStateSpaceSchemaTest, MambaPrefillRowsAndSlots) {
  auto shapes = MambaShapes();
  // Prefill writes only slot zero, so a single slot serves any chunk length up to 2048 rows.
  for (int64_t rows : {3, 17, 2048}) {
    shapes[0][0] = rows;
    for (int64_t slots : {1, 4}) {
      shapes[1][0] = slots;
      shapes[2][0] = slots;
      TestMambaSchema(shapes, "", true, 2, 1, 3, "Mamba2Prefill");
    }
  }
  shapes[0][0] = 2049;
  TestMambaSchema(shapes, "Rows must be in [1,2048]", true, 2, 1, 3, "Mamba2Prefill");
  shapes = MambaShapes();
  shapes[0][0] = 17;
  shapes[1][0] = 17;
  shapes[2][0] = 17;
  TestMambaSchema(shapes, "Rows must be in [1,16]");
  shapes[1][0] = 0;
  shapes[2][0] = 0;
  TestMambaSchema(shapes, "Snapshot slots must be >= 1", true, 2, 1, 3, "Mamba2Prefill");
}
#endif

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace {
void RunCuda(OpTester& tester) {
  std::vector<std::unique_ptr<IExecutionProvider> > providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
float Rounded(float x) { return BFloat16(x).ToFloat(); }
float MambaTimeStep(float raw, float time_step_min) {
  const float dt = raw > 20.f ? raw : std::log1p(std::exp(raw));
  return dt < time_step_min ? time_step_min : dt;
}

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
  explicit MambaTester(const char* op = "Mamba2Verify") : OpTester(op, 1, kMSDomain) {
    SetCustomOutputVerifier([this](const std::vector<OrtValue>& actual, const std::string& provider) {
      auto& expected = GetOutputData();
      size_t index = 0;
      for (const auto& output : expected) {
        if (!output.def.Exists()) continue;
        ASSERT_LT(index, actual.size());
        const auto& value = actual[index++];
        const auto& name = output.def.Name();
        const bool check_ulp = name == "Y" || name == "xbc";
        if (!check_ulp) {
          CheckOrtValuesAreEqual(name, output.data, value, output.validation_params, provider);
        }
        const auto& want = output.data.Get<Tensor>();
        if (!want.IsDataType<BFloat16>()) continue;
        const auto& got = value.Get<Tensor>();
        ASSERT_EQ(want.Shape(), got.Shape());
        ASSERT_TRUE(got.IsDataType<BFloat16>());
        int64_t finite_count = 0, mismatches = 0;
        // The base BF16 checker checks the expected, not actual, NaN.
        for (int64_t j = 0; j < want.Shape().Size(); ++j) {
          const auto reference = want.Data<BFloat16>()[j], result = got.Data<BFloat16>()[j];
          if (std::isnan(reference.ToFloat())) {
            EXPECT_TRUE(std::isnan(result.ToFloat())) << name << ", " << j;
          } else if (check_ulp) {
            ASSERT_TRUE(std::isfinite(reference.ToFloat()));
            ASSERT_TRUE(std::isfinite(result.ToFloat())) << name << ", " << j;
            // Monotonic BF16 keys give adjacent values distance one across either sign.
            const auto key = [](uint16_t bits) {
              return bits & 0x8000 ? 0x8000 - (bits & 0x7fff) : 0x8000 + bits;
            };
            const int ulps = std::abs(key(reference.val) - key(result.val));
            EXPECT_LE(ulps, 1) << name << ", " << j;
            mismatches += ulps != 0;
            ++finite_count;
          }
        }
        // Allow rare transcendental/reduction rounding boundaries, not systematic extra BF16 casts.
        if (check_ulp) {
          EXPECT_LE(mismatches * 1000, finite_count) << name << ": more than 0.1% differ";
        }
      }
      ASSERT_EQ(index, actual.size());
    });
  }

  // written_slots is the number of leading state slots the operator overwrites; later slots must be preserved.
  void CheckDeviceAliasing(int written_slots, bool capture = false) {
    SetTestFunctionCalled();
    auto& model = BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    SessionOptions options;
    MambaSession session(options, GetEnvironment());
    OrtCUDAProviderOptionsV2 provider_options{};
    provider_options.enable_cuda_graph = capture;
    auto provider = CudaExecutionProviderWithOptions(&provider_options);
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
    for (int mask = 0; mask < (capture ? 1 : 4); ++mask) {
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
      for (int run = 0; run < (capture ? 4 : 1); ++run) {
        ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
        ASSERT_STATUS_OK(binding->SynchronizeOutputs());
        const auto& outputs = binding->GetOutputs();
        ASSERT_EQ(outputs.size(), names.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          const auto& tensor = outputs[i].Get<Tensor>();
          OrtValue actual;
          Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(), cpu_allocator, actual);
          ASSERT_STATUS_OK(transfer->CopyTensor(tensor, *actual.GetMutable<Tensor>()));
          if (mask == 0 && run == 0) {
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
            const size_t offset = input.SizeInBytes() / static_cast<size_t>(input.Shape()[0]) * written_slots;
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
      if (capture) {
        EXPECT_TRUE(ep->IsGraphCaptured(0));
      }
    }
  }
};

enum class MambaDtCase { Default,
                         Extremes,
                         ProjectionNaN,
                         BiasNaN };

void TestMambaNonuniform(int rows, int selected, bool alias = false, bool trailing_slots = true,
                         int heads = 4, int groups = 2, float time_step_min = 0.01f,
                         MambaDtCase dt_case = MambaDtCase::Default, int intermediates = 3,
                         bool capture = false, bool prefill = false) {
  const int di = heads * 64, channels = di + groups * 256;
  const int stride = di + channels + heads, state_stride = heads * 128 * 64;
  // Mamba2Prefill writes only slot zero, so it needs neither one slot per row nor trailing rows.
  const int slots = prefill ? (trailing_slots ? 4 : 1) : (trailing_slots ? std::max(rows + 2, 4) : rows);
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
  if (dt_case == MambaDtCase::Extremes) {
    dt_bias[0] = -8.f;
    dt_bias[1] = 100.f;
  } else if (dt_case == MambaDtCase::ProjectionNaN) {
    proj[std::min(rows - 1, 3) * stride + di + channels + 1] = BFloat16(std::numeric_limits<float>::quiet_NaN());
  } else if (dt_case == MambaDtCase::BiasNaN) {
    dt_bias[1] = std::numeric_limits<float>::quiet_NaN();
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
      if (!prefill || r == rows - 1) {
        const int slot = prefill ? 0 : r;
        for (int k = 0; k < 3; ++k)
          conv_expected[(slot * channels + j) * 3 + k] = BFloat16(valid ? history[j * 3 + k] : nan);
      }
    }
    for (int h = 0; h < heads; ++h) {
      const int g = h / (heads / groups);
      const float raw = proj[r * stride + di + channels + h].ToFloat() + dt_bias[h];
      const float dt = MambaTimeStep(raw, time_step_min);
      const float dA = std::exp(decay[h] * dt);
      for (int p = 0; p < 64; ++p) {
        const float x = xbc[r * channels + h * 64 + p].ToFloat();
        const int parts = (!prefill && rows == 1 && slots == 1) ? 2 : 4;
        float result = skip[h] * x;
        for (int q = 0; q < parts; ++q) {
          float partial = 0.f;
          for (int n = q * (128 / parts); n < (q + 1) * (128 / parts); ++n) {
            const int pos = (h * 128 + n) * 64 + p;
            const float b = xbc[r * channels + di + g * 128 + n].ToFloat();
            const float c = xbc[r * channels + di + groups * 128 + g * 128 + n].ToFloat();
            st[pos] = std::fma(st[pos], dA, (dt * x) * b);
            if (!prefill || r == rows - 1) state_expected[(prefill ? 0 : r) * state_stride + pos] = valid ? st[pos] : nan;
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
  MambaTester tester(prefill ? "Mamba2Prefill" : "Mamba2Verify");
  tester.AddAttribute("heads", static_cast<int64_t>(heads));
  tester.AddAttribute("n_groups", static_cast<int64_t>(groups));
  tester.AddAttribute("time_step_min", time_step_min);
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
  if (intermediates & 1) {
    tester.AddOutput<BFloat16>("xbc", {rows, channels}, xbc);
  } else if (intermediates & 2) {
    tester.AddOptionalOutputEdge<BFloat16>();
  }
  if (intermediates & 2) tester.AddOutput<float>("readout", {rows, di}, readout);
  tester.SetOutputAbsErr("conv_snapshots", 0.f);
  tester.SetOutputRelErr("conv_snapshots", 0.f);
  tester.SetOutputAbsErr("ssm_snapshots", 2e-5f);
  if (intermediates & 2) tester.SetOutputAbsErr("readout", 2e-4f);
  RunCuda(tester);
  if (alias || capture) tester.CheckDeviceAliasing(prefill ? 1 : rows, capture);
}

void TestMambaSnapshots(int rows, int selected, int heads = 2) {
  const int di = heads * 64, channels = di + 256, stride = di + channels + heads;
  const int slots = std::max(rows, selected + 1) + 1;
  constexpr float time_step_min = 0.01f;
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
    const float dt = MambaTimeStep(-0.125f + r * 0.03125f, time_step_min);
    float mean_square = 0.f;
    for (int h = 0; h < heads; ++h) {
      st[h] = std::fma(st[h], std::exp((-0.125f - h * 0.0078125f) * dt), (dt * x) * x);
      std::fill_n(state_expected.begin() + (static_cast<size_t>(r) * heads + h) * 128 * 64, 128 * 64, st[h]);
      float partial = 0.f;
      const int parts = (rows == 1 && slots == 1) ? 2 : 4;
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
  MambaTester tester;
  tester.AddAttribute("heads", static_cast<int64_t>(heads));
  tester.AddAttribute("n_groups", int64_t{1});
  tester.AddAttribute("time_step_min", time_step_min);
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
  tester.SetOutputAbsErr("conv_snapshots", 0.f);
  tester.SetOutputRelErr("conv_snapshots", 0.f);
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
  TestMambaNonuniform(8, 3, false, true, 16, 1);
  TestMambaNonuniform(8, 3, false, true, 32, 2);
}

TEST(FusedStateSpaceTest, MambaTimeStepClampAndLargeRaw) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8}) {
    for (float time_step_min : {0.01f, 0.75f}) {
      TestMambaNonuniform(rows, 0, false, true, 4, 2, time_step_min, MambaDtCase::Extremes);
    }
  }
}

TEST(FusedStateSpaceTest, MambaTimeStepNaNPropagates) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8}) {
    TestMambaNonuniform(rows, 0, false, true, 4, 2, 0.01f, MambaDtCase::ProjectionNaN);
    TestMambaNonuniform(rows, 0, false, true, 4, 2, 0.01f, MambaDtCase::BiasNaN);
  }
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

TEST(FusedStateSpaceTest, MambaOptionalIntermediates) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int intermediates = 0; intermediates < 4; ++intermediates) {
    SCOPED_TRACE(intermediates);
    for (int rows : {1, 8}) {
      TestMambaNonuniform(rows, 3, true, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates);
      TestMambaNonuniform(rows, -1, false, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates);
      TestMambaNonuniform(rows, std::max(rows + 2, 4), false, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates);
      TestMambaNonuniform(rows, rows - 1, false, false, 4, 2, 0.01f, MambaDtCase::Default, intermediates);
    }
  }
}

TEST(FusedStateSpaceTest, MambaOptionalIntermediatesCudaGraph) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int intermediates = 0; intermediates < 4; ++intermediates) {
    SCOPED_TRACE(intermediates);
    for (int rows : {1, 8}) {
      TestMambaNonuniform(rows, 3, false, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates, true);
    }
  }
}

TEST(FusedStateSpaceTest, MambaFullyOverwritesSnapshotsWithoutTrailingSlots) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaNonuniform(1, 0, false, false);
  TestMambaNonuniform(8, 3, false, false);
  TestMambaNonuniform(8, -1, false, false);
}

TEST(FusedStateSpaceTest, MambaInvalidSelectorsProduceNaNWithoutStateReads) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8}) {
    TestMambaNonuniform(rows, -1);
    TestMambaNonuniform(rows, std::max(rows + 2, 4));
  }
}

TEST(FusedStateSpaceTest, MambaPrefillTerminalStateMatchesReference) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 2, 3, 8, 15, 16, 17, 33, 300}) {
    SCOPED_TRACE(rows);
    TestMambaNonuniform(rows, 2, false, true, 4, 2, 0.01f, MambaDtCase::Default, 3, false, true);
    TestMambaNonuniform(rows, 0, false, false, 4, 2, 0.01f, MambaDtCase::Default, 3, false, true);
  }
  TestMambaNonuniform(2048, 1, false, true, 2, 1, 0.01f, MambaDtCase::Default, 3, false, true);
}

TEST(FusedStateSpaceTest, MambaPrefillTimeStepEdgeCases) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (auto dt_case : {MambaDtCase::Extremes, MambaDtCase::ProjectionNaN, MambaDtCase::BiasNaN}) {
    TestMambaNonuniform(24, 0, false, true, 4, 2, 0.01f, dt_case, 3, false, true);
  }
}

TEST(FusedStateSpaceTest, MambaPrefillAliasingIntermediatesAndCapture) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int intermediates = 0; intermediates < 4; ++intermediates) {
    SCOPED_TRACE(intermediates);
    // Selector 0 reads and writes the same aliased slot.
    for (int selected : {0, 3}) {
      TestMambaNonuniform(40, selected, true, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates, false, true);
    }
    TestMambaNonuniform(40, 2, false, true, 4, 2, 0.01f, MambaDtCase::Default, intermediates, true, true);
  }
}

TEST(FusedStateSpaceTest, MambaPrefillInvalidSelectorsProduceNaN) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int selected : {-1, 4}) {
    TestMambaNonuniform(20, selected, true, true, 4, 2, 0.01f, MambaDtCase::Default, 3, false, true);
  }
}

// A bulk-prefill graph must reproduce eight-row verification exactly, so one Mamba2Prefill call
// is compared byte for byte with successive Mamba2Verify calls that continue from their last row.
TEST(FusedStateSpaceTest, MambaPrefillMatchesChainedEightRowVerification) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int heads = 8, groups = 2, slots = 8, rows = 56, selected = 5;
  constexpr int di = heads * 64, channels = di + groups * 256, stride = di + channels + heads;
  constexpr int state_stride = heads * 128 * 64;
  std::vector<BFloat16> proj(rows * stride), conv(slots * channels * 3), cw(channels * 4), cb(channels);
  std::vector<float> state(slots * state_stride), decay(heads), skip(heads), dt_bias(heads), gamma(di);
  for (int i = 0; i < rows * stride; ++i) proj[i] = BFloat16(std::sin(0.37f * static_cast<float>(i % 1013)) * 0.75f);
  for (int i = 0; i < slots * channels * 3; ++i) conv[i] = BFloat16(std::cos(0.11f * static_cast<float>(i % 389)) / 4.f);
  for (int i = 0; i < slots * state_stride; ++i) state[i] = std::sin(0.023f * static_cast<float>(i % 7919)) / 8.f;
  for (int i = 0; i < channels * 4; ++i) cw[i] = BFloat16(std::sin(0.7f * static_cast<float>(i % 101)) / 2.f);
  for (int j = 0; j < channels; ++j) cb[j] = BFloat16((j % 7 - 3) / 64.f);
  for (int h = 0; h < heads; ++h) {
    decay[h] = -(h + 1) / 8.f;
    skip[h] = (h + 1) / 8.f;
    dt_bias[h] = (h - 3) / 16.f;
  }
  for (int j = 0; j < di; ++j) gamma[j] = (j % 9 + 1) / 8.f;

  struct Fetched {
    std::vector<BFloat16> y, conv, xbc;
    std::vector<float> ssm, readout;
  };
  const auto run = [&](const char* op, int first_row, int count, int row,
                       const std::vector<BFloat16>& conv_in, const std::vector<float>& ssm_in) {
    OpTester tester(op, 1, kMSDomain);
    tester.AddAttribute("heads", static_cast<int64_t>(heads));
    tester.AddAttribute("n_groups", static_cast<int64_t>(groups));
    tester.AddAttribute("time_step_min", 0.01f);
    tester.AddInput<BFloat16>("projection", {count, stride},
                              std::vector<BFloat16>(proj.begin() + first_row * stride,
                                                    proj.begin() + (first_row + count) * stride));
    tester.AddInput<BFloat16>("conv_state", {slots, channels, 3}, conv_in);
    tester.AddInput<float>("ssm_state", {slots, heads, 128, 64}, ssm_in);
    tester.AddInput<BFloat16>("conv_weight", {channels, 4}, cw);
    tester.AddInput<BFloat16>("conv_bias", {channels}, cb);
    tester.AddInput<float>("dt_bias", {heads}, dt_bias);
    tester.AddInput<float>("A_negative", {heads}, decay);
    tester.AddInput<float>("D", {heads}, skip);
    tester.AddInput<float>("norm_weight", {di}, gamma);
    tester.AddInput<int32_t>("state_row", {1}, {row});
    tester.AddOutput<BFloat16>("Y", {count, di}, std::vector<BFloat16>(count * di));
    tester.AddOutput<BFloat16>("conv_out", {slots, channels, 3}, conv_in);
    tester.AddOutput<float>("ssm_out", {slots, heads, 128, 64}, ssm_in);
    tester.AddOutput<BFloat16>("xbc", {count, channels}, std::vector<BFloat16>(count * channels));
    tester.AddOutput<float>("readout", {count, di}, std::vector<float>(count * di));
    Fetched fetched;
    tester.SetCustomOutputVerifier([&fetched](const std::vector<OrtValue>& outputs, const std::string&) {
      ASSERT_EQ(outputs.size(), 5U);
      const auto copy = [](const OrtValue& value, auto& destination) {
        using Element = typename std::decay_t<decltype(destination)>::value_type;
        const auto data = value.Get<Tensor>().DataAsSpan<Element>();
        destination.assign(data.begin(), data.end());
      };
      copy(outputs[0], fetched.y);
      copy(outputs[1], fetched.conv);
      copy(outputs[2], fetched.ssm);
      copy(outputs[3], fetched.xbc);
      copy(outputs[4], fetched.readout);
    });
    RunCuda(tester);
    return fetched;
  };

  Fetched chained;
  auto conv_state = conv;
  auto ssm_state = state;
  int row = selected;
  for (int first = 0; first < rows; first += 8) {
    auto step = run("Mamba2Verify", first, 8, row, conv_state, ssm_state);
    chained.y.insert(chained.y.end(), step.y.begin(), step.y.end());
    chained.xbc.insert(chained.xbc.end(), step.xbc.begin(), step.xbc.end());
    chained.readout.insert(chained.readout.end(), step.readout.begin(), step.readout.end());
    conv_state = std::move(step.conv);
    ssm_state = std::move(step.ssm);
    row = 7;
  }
  const auto bulk = run("Mamba2Prefill", 0, rows, selected, conv, state);
  const auto same_bytes = [](const auto& a, const auto& b, size_t offset_a, size_t offset_b, size_t count) {
    return std::memcmp(a.data() + offset_a, b.data() + offset_b, count * sizeof(a[0])) == 0;
  };
  EXPECT_TRUE(same_bytes(chained.y, bulk.y, 0, 0, chained.y.size()));
  EXPECT_TRUE(same_bytes(chained.xbc, bulk.xbc, 0, 0, chained.xbc.size()));
  EXPECT_TRUE(same_bytes(chained.readout, bulk.readout, 0, 0, chained.readout.size()));
  // The verifier's terminal snapshot is row 7; Mamba2Prefill writes it to slot zero.
  EXPECT_TRUE(same_bytes(conv_state, bulk.conv, size_t{7} * channels * 3, 0, size_t{channels} * 3));
  EXPECT_TRUE(same_bytes(ssm_state, bulk.ssm, size_t{7} * state_stride, 0, size_t{state_stride}));
  // Slots other than zero keep their incoming values.
  EXPECT_TRUE(same_bytes(conv, bulk.conv, size_t{channels} * 3, size_t{channels} * 3, size_t{slots - 1} * channels * 3));
  EXPECT_TRUE(same_bytes(state, bulk.ssm, size_t{state_stride}, size_t{state_stride}, size_t{slots - 1} * state_stride));
}

#endif
}  // namespace onnxruntime::test
