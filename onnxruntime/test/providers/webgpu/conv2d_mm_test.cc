// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/nn/conv2d_mm.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct Conv2dMMKeyInfo {
  std::string key;
  std::string cache_hint;
  uint32_t workgroup_size_y;
};

Conv2dMMKeyInfo BuildConv2dMMKeyInfo(int64_t output_height,
                                     int64_t output_width,
                                     const webgpu::PackedTileCaps& caps,
                                     int64_t in_channels = 8,
                                     int64_t out_channels = 16) {
  const webgpu::Activation activation;  // ActivationKind::None
  AllocatorPtr allocator = std::make_shared<CPUAllocator>();

  const TensorShape input_shape({1, output_height, output_width, in_channels});
  const TensorShape kernel_shape({1, 1, in_channels, out_channels});
  const TensorShape output_shape({1, output_height, output_width, out_channels});

  Tensor input(DataTypeImpl::GetType<float>(), input_shape, allocator);
  Tensor kernel(DataTypeImpl::GetType<float>(), kernel_shape, allocator);
  Tensor output(DataTypeImpl::GetType<float>(), output_shape, allocator);

  const std::vector<const Tensor*> inputs{&input, &kernel};
  const std::vector<TensorShape> input_output_shapes{input_shape, kernel_shape, output_shape};
  const std::vector<uint32_t> pads{0, 0, 0, 0};
  const std::vector<uint32_t> strides{1, 1};
  const std::vector<uint32_t> dilations{1, 1};

  // Matches conv.cc: channels-last maps rows to the output spatial extent, columns to channels.
  const auto dim_a_outer = static_cast<uint32_t>(output_height * output_width);
  const auto dim_b_outer = static_cast<uint32_t>(out_channels);
  const auto dim_inner = static_cast<uint32_t>(in_channels);

  webgpu::Conv2dMMProgram program = webgpu::CreateConv2dMMProgram(
      activation, inputs, pads, strides, dilations, &output,
      dim_a_outer, dim_b_outer, dim_inner, /*is_channels_last=*/true,
      caps, input_output_shapes);

  std::vector<uint32_t> inputs_segments(program.Inputs().size(), 1u);
  std::vector<uint32_t> outputs_segments(program.Outputs().size(), 1u);
  const bool is_1d_dispatch = program.DispatchGroupSizeY() == 1 && program.DispatchGroupSizeZ() == 1;

  // On a non-const ProgramBase the variadic CacheHint() setter wins over the getter and
  // clears the hint, so read it through a const reference.
  const webgpu::ProgramBase& const_program = program;

  return Conv2dMMKeyInfo{
      webgpu::CalculateProgramCacheKey(program, inputs_segments, outputs_segments, is_1d_dispatch),
      const_program.CacheHint(),
      program.WorkgroupSizeY()};
}

// dim_a_outer = 32 * 32 = 1024, above the > 8 tuning gate.
constexpr int64_t kTunedOutputHeight = 32;
constexpr int64_t kTunedOutputWidth = 32;

// NVIDIA reports a 32-lane subgroup (the warp size); desktop adapters report 1024 for both
// compute workgroup limits.
constexpr webgpu::PackedTileCaps kNvidiaCaps{/*subgroup_size=*/32, /*max_workgroup_size_y=*/1024,
                                             /*max_invocations_per_workgroup=*/1024, /*is_nvidia=*/true};
constexpr webgpu::PackedTileCaps kNonNvidiaCaps{/*subgroup_size=*/32, /*max_workgroup_size_y=*/1024,
                                                /*max_invocations_per_workgroup=*/1024, /*is_nvidia=*/false};

}  // namespace

// Both paths reach tile_a_outer 32, as 32*1 and 8*4, so they share a cache hint even though
// they bake different elements_per_thread into their WGSL.
TEST(Conv2dMMCacheKeyTest, TunedAndUntunedShareAHintButNotAKey) {
  const auto tuned = BuildConv2dMMKeyInfo(kTunedOutputHeight, kTunedOutputWidth, kNvidiaCaps);
  const auto untuned = BuildConv2dMMKeyInfo(kTunedOutputHeight, kTunedOutputWidth, kNonNvidiaCaps);

  ASSERT_EQ(tuned.cache_hint, untuned.cache_hint);
  EXPECT_EQ(tuned.workgroup_size_y, 32u);
  EXPECT_EQ(untuned.workgroup_size_y, 8u);

  // program_cache_key.cc writes the workgroup size as "<x>,<y>,<z>".
  EXPECT_NE(tuned.key.find("8,32,1"), std::string::npos);
  EXPECT_NE(untuned.key.find("8,8,1"), std::string::npos);
  EXPECT_NE(tuned.key, untuned.key);
}

// dim_a_outer = 8 does not clear the > 8 gate, so both paths emit the same WGSL.
TEST(Conv2dMMCacheKeyTest, TuningIsInertBelowTheGate) {
  const auto tuned = BuildConv2dMMKeyInfo(2, 4, kNvidiaCaps);
  const auto untuned = BuildConv2dMMKeyInfo(2, 4, kNonNvidiaCaps);

  EXPECT_EQ(tuned.workgroup_size_y, 8u);
  EXPECT_EQ(untuned.workgroup_size_y, 8u);
  EXPECT_EQ(tuned.key, untuned.key);
}

// The derived workgroup must respect the device's reported limits, not just the vendor.
TEST(Conv2dMMCacheKeyTest, WorkgroupSizeYLimitClampsTheDerivedConfig) {
  webgpu::PackedTileCaps capped = kNvidiaCaps;
  capped.max_workgroup_size_y = 16;

  const auto clamped = BuildConv2dMMKeyInfo(kTunedOutputHeight, kTunedOutputWidth, capped);

  EXPECT_EQ(clamped.workgroup_size_y, 16u);
  EXPECT_NE(clamped.key.find("8,16,1"), std::string::npos);
}

// Runs the tiled Conv2dMM shader: the subgroup-derived workgroup on NVIDIA adapters, the
// default 8x4 elsewhere.
TEST(Conv2dMMTest, ChannelsLastVec4ConvAboveTheTilingGate) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t kInChannels = 4;   // divisible by 4, so the shader packs vec4
  constexpr int64_t kOutChannels = 4;  // divisible by 4, likewise
  constexpr int64_t kInSize = 8;
  constexpr int64_t kKernel = 3;
  constexpr int64_t kOutSize = kInSize - kKernel + 1;  // 6: dim_a_outer 36, a full tile plus a partial one

  const std::vector<int64_t> x_shape{1, kInChannels, kInSize, kInSize};
  const std::vector<int64_t> w_shape{kOutChannels, kInChannels, kKernel, kKernel};
  const std::vector<int64_t> y_shape{1, kOutChannels, kOutSize, kOutSize};

  // Quarters and halves stay exact in fp32, so tiling order cannot change the result.
  std::vector<float> x_vals(static_cast<size_t>(kInChannels * kInSize * kInSize));
  for (size_t i = 0; i < x_vals.size(); ++i) {
    x_vals[i] = static_cast<float>(i % 7) * 0.25f - 0.75f;
  }
  std::vector<float> w_vals(static_cast<size_t>(kOutChannels * kInChannels * kKernel * kKernel));
  for (size_t i = 0; i < w_vals.size(); ++i) {
    w_vals[i] = static_cast<float>(i % 5) * 0.5f - 1.0f;
  }

  std::vector<float> expected(static_cast<size_t>(kOutChannels * kOutSize * kOutSize));
  for (int64_t m = 0; m < kOutChannels; ++m) {
    for (int64_t oh = 0; oh < kOutSize; ++oh) {
      for (int64_t ow = 0; ow < kOutSize; ++ow) {
        float acc = 0.0f;
        for (int64_t c = 0; c < kInChannels; ++c) {
          for (int64_t kh = 0; kh < kKernel; ++kh) {
            for (int64_t kw = 0; kw < kKernel; ++kw) {
              acc += x_vals[static_cast<size_t>((c * kInSize + oh + kh) * kInSize + ow + kw)] *
                     w_vals[static_cast<size_t>(((m * kInChannels + c) * kKernel + kh) * kKernel + kw)];
            }
          }
        }
        expected[static_cast<size_t>((m * kOutSize + oh) * kOutSize + ow)] = acc;
      }
    }
  }

  OpTester test("Conv", 11);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kKernel, kKernel});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", x_shape, x_vals);
  test.AddInput<float>("W", w_shape, w_vals);
  test.AddOutput<float>("Y", y_shape, expected);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
