// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "core/diffusion/ddim.h"
#include "core/framework/tensor.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime::test {
namespace {

class DdimSchedulerTest : public testing::Test {
 protected:
  const AllocatorPtr allocator_ = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  DdimOptions Options(int64_t steps = 3) {
    return stateful::MakeDdimOptions(steps);
  }

  template <typename T>
  OrtValue Value(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
    OrtValue value;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape(shape), allocator_, value);
    EXPECT_EQ(value.Get<Tensor>().Shape().Size(), static_cast<int64_t>(data.size()));
    std::copy(data.begin(), data.end(), value.GetMutable<Tensor>()->MutableData<T>());
    return value;
  }

  std::shared_ptr<const DdimScheduler> Scheduler(DdimOptions options) {
    std::shared_ptr<const DdimScheduler> scheduler;
    ORT_THROW_IF_ERROR(DdimScheduler::Create(std::move(options), scheduler));
    return scheduler;
  }
};

TEST_F(DdimSchedulerTest, PinnedCoefficientsAndExplicitLeadingSchedule) {
  auto configuration = Options();
  const auto scheduler = Scheduler(configuration);
  configuration.profile = "changed";
  configuration.num_inference_steps = 1;
  EXPECT_EQ(scheduler->Options().profile, "sd15_ddim_v1");
  EXPECT_EQ(scheduler->Options().num_inference_steps, 3);
  for (const auto& [index, expected] : std::array<std::pair<size_t, float>, 7>{
           {{0, 0.9991499782f}, {1, 0.9982960224f}, {334, 0.5332823396f}, {501, 0.2749988437f}, {667, 0.1043594778f}, {981, 0.0057754959f}, {999, 0.0046600951f}}}) {
    EXPECT_NEAR(scheduler->AlphaProducts()[index], expected, 1e-5f + 1e-5f * std::abs(expected));
  }
  for (const auto count : {1, 2, 50, 999}) {
    const auto instance = Scheduler(Options(count));
    int64_t timestep = -1;
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_STATUS_OK(instance->Timestep(i, timestep));
      EXPECT_EQ(timestep, (count - 1 - i) * (1000 / count) + 1);
      EXPECT_GE(timestep, 0);
      EXPECT_LE(timestep, 999);
    }
    EXPECT_FALSE(instance->Timestep(-1, timestep).IsOK());
    EXPECT_FALSE(instance->Timestep(count, timestep).IsOK());
    EXPECT_EQ(timestep, 1);
  }
}

TEST_F(DdimSchedulerTest, PinnedReferenceTrajectoriesForBothGuidanceConventions) {
  const std::array<int64_t, 4> counts{1, 2, 50, 999};
  // Generated with Diffusers v0.35.1 DDIMScheduler, CPU Torch 2.8, common supplied latents.
  const std::array<std::array<float, 3>, 4> expected{{{0.0626358911f, -0.3119553328f, 0.9366819859f},
                                                      {-8.0220680237f, -8.6617479324f, -6.5294814110f},
                                                      {-50.6662826538f, -53.9910202026f, -42.9084472656f},
                                                      {-55.1858596802f, -58.8017349243f, -46.7493515015f}}};
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (size_t case_index = 0; case_index < counts.size(); ++case_index) {
      auto options = Options(counts[case_index]);
      options.guidance_mode = mode;
      const auto scheduler = Scheduler(options);
      OrtValue latents = Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f});
      for (int64_t step = 0; step < counts[case_index]; ++step) {
        int64_t timestep = 0;
        ASSERT_STATUS_OK(scheduler->Timestep(step, timestep));
        const auto input = latents.Get<Tensor>().DataAsSpan<float>();
        OrtValue uncond = Value<float>({1, 1, 1, 3}, {0.f, 0.f, 0.f});
        OrtValue cond = Value<float>({1, 1, 1, 3}, {0.f, 0.f, 0.f});
        for (size_t i = 0; i < input.size(); ++i) {
          const float common = input[i] * 0.125f + static_cast<float>(timestep) * 0.0001f;
          uncond.GetMutable<Tensor>()->MutableData<float>()[i] = common - 0.5f;
          cond.GetMutable<Tensor>()->MutableData<float>()[i] = common + 0.25f;
        }
        InlinedVector<OrtValue> predictions;
        if (mode == DdimGuidanceMode::Batched) {
          auto joined = Value<float>({2, 1, 1, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
          auto* data = joined.GetMutable<Tensor>()->MutableData<float>();
          std::copy_n(uncond.Get<Tensor>().Data<float>(), 3, data);
          std::copy_n(cond.Get<Tensor>().Data<float>(), 3, data + 3);
          predictions.push_back(std::move(joined));
        } else {
          predictions = {uncond, cond};
        }
        OrtValue next;
        ASSERT_STATUS_OK(scheduler->Update(latents, predictions, step, allocator_, next));
        latents = std::move(next);
      }
      for (size_t i = 0; i < 3; ++i) {
        const float reference = expected[case_index][i];
        EXPECT_NEAR(latents.Get<Tensor>().Data<float>()[i], reference, 1e-4f + 1e-4f * std::abs(reference));
      }
    }
  }
}

TEST_F(DdimSchedulerTest, SeedReplayHighBitsOddCountsAndPrivateSuppliedCopy) {
  const auto scheduler = Scheduler(Options());
  const std::array<std::array<float, 3>, 2> reference{{{1.912804484f, -0.09447956085f, -2.07940793f},
                                                       {-0.5412746668f, -2.648202419f, -2.542910576f}}};
  size_t case_index = 0;
  for (const uint64_t seed : {uint64_t{0}, std::numeric_limits<uint64_t>::max()}) {
    auto input = Value<uint64_t>({}, {seed});
    OrtValue first, second;
    ASSERT_STATUS_OK(scheduler->Initialize(input, allocator_, first));
    ASSERT_STATUS_OK(scheduler->Initialize(input, allocator_, second));
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_TRUE(std::isfinite(first.Get<Tensor>().Data<float>()[i]));
      EXPECT_EQ(first.Get<Tensor>().Data<float>()[i], second.Get<Tensor>().Data<float>()[i]);
      EXPECT_NEAR(first.Get<Tensor>().Data<float>()[i], reference[case_index][i], 1e-6f);
    }
    EXPECT_NE(first.Get<Tensor>().DataRaw(), second.Get<Tensor>().DataRaw());
    ++case_index;
  }
  auto input = Value<float>({1, 1, 1, 3}, {1.f, 2.f, 3.f});
  OrtValue copy;
  ASSERT_STATUS_OK(scheduler->Initialize(input, allocator_, copy));
  input.GetMutable<Tensor>()->MutableData<float>()[0] = 9.f;
  EXPECT_EQ(copy.Get<Tensor>().Data<float>()[0], 1.f);
}

TEST_F(DdimSchedulerTest, GuidanceBoundariesMatchAnalyticalUpdate) {
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (const float scale : {0.f, 1.f, 7.5f}) {
      auto options = Options(2);
      options.guidance_mode = mode;
      options.guidance_scale = scale;
      const auto scheduler = Scheduler(options);
      const auto latents = Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f});
      InlinedVector<OrtValue> predictions;
      if (mode == DdimGuidanceMode::Batched) {
        predictions = {Value<float>({2, 1, 1, 3}, {0.f, 0.f, 0.f, 1.f, 1.f, 1.f})};
      } else {
        predictions = {Value<float>({1, 1, 1, 3}, {0.f, 0.f, 0.f}),
                       Value<float>({1, 1, 1, 3}, {1.f, 1.f, 1.f})};
      }
      OrtValue result;
      ASSERT_STATUS_OK(scheduler->Update(latents, predictions, 0, allocator_, result));
      const double alpha = 0.27499884366989136;
      const double previous_alpha = 0.9982960224151611;
      for (size_t i = 0; i < 3; ++i) {
        const double initial = latents.Get<Tensor>().Data<float>()[i];
        const double expected = std::sqrt(previous_alpha / alpha) * (initial - std::sqrt(1. - alpha) * scale) +
                                std::sqrt(1. - previous_alpha) * scale;
        EXPECT_NEAR(result.Get<Tensor>().Data<float>()[i], expected, 1e-5 + 1e-5 * std::abs(expected));
      }
    }
  }
}

TEST_F(DdimSchedulerTest, ConditioningOrderBatchAndAllTimestepEncodings) {
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (const auto format : {DdimTimestepFormat::Int64Scalar, DdimTimestepFormat::Int64Vector,
                              DdimTimestepFormat::Float32Scalar, DdimTimestepFormat::Float32Vector}) {
      auto options = Options();
      options.guidance_mode = mode;
      options.timestep_format = format;
      options.latent_shape = {2, 1, 1, 1};
      const auto scheduler = Scheduler(options);
      const auto latents = Value<float>({2, 1, 1, 1}, {1.f, 2.f});
      const auto positive = Value<float>({2, 1}, {3.f, 4.f});
      const auto negative = Value<float>({2, 1}, {-3.f, -4.f});
      for (const bool conditional : {false, true}) {
        DdimModelInputs inputs;
        ASSERT_STATUS_OK(scheduler->Prepare(latents, positive, negative, 0, conditional, allocator_, inputs));
        const bool batched = mode == DdimGuidanceMode::Batched;
        EXPECT_EQ(inputs.latents.Get<Tensor>().Shape()[0], batched ? 4 : 2);
        const float first = inputs.conditioning.Get<Tensor>().Data<float>()[0];
        EXPECT_EQ(first, batched || !conditional ? -3.f : 3.f);
        if (batched) {
          EXPECT_EQ(inputs.conditioning.Get<Tensor>().Data<float>()[2], 3.f);
        }
        const auto& time = inputs.timestep.Get<Tensor>();
        const bool vector = format == DdimTimestepFormat::Int64Vector || format == DdimTimestepFormat::Float32Vector;
        EXPECT_EQ(time.Shape().NumDimensions(), vector ? 1u : 0u);
        if (time.IsDataType<float>()) {
          EXPECT_EQ(*time.Data<float>(), 667.f);
        } else {
          EXPECT_EQ(*time.Data<int64_t>(), 667);
        }
      }
    }
  }
}

TEST_F(DdimSchedulerTest, InvalidConfigurationIsTransactional) {
  const auto original = Scheduler(Options());
  auto output = original;
  auto reject = [&](DdimOptions options) {
    EXPECT_FALSE(DdimScheduler::Create(std::move(options), output).IsOK());
    EXPECT_EQ(output, original);
  };
  auto options = Options();
  options.profile = "other";
  reject(options);
  for (const int64_t steps : {-1, 0, 1000}) {
    options = Options(steps);
    reject(options);
  }
  for (const float scale : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    options = Options();
    options.guidance_scale = scale;
    reject(options);
  }
  options = Options();
  options.guidance_mode = static_cast<DdimGuidanceMode>(-1);
  reject(options);
  options = Options();
  options.timestep_format = static_cast<DdimTimestepFormat>(-1);
  reject(options);
  for (const int64_t dimension : {int64_t{0}, int64_t{-1}, std::numeric_limits<int64_t>::max()}) {
    options = Options();
    options.latent_shape[0] = dimension;
    reject(options);
  }
  options = Options();
  options.latent_input.clear();
  reject(options);
  options.latent_input = options.timestep_input;
  reject(options);
  options.latent_input = std::string("a\0b", 3);
  reject(options);
}

TEST_F(DdimSchedulerTest, InvalidDataAndOverflowDoNotPublishResults) {
  const auto scheduler = Scheduler(Options());
  const auto valid = Value<float>({1, 1, 1, 3}, {1.f, 2.f, 3.f});
  const auto conditioning = Value<float>({1, 1}, {1.f});
  OrtValue output = valid;
  for (const auto& invalid : {OrtValue{}, Value<int64_t>({}, {1}), Value<uint64_t>({1}, {1}),
                              Value<float>({1}, {1.f}),
                              Value<float>({1, 1, 1, 3}, {1.f, 2.f, std::numeric_limits<float>::infinity()})}) {
    EXPECT_FALSE(scheduler->Initialize(invalid, allocator_, output).IsOK());
    EXPECT_EQ(output.Get<Tensor>().DataRaw(), valid.Get<Tensor>().DataRaw());
  }
  DdimModelInputs feeds;
  EXPECT_FALSE(scheduler->Prepare(valid, conditioning, Value<float>({1, 2}, {1.f, 2.f}),
                                  0, false, allocator_, feeds)
                   .IsOK());
  EXPECT_FALSE(scheduler->Prepare(valid, Value<float>({}, {1.f}), conditioning,
                                  0, false, allocator_, feeds)
                   .IsOK());
  EXPECT_FALSE(scheduler->Prepare(valid, conditioning, conditioning, -1, false, allocator_, feeds).IsOK());
  const InlinedVector<OrtValue> none;
  EXPECT_FALSE(scheduler->Update(valid, none, 0, allocator_, output).IsOK());
  const InlinedVector<OrtValue> wrong_shape{conditioning};
  EXPECT_FALSE(scheduler->Update(valid, wrong_shape, 0, allocator_, output).IsOK());
  auto options = Options();
  options.guidance_scale = std::numeric_limits<float>::max();
  const auto overflow = Scheduler(options);
  const InlinedVector<OrtValue> prediction{
      Value<float>({2, 1, 1, 3}, {-1.f, -1.f, -1.f, 1.f, 1.f, 1.f})};
  EXPECT_FALSE(overflow->Update(valid, prediction, 0, allocator_, output).IsOK());
  EXPECT_EQ(output.Get<Tensor>().DataRaw(), valid.Get<Tensor>().DataRaw());
}

}  // namespace
}  // namespace onnxruntime::test
