// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/attention_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

constexpr int kBatchSize = 1;
constexpr int kSequenceLength = 129;
constexpr int kNumHeads = 2;
constexpr int kHeadSize = 128;
constexpr int kCapacity = 208;

InlinedVector<MLFloat16> MakePrefillData(size_t size, int seed) {
  InlinedVector<MLFloat16> data(size);
  for (size_t i = 0; i < size; ++i) {
    const int value = static_cast<int>((i * seed + i / kHeadSize) % 29) - 14;
    data[i] = MLFloat16(static_cast<float>(value) * 0.0625f);
  }
  return data;
}

InlinedVector<MLFloat16> PrefillReference(gsl::span<const MLFloat16> query,
                                          gsl::span<const MLFloat16> key_cache,
                                          gsl::span<const MLFloat16> value_cache,
                                          const std::array<int, kBatchSize>& past_lengths,
                                          int window) {
  InlinedVector<MLFloat16> expected(query.size());
  for (int batch = 0; batch < kBatchSize; ++batch) {
    for (int row = 0; row < kSequenceLength; ++row) {
      const int end = past_lengths[batch] + row + 1;
      const int start = window > 0 ? std::max(0, end - window) : 0;
      for (int head = 0; head < kNumHeads; ++head) {
        const int q_offset = ((batch * kSequenceLength + row) * kNumHeads + head) * kHeadSize;
        InlinedVector<double> scores(end);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = start; key < end; ++key) {
          double score = 0.0;
          for (int dim = 0; dim < kHeadSize; ++dim) {
            score += static_cast<double>(query[q_offset + dim].ToFloat()) *
                     key_cache[(batch * kCapacity + key) * kHeadSize + dim].ToFloat();
          }
          scores[key] = score / std::sqrt(static_cast<double>(kHeadSize));
          maximum = std::max(maximum, scores[key]);
        }
        double denominator = 0.0;
        for (int key = start; key < end; ++key) {
          scores[key] = std::exp(scores[key] - maximum);
          denominator += scores[key];
        }
        for (int dim = 0; dim < kHeadSize; ++dim) {
          double sum = 0.0;
          for (int key = start; key < end; ++key) {
            sum += scores[key] * value_cache[(batch * kCapacity + key) * kHeadSize + dim].ToFloat();
          }
          expected[q_offset + dim] = MLFloat16(static_cast<float>(sum / denominator));
        }
      }
    }
  }
  return expected;
}

class CooperativePrefillTester final : public OpTester {
 public:
  CooperativePrefillTester() : OpTester("GroupQueryAttention", 1, kMSDomain) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CooperativePrefillTester);

  void RunPrefill(bool capture, int window) {
    SetTestFunctionCalled();
    const auto query = MakePrefillData(kBatchSize * kSequenceLength * kNumHeads * kHeadSize, 7);
    const auto key = MakePrefillData(kBatchSize * kSequenceLength * kHeadSize, 11);
    const auto value = MakePrefillData(key.size(), 13);
    const auto initial_key = MakePrefillData(kBatchSize * kCapacity * kHeadSize, 17);
    const auto initial_value = MakePrefillData(initial_key.size(), 19);
    const TensorShape output_shape{kBatchSize, kSequenceLength, kNumHeads * kHeadSize};
    const TensorShape cache_shape{kBatchSize, 1, kCapacity, kHeadSize};
    AddAttribute<int64_t>("num_heads", kNumHeads);
    AddAttribute<int64_t>("kv_num_heads", 1);
    AddAttribute<int64_t>("local_window_size", window);
    AddInput<MLFloat16>("query", {kBatchSize, kSequenceLength, kNumHeads * kHeadSize}, query.data(), query.size());
    AddInput<MLFloat16>("key", {kBatchSize, kSequenceLength, kHeadSize}, key.data(), key.size());
    AddInput<MLFloat16>("value", {kBatchSize, kSequenceLength, kHeadSize}, value.data(), value.size());
    AddInput<MLFloat16>("past_key", {kBatchSize, 1, kCapacity, kHeadSize}, initial_key.data(), initial_key.size());
    AddInput<MLFloat16>("past_value", {kBatchSize, 1, kCapacity, kHeadSize}, initial_value.data(), initial_value.size());
    AddInput<int32_t>("seqlens_k", {kBatchSize}, {145});
    AddInput<int32_t>("total_sequence_length", {1}, {146});
    AddOutput<MLFloat16>("output", {kBatchSize, kSequenceLength, kNumHeads * kHeadSize}, query.data(), query.size());
    AddOutput<MLFloat16>("present_key", {kBatchSize, 1, kCapacity, kHeadSize}, initial_key.data(), initial_key.size());
    AddOutput<MLFloat16>("present_value", {kBatchSize, 1, kCapacity, kHeadSize}, initial_value.data(), initial_value.size());
    auto& model = BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve(GetRunContext().resolve_options));
    const auto serialized = model.ToProto().SerializeAsString();

    ConfigOptions provider_options;
    if (capture) {
      ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                                       webgpu::options::kEnableGraphCapture_ON));
    }
    auto provider = WebGpuExecutionProviderWithOptions(provider_options);
    ASSERT_NE(provider, nullptr);
    auto* ep = provider.get();
    auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    InferenceSession session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const auto allocators = ep->CreatePreferredAllocators();
    ASSERT_FALSE(allocators.empty());
    const auto allocator = allocators.back();
    ASSERT_NE(allocator->Info().device.Type(), OrtDevice::CPU);
    std::unordered_map<std::string, OrtValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);
    std::unique_ptr<IOBinding> binding;
    ASSERT_STATUS_OK(session.NewIOBinding(&binding));
    for (auto& [name, input] : feeds) {
      if (capture || name != "total_sequence_length") {
        OrtValue gpu_input;
        const auto& source = input.Get<Tensor>();
        Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, gpu_input);
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(source, *gpu_input.GetMutable<Tensor>()));
        input = std::move(gpu_input);
      }
      ASSERT_STATUS_OK(binding->BindInput(name, input));
    }
    OrtValue output;
    Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), output_shape, allocator, output);
    ASSERT_STATUS_OK(binding->BindOutput("output", output));
    ASSERT_STATUS_OK(binding->BindOutput("present_key", feeds.at("past_key")));
    ASSERT_STATUS_OK(binding->BindOutput("present_value", feeds.at("past_value")));
    ASSERT_STATUS_OK(binding->SynchronizeInputs());

    auto upload = [&](const char* name, const void* data) {
      auto& target = *feeds.at(name).GetMutable<Tensor>();
      const Tensor source(target.DataType(), target.Shape(), const_cast<void*>(data),
                          CPUAllocator::DefaultInstance()->Info());
      ORT_THROW_IF_ERROR(session.GetDataTransferManager().CopyTensor(source, target));
    };
    const std::array<std::array<int, kBatchSize>, 3> past_lengths{{{17}, {41}, {3}}};
    for (size_t run = 0; run < (capture ? past_lengths.size() : 1u); ++run) {
      SCOPED_TRACE(testing::Message() << "capture=" << capture << " window=" << window << " run=" << run);
      const auto& past = past_lengths[run];
      std::array<int32_t, kBatchSize> lengths{};
      for (int batch = 0; batch < kBatchSize; ++batch) {
        lengths[batch] = past[batch] + kSequenceLength - 1;
      }
      const int32_t total = *std::max_element(lengths.begin(), lengths.end()) + 1;
      upload("seqlens_k", lengths.data());
      if (capture) {
        // A device-resident total leaves the host total-length parameter at zero.
        ASSERT_NE(feeds.at("total_sequence_length").Get<Tensor>().Location().device.Type(), OrtDevice::CPU);
        upload("total_sequence_length", &total);
      }
      // Start each replay with known past caches; update the same bound buffers.
      upload("past_key", initial_key.data());
      upload("past_value", initial_value.data());
      Tensor cpu_output(DataTypeImpl::GetType<MLFloat16>(), output_shape, CPUAllocator::DefaultInstance());
      std::fill_n(cpu_output.MutableData<MLFloat16>(), query.size(), MLFloat16(-99.0f));
      ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_output, *output.GetMutable<Tensor>()));
      profiling::Events events;
      context.SetProfilingStartTime(std::chrono::high_resolution_clock::now());
      context.StartProfiling();
      {
        auto stop_profiling = gsl::finally([&] { context.CollectProfilingData(events); });
        if (run == 0) {
          ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
        } else {
          ASSERT_TRUE(ep->IsGraphCaptured(0));
          ASSERT_STATUS_OK(ep->ReplayGraph(0));
        }
        ASSERT_STATUS_OK(binding->SynchronizeOutputs());
      }
      ExpectCooperativeDispatch(events);
      ASSERT_FALSE(testing::Test::HasFailure());
      if (capture) {
        ASSERT_TRUE(ep->IsGraphCaptured(0));
      }
      auto expected_key = initial_key;
      auto expected_value = initial_value;
      for (int batch = 0; batch < kBatchSize; ++batch) {
        const int source = batch * kSequenceLength * kHeadSize;
        const int target = (batch * kCapacity + past[batch]) * kHeadSize;
        std::copy_n(key.begin() + source, kSequenceLength * kHeadSize, expected_key.begin() + target);
        std::copy_n(value.begin() + source, kSequenceLength * kHeadSize, expected_value.begin() + target);
      }
      const auto expected = PrefillReference(query, expected_key, expected_value, past, window);
      ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(output.Get<Tensor>(), cpu_output));
      for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(cpu_output.Data<MLFloat16>()[i].ToFloat(), expected[i].ToFloat(), 0.001f) << "index=" << i;
      }
      for (const auto& [name, expected_cache] : {std::pair{"past_key", &expected_key},
                                                 std::pair{"past_value", &expected_value}}) {
        Tensor cache(DataTypeImpl::GetType<MLFloat16>(), cache_shape, CPUAllocator::DefaultInstance());
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(feeds.at(name).Get<Tensor>(), cache));
        ASSERT_TRUE(std::equal(expected_cache->begin(), expected_cache->end(), cache.Data<MLFloat16>())) << name;
      }
    }
  }
};

void RunCooperativePrefillTest(bool capture) {
  if (!SupportsQualcommCooperativePrefill()) {
    GTEST_SKIP() << "Requires Qualcomm FP16 and 128-lane subgroup control";
  }
  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::TimestampQuery)) {
    GTEST_SKIP() << "Dispatch observation requires WebGPU timestamp queries";
  }
  for (int window : {-1, 17}) {
    CooperativePrefillTester test;
    test.RunPrefill(capture, window);
  }
}

}  // namespace

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_Dispatch) {
  RunCooperativePrefillTest(false);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_GraphCaptureReplay) {
  RunCooperativePrefillTest(true);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_SmallQueryScale) {
  if (!SupportsCooperativeDispatchObservation()) {
    GTEST_SKIP() << "Requires Qualcomm cooperative prefill and timestamp queries";
  }
  constexpr int sequence_length = 129;
  constexpr int head_size = 128;
  constexpr float query_value = 0x1p-14f;
  constexpr float key_value = 65504.0f;
  constexpr float scale = 0x1p-13f;
  const InlinedVector<MLFloat16> query(sequence_length * head_size, MLFloat16(query_value));
  InlinedVector<MLFloat16> key(query.size());
  InlinedVector<MLFloat16> value(query.size());
  for (int row = 0; row < sequence_length; ++row) {
    const float sign = row % 2 == 0 ? 1.0f : -1.0f;
    std::fill_n(key.begin() + row * head_size, head_size, MLFloat16(sign * key_value));
    std::fill_n(value.begin() + row * head_size, head_size, MLFloat16(sign));
  }
  for (float softcap : {0.0f, 0.05f}) {
    SCOPED_TRACE(softcap);
    // Scaling Q in FP16 would round 2^-27 to zero and erase these scores.
    double score = static_cast<double>(query_value) * key_value * head_size * scale;
    if (softcap > 0.0f) {
      score = std::tanh(score / softcap) * softcap;
    }
    InlinedVector<MLFloat16> expected(query.size());
    for (int row = 0; row < sequence_length; ++row) {
      const double positive = (row / 2 + 1) * std::exp(score);
      const double negative = ((row + 1) / 2) * std::exp(-score);
      const auto result = MLFloat16(static_cast<float>((positive - negative) / (positive + negative)));
      std::fill_n(expected.begin() + row * head_size, head_size, result);
    }
    OpTester test("GroupQueryAttention", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", 1);
    test.AddAttribute<int64_t>("kv_num_heads", 1);
    test.AddAttribute<float>("scale", scale);
    test.AddAttribute<float>("softcap", softcap);
    test.AddInput<MLFloat16>("query", {1, sequence_length, head_size}, query.data(), query.size());
    test.AddInput<MLFloat16>("key", {1, sequence_length, head_size}, key.data(), key.size());
    test.AddInput<MLFloat16>("value", {1, sequence_length, head_size}, value.data(), value.size());
    test.AddOptionalInputEdge<MLFloat16>();  // past_key
    test.AddOptionalInputEdge<MLFloat16>();  // past_value
    test.AddInput<int32_t>("seqlens_k", {1}, {sequence_length - 1});
    test.AddInput<int32_t>("total_sequence_length", {1}, {sequence_length}, true);
    test.AddOutput<MLFloat16>("output", {1, sequence_length, head_size}, expected.data(), expected.size());
    test.SetOutputAbsErr("output", 0.001f);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    RunWithCooperativeDispatchCheck(true, [&] {
      test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
    });
  }
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
