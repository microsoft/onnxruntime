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
#include "test/common/tensor_op_test_utils.h"
#include "test/contrib_ops/group_query_attention_op_test_helper.h"
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
                                          int window, int capacity) {
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
                     key_cache[(batch * capacity + key) * kHeadSize + dim].ToFloat();
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
            sum += scores[key] * value_cache[(batch * capacity + key) * kHeadSize + dim].ToFloat();
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

  void RunPrefill(bool capture, int window, uint64_t binding_limit = 0) {
    const int capacity = binding_limit > 0 ? 512 : kCapacity;
    const int first_past = binding_limit > 0 ? 193 : 17;
    SetTestFunctionCalled();
    const auto query = MakePrefillData(kBatchSize * kSequenceLength * kNumHeads * kHeadSize, 7);
    const auto key = MakePrefillData(kBatchSize * kSequenceLength * kHeadSize, 11);
    const auto value = MakePrefillData(key.size(), 13);
    const auto initial_key = MakePrefillData(kBatchSize * capacity * kHeadSize, 17);
    const auto initial_value = MakePrefillData(initial_key.size(), 19);
    const TensorShape output_shape{kBatchSize, kSequenceLength, kNumHeads * kHeadSize};
    const TensorShape cache_shape{kBatchSize, 1, capacity, kHeadSize};
    AddAttribute<int64_t>("num_heads", kNumHeads);
    AddAttribute<int64_t>("kv_num_heads", 1);
    AddAttribute<int64_t>("local_window_size", window);
    AddInput<MLFloat16>("query", {kBatchSize, kSequenceLength, kNumHeads * kHeadSize}, query.data(), query.size());
    AddInput<MLFloat16>("key", {kBatchSize, kSequenceLength, kHeadSize}, key.data(), key.size());
    AddInput<MLFloat16>("value", {kBatchSize, kSequenceLength, kHeadSize}, value.data(), value.size());
    AddInput<MLFloat16>("past_key", {kBatchSize, 1, capacity, kHeadSize}, initial_key.data(), initial_key.size());
    AddInput<MLFloat16>("past_value", {kBatchSize, 1, capacity, kHeadSize}, initial_value.data(), initial_value.size());
    AddInput<int32_t>("seqlens_k", {kBatchSize}, {first_past + kSequenceLength - 1});
    AddInput<int32_t>("total_sequence_length", {1}, {first_past + kSequenceLength});
    AddOutput<MLFloat16>("output", {kBatchSize, kSequenceLength, kNumHeads * kHeadSize}, query.data(), query.size());
    AddOutput<MLFloat16>("present_key", {kBatchSize, 1, capacity, kHeadSize}, initial_key.data(), initial_key.size());
    AddOutput<MLFloat16>("present_value", {kBatchSize, 1, capacity, kHeadSize}, initial_value.data(), initial_value.size());
    auto& model = BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve(GetRunContext().resolve_options));
    const auto serialized = model.ToProto().SerializeAsString();

    ConfigOptions provider_options;
    if (capture) {
      ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                                       webgpu::options::kEnableGraphCapture_ON));
    }
    WebGpuExecutionProviderTestOptions test_options{};
    test_options.max_storage_buffer_binding_size = binding_limit;
    auto provider = binding_limit > 0 ? WebGpuExecutionProviderWithTestOptions(provider_options, test_options)
                                      : WebGpuExecutionProviderWithOptions(provider_options);
    ASSERT_NE(provider, nullptr);
    auto* ep = provider.get();
    auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
    if (binding_limit > 0) {
      ASSERT_EQ(context.DeviceLimits().maxStorageBufferBindingSize, binding_limit);
    }
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
    const std::array<std::array<int, kBatchSize>, 3> past_lengths =
        binding_limit > 0 ? std::array<std::array<int, kBatchSize>, 3>{{{193}, {257}, {129}}}
                          : std::array<std::array<int, kBatchSize>, 3>{{{17}, {41}, {3}}};
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
        const int target = (batch * capacity + past[batch]) * kHeadSize;
        std::copy_n(key.begin() + source, kSequenceLength * kHeadSize, expected_key.begin() + target);
        std::copy_n(value.begin() + source, kSequenceLength * kHeadSize, expected_value.begin() + target);
      }
      const auto expected = PrefillReference(query, expected_key, expected_value, past, window, capacity);
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

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill) {
  if (!SupportsQualcommCooperativePrefill()) {
    GTEST_SKIP() << "Requires Qualcomm FP16 and 128-lane subgroup control";
  }
  for (int sequence_length : {128, 129, 257}) {
    for (int batch_size : {1, 2}) {
      SCOPED_TRACE(testing::Message() << "sequence_length=" << sequence_length << ", batch=" << batch_size);
      constexpr int num_heads = 6;
      constexpr int kv_num_heads = 2;
      constexpr int head_size = 128;
      RandomValueGenerator random{1234};
      const auto query = random.Gaussian<float>(std::array<int64_t, 3>{batch_size, sequence_length, num_heads * head_size}, 0.0f, 1.0f);
      const auto key = random.Gaussian<float>(std::array<int64_t, 3>{batch_size, sequence_length, kv_num_heads * head_size}, 0.0f, 1.0f);
      const auto value = random.Gaussian<float>(std::array<int64_t, 3>{batch_size, sequence_length, kv_num_heads * head_size}, 0.0f, 1.0f);
      const bool rotary = batch_size == 2;
      const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                             query, key, value, rotary);
      const auto actual = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                          query, key, value, rotary, /*use_fp16=*/true);
      ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.003f,
                                /*max_absolute_error=*/0.005f);
    }
  }
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_HighMagnitude) {
  if (!SupportsQualcommCooperativePrefill()) {
    GTEST_SKIP() << "Requires Qualcomm FP16 and 128-lane subgroup control";
  }
  RunFp16HighMagnitudeAttention(/*sequence_length=*/129, /*bit_width=*/0);
}

static void RunWebGpuPrefillFeatures(bool verify_dispatch) {
  if (!SupportsQualcommCooperativePrefill()) {
    GTEST_SKIP() << "Requires Qualcomm FP16 and 128-lane subgroup control";
  }
  constexpr int batch_size = 2;
  constexpr int sequence_length = 129;
  constexpr int num_heads = 6;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  RandomValueGenerator random{4321};
  const auto query = ToFloat16(random.Gaussian<float>(
      std::array<int64_t, 3>{batch_size, sequence_length, num_heads * head_size}, 0.0f, 1.0f));
  const auto key = ToFloat16(random.Gaussian<float>(
      std::array<int64_t, 3>{batch_size, sequence_length, kv_num_heads * head_size}, 0.0f, 1.0f));
  const auto value = ToFloat16(random.Gaussian<float>(
      std::array<int64_t, 3>{batch_size, sequence_length, kv_num_heads * head_size}, 0.0f, 1.0f));
  const auto sinks = ToFloat16({-2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  struct FeatureCase {
    float softcap;
    int bias_batches;
    int bias_heads;
    bool sink;
    int window;
  };
  for (const auto& c : {FeatureCase{0.5f, 0, 0, false, -1},
                        FeatureCase{0.0f, 1, 1, false, -1},
                        FeatureCase{0.0f, 1, num_heads, false, -1},
                        FeatureCase{0.0f, batch_size, 1, false, -1},
                        FeatureCase{0.0f, batch_size, num_heads, false, -1},
                        FeatureCase{0.0f, 0, 0, true, -1},
                        FeatureCase{0.0f, 0, 0, false, 1},
                        FeatureCase{0.0f, 0, 0, false, 17},
                        FeatureCase{0.0f, 0, 0, false, 2048},
                        FeatureCase{0.5f, batch_size, num_heads, true, 17}}) {
    SCOPED_TRACE(testing::Message() << "softcap=" << c.softcap << " bias=" << c.bias_batches << ","
                                    << c.bias_heads << " sink=" << c.sink << " window=" << c.window);
    const auto bias = ToFloat16(random.Gaussian<float>(
        std::array<int64_t, 4>{c.bias_batches, c.bias_heads, sequence_length, sequence_length}, 0.0f, 1.0f));
    InlinedVector<MLFloat16> expected(query.size());
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int row = 0; row < sequence_length; ++row) {
        const int first = c.window > 0 ? std::max(0, row + 1 - c.window) : 0;
        for (int head = 0; head < num_heads; ++head) {
          const int kv_head = head / (num_heads / kv_num_heads);
          const int q_offset = ((batch * sequence_length + row) * num_heads + head) * head_size;
          InlinedVector<double> scores(row + 1);
          double maximum = c.sink ? sinks[head].ToFloat() : -std::numeric_limits<double>::infinity();
          for (int k = first; k <= row; ++k) {
            const int k_offset = ((batch * sequence_length + k) * kv_num_heads + kv_head) * head_size;
            double score = 0.0;
            for (int d = 0; d < head_size; ++d) {
              score += static_cast<double>(query[q_offset + d].ToFloat()) * key[k_offset + d].ToFloat();
            }
            score *= scale;
            if (c.softcap > 0.0f) {
              score = std::tanh(score / c.softcap) * c.softcap;
            }
            if (!bias.empty()) {
              const int bias_batch = c.bias_batches == 1 ? 0 : batch;
              const int bias_head = c.bias_heads == 1 ? 0 : head;
              const int offset =
                  ((bias_batch * c.bias_heads + bias_head) * sequence_length + row) * sequence_length + k;
              score += bias[offset].ToFloat();
            }
            scores[k] = score;
            maximum = std::max(maximum, score);
          }
          double denominator = c.sink ? std::exp(sinks[head].ToFloat() - maximum) : 0.0;
          for (int k = first; k <= row; ++k) {
            scores[k] = std::exp(scores[k] - maximum);
            denominator += scores[k];
          }
          for (int d = 0; d < head_size; ++d) {
            double sum = 0.0;
            for (int k = first; k <= row; ++k) {
              const int offset = ((batch * sequence_length + k) * kv_num_heads + kv_head) * head_size + d;
              sum += scores[k] * value[offset].ToFloat();
            }
            expected[q_offset + d] = MLFloat16(static_cast<float>(sum / denominator));
          }
        }
      }
    }
    OpTester test("GroupQueryAttention", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
    test.AddAttribute<float>("softcap", c.softcap);
    test.AddAttribute<int64_t>("local_window_size", c.window);
    test.AddInput<MLFloat16>("query", {batch_size, sequence_length, num_heads * head_size}, query);
    test.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_num_heads * head_size}, key);
    test.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_num_heads * head_size}, value);
    test.AddOptionalInputEdge<MLFloat16>();  // past_key
    test.AddOptionalInputEdge<MLFloat16>();  // past_value
    test.AddInput<int32_t>("seqlens_k", {batch_size}, {sequence_length - 1, sequence_length - 1});
    test.AddInput<int32_t>("total_sequence_length", {1}, {sequence_length}, true);
    test.AddOptionalInputEdge<MLFloat16>();  // cos_cache
    test.AddOptionalInputEdge<MLFloat16>();  // sin_cache
    test.AddOptionalInputEdge<int64_t>();    // position_ids
    if (!bias.empty()) {
      test.AddInput<MLFloat16>("attention_bias", {c.bias_batches, c.bias_heads, sequence_length, sequence_length},
                               bias);
    } else {
      test.AddOptionalInputEdge<MLFloat16>();
    }
    if (c.sink) {
      test.AddInput<MLFloat16>("head_sink", {num_heads}, sinks);
    }
    test.AddOutput<MLFloat16>("output", {batch_size, sequence_length, num_heads * head_size},
                              expected.data(), expected.size());
    test.SetOutputAbsErr("output", 0.003f);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    RunWithCooperativeDispatchCheck(verify_dispatch, [&] {
      test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
    });
  }
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_Features) {
  RunWebGpuPrefillFeatures(false);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_Features_Dispatch) {
  if (!SupportsCooperativeDispatchObservation()) {
    GTEST_SKIP() << "Requires Qualcomm cooperative prefill and timestamp queries";
  }
  RunWebGpuPrefillFeatures(true);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_PastAndPadding) {
  if (!SupportsQualcommCooperativePrefill()) {
    GTEST_SKIP() << "Requires Qualcomm FP16 and 128-lane subgroup control";
  }
  constexpr int sequence_length = 129;
  constexpr int num_heads = 6;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 128;
  for (int past_length : {0, 17}) {
    SCOPED_TRACE(past_length);
    const int batch_size = past_length == 0 ? 2 : 1;
    const int capacity = past_length + sequence_length + 11;
    InlinedVector<MLFloat16> query(batch_size * sequence_length * num_heads * head_size, MLFloat16(0.0f));
    InlinedVector<MLFloat16> key(batch_size * sequence_length * kv_num_heads * head_size, MLFloat16(0.0f));
    InlinedVector<MLFloat16> value(key.size());
    InlinedVector<MLFloat16> past_key(batch_size * kv_num_heads * capacity * head_size, MLFloat16(0.0f));
    InlinedVector<MLFloat16> past_value(past_key.size(), MLFloat16(0.0f));
    InlinedVector<MLFloat16> expected(query.size());
    InlinedVector<int32_t> seqlens_k(batch_size);
    const auto value_at = [](int batch, int kv_head, int row, int dim) {
      return batch * 0.5f + kv_head * 0.25f + (row % 7) * 0.125f + (dim % 4) * 0.0625f;
    };
    for (int batch = 0; batch < batch_size; ++batch) {
      const int valid_keys = past_length == 0 && batch == 1 ? 101 : past_length + sequence_length;
      seqlens_k[batch] = valid_keys - 1;
      for (int kv_head = 0; kv_head < kv_num_heads; ++kv_head) {
        for (int row = 0; row < past_length; ++row) {
          for (int dim = 0; dim < head_size; ++dim) {
            past_value[((batch * kv_num_heads + kv_head) * capacity + row) * head_size + dim] =
                MLFloat16(value_at(batch, kv_head, row, dim));
          }
        }
        for (int row = 0; row < sequence_length; ++row) {
          for (int dim = 0; dim < head_size; ++dim) {
            value[((batch * sequence_length + row) * kv_num_heads + kv_head) * head_size + dim] =
                MLFloat16(value_at(batch, kv_head, past_length + row, dim));
          }
        }
      }
      // Zero queries give uniform probabilities over the visible causal prefix.
      for (int row = 0; row < sequence_length; ++row) {
        const int count = std::min(valid_keys, past_length + row + 1);
        for (int head = 0; head < num_heads; ++head) {
          for (int dim = 0; dim < head_size; ++dim) {
            float sum = 0.0f;
            for (int k = 0; k < count; ++k) {
              sum += value_at(batch, head / (num_heads / kv_num_heads), k, dim);
            }
            expected[((batch * sequence_length + row) * num_heads + head) * head_size + dim] =
                MLFloat16(sum / count);
          }
        }
      }
    }
    OpTester test("GroupQueryAttention", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
    test.AddInput<MLFloat16>("query", {batch_size, sequence_length, num_heads * head_size}, query.data(), query.size());
    test.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_num_heads * head_size}, key.data(), key.size());
    test.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_num_heads * head_size}, value.data(), value.size());
    test.AddInput<MLFloat16>("past_key", {batch_size, kv_num_heads, capacity, head_size}, past_key.data(), past_key.size());
    test.AddInput<MLFloat16>("past_value", {batch_size, kv_num_heads, capacity, head_size}, past_value.data(), past_value.size());
    test.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k.data(), seqlens_k.size());
    test.AddInput<int32_t>("total_sequence_length", {1}, {past_length + sequence_length}, true);
    test.AddOutput<MLFloat16>("output", {batch_size, sequence_length, num_heads * head_size}, expected.data(), expected.size());
    test.SetOutputAbsErr("output", 0.001f);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
  }
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_Dispatch) {
  RunCooperativePrefillTest(false);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_GraphCaptureReplay) {
  RunCooperativePrefillTest(true);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_CooperativePrefill_SegmentedBuffers) {
  if (!SupportsCooperativeDispatchObservation()) {
    GTEST_SKIP() << "Requires Qualcomm cooperative prefill and timestamp queries";
  }
  {
    auto ep = DefaultWebGpuExecutionProvider();
    const auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
    if (context.DeviceLimits().maxStorageBuffersPerShaderStage < 9) {
      GTEST_SKIP() << "Segmented Q, K, V and output require nine storage bindings";
    }
  }
  for (bool capture : {false, true}) {
    for (int window : {-1, 17}) {
      CooperativePrefillTester test;
      // Q/output span two bindings; the 512-position caches also cross the
      // 256-position binding boundary in each run, including graph replays.
      test.RunPrefill(capture, window, /*binding_limit=*/65536);
    }
  }
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
