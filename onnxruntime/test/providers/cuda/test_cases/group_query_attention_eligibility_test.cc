// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cstdint>
#include <type_traits>

#include "contrib_ops/cuda/bert/group_query_attention_eligibility.h"

namespace onnxruntime {
namespace test {
namespace {

using contrib::KVQuantizationType;
using contrib::cuda::GQAXqaSeqFreeInputs;
using contrib::cuda::IsGQACudnnSdpaCoreEligibleSeqFree;
using contrib::cuda::IsGQAXqaEligibleSeqFree;

template <typename U>
GQAXqaSeqFreeInputs XqaInputs() {
  GQAXqaSeqFreeInputs inputs;
  inputs.enable_xqa = true;
  inputs.is_unidirectional = true;
  inputs.device_major = 8;
  inputs.device_minor = 9;
  inputs.qk_norm_ok = true;
  inputs.smooth_softmax_supported = true;
  inputs.head_size = 128;
  inputs.num_heads = 8;
  inputs.kv_num_heads = 2;
  inputs.is_inputs_quantized = !std::is_same_v<U, MLFloat16> && !std::is_same_v<U, BFloat16>;
  if (inputs.is_inputs_quantized) {
    inputs.k_quant_type = KVQuantizationType::PER_TENSOR;
    inputs.v_quant_type = KVQuantizationType::PER_TENSOR;
  }
  return inputs;
}

template <typename U>
class GroupQueryAttentionXqaEligibilityTest : public testing::Test {};

#ifdef USE_FP8_KV_CACHE
using XqaCacheTypes = testing::Types<MLFloat16, BFloat16, int8_t, Float8E4M3FN>;
#else
using XqaCacheTypes = testing::Types<MLFloat16, BFloat16, int8_t>;
#endif
TYPED_TEST_SUITE(GroupQueryAttentionXqaEligibilityTest, XqaCacheTypes);

TYPED_TEST(GroupQueryAttentionXqaEligibilityTest, SupportsValidGeometryAndScaleCombinations) {
  auto inputs = XqaInputs<TypeParam>();
  for (int head_size : {64, 128, 256}) {
    for (int group_size : {1, 2, 4, 5, 8, 16, 32}) {
      SCOPED_TRACE(testing::Message() << "head_size=" << head_size << ", group_size=" << group_size);
      inputs.head_size = head_size;
      inputs.num_heads = group_size * inputs.kv_num_heads;
      const bool expected = !inputs.is_inputs_quantized || (group_size != 1 && group_size != 2 && group_size != 5);
      if (inputs.is_inputs_quantized) {
        for (auto k_type : {KVQuantizationType::PER_TENSOR, KVQuantizationType::PER_CHANNEL}) {
          for (auto v_type : {KVQuantizationType::PER_TENSOR, KVQuantizationType::PER_CHANNEL}) {
            SCOPED_TRACE(testing::Message() << "K=" << static_cast<int>(k_type) << ", V=" << static_cast<int>(v_type));
            inputs.k_quant_type = k_type;
            inputs.v_quant_type = v_type;
            EXPECT_EQ(IsGQAXqaEligibleSeqFree<TypeParam>(inputs), expected);
          }
        }
      } else {
        EXPECT_EQ(IsGQAXqaEligibleSeqFree<TypeParam>(inputs), expected);
      }
    }
  }
}

// H512 accepts arbitrary positive integral head ratios only with unquantized FP16/BF16 caches.
TYPED_TEST(GroupQueryAttentionXqaEligibilityTest, SupportsUnquantizedH512GeometryOnly) {
  auto inputs = XqaInputs<TypeParam>();
  inputs.head_size = 512;
  for (int group_size : {1, 2, 3, 5, 7, 8, 16, 32, 64}) {
    SCOPED_TRACE(testing::Message() << "group_size=" << group_size);
    inputs.num_heads = group_size * inputs.kv_num_heads;
    EXPECT_EQ(IsGQAXqaEligibleSeqFree<TypeParam>(inputs), !inputs.is_inputs_quantized);
  }
  inputs.device_major = 7;
  EXPECT_FALSE(IsGQAXqaEligibleSeqFree<TypeParam>(inputs));
}

TYPED_TEST(GroupQueryAttentionXqaEligibilityTest, RejectsEachIndependentGate) {
  const auto valid = XqaInputs<TypeParam>();
  ASSERT_TRUE(IsGQAXqaEligibleSeqFree<TypeParam>(valid));

  const struct {
    const char* name;
    void (*reject)(GQAXqaSeqFreeInputs&);
  } cases[] = {
      {"Disabled", [](auto& in) { in.enable_xqa = false; }},
      {"Noncausal", [](auto& in) { in.is_unidirectional = false; }},
      {"AttentionBias", [](auto& in) { in.has_attention_bias = true; }},
      {"PreAmpere", [](auto& in) { in.device_major = 7; }},
      {"Softcap", [](auto& in) { in.softcap = 1.0f; }},
      {"QkNorm", [](auto& in) { in.qk_norm_ok = false; }},
      {"SmoothSoftmax", [](auto& in) { in.smooth_softmax_supported = false; }},
      {"HeadSize", [](auto& in) { in.head_size = 96; }},
      {"GroupSize", [](auto& in) { in.num_heads = 6; }},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto inputs = valid;
    test_case.reject(inputs);
    EXPECT_FALSE(IsGQAXqaEligibleSeqFree<TypeParam>(inputs));
  }

  if (valid.is_inputs_quantized) {
    auto inputs = valid;
    inputs.k_quant_type = KVQuantizationType::NONE;
    EXPECT_FALSE(IsGQAXqaEligibleSeqFree<TypeParam>(inputs));
    inputs = valid;
    inputs.v_quant_type = KVQuantizationType::NONE;
    EXPECT_FALSE(IsGQAXqaEligibleSeqFree<TypeParam>(inputs));
  } else {
    auto inputs = valid;
    inputs.is_inputs_quantized = true;
    inputs.k_quant_type = KVQuantizationType::PER_TENSOR;
    inputs.v_quant_type = KVQuantizationType::PER_TENSOR;
    EXPECT_FALSE(IsGQAXqaEligibleSeqFree<TypeParam>(inputs));
  }
}

TEST(GroupQueryAttentionEligibilityTest, XqaInt8AndFp8ArchitectureGates) {
  const struct {
    int major;
    int minor;
    bool int8_supported;
    bool fp8_supported;
  } cases[] = {
      {7, 5, false, false},
      {8, 0, true, false},
      {8, 6, true, false},
      {8, 8, true, false},
      {8, 9, true, true},
      {9, 0, true, true},
      {10, 0, true, true},
      {12, 0, true, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(testing::Message() << "SM=" << test_case.major << test_case.minor);
    auto inputs = XqaInputs<int8_t>();
    inputs.device_major = test_case.major;
    inputs.device_minor = test_case.minor;
    EXPECT_EQ(IsGQAXqaEligibleSeqFree<int8_t>(inputs), test_case.int8_supported);
#if !defined(DISABLE_FLOAT8_TYPES)
#ifdef USE_FP8_KV_CACHE
    EXPECT_EQ(IsGQAXqaEligibleSeqFree<Float8E4M3FN>(inputs), test_case.fp8_supported);
#else
    EXPECT_FALSE(IsGQAXqaEligibleSeqFree<Float8E4M3FN>(inputs));
#endif
#endif
  }
}

TEST(GroupQueryAttentionEligibilityTest, XqaRejectsUnsupportedQuantizedStorage) {
  EXPECT_FALSE(IsGQAXqaEligibleSeqFree<uint8_t>(XqaInputs<uint8_t>()));
}

TEST(GroupQueryAttentionEligibilityTest, ExactGeometryRejectionDoesNotExcludeSmallerSupportedGeometry) {
  auto inputs = XqaInputs<MLFloat16>();
  inputs.head_size = 264;
  EXPECT_FALSE(IsGQAXqaEligibleSeqFree<MLFloat16>(inputs));
  inputs.head_size = 256;
  EXPECT_TRUE(IsGQAXqaEligibleSeqFree<MLFloat16>(inputs));

  inputs.num_heads = 6 * inputs.kv_num_heads;
  EXPECT_FALSE(IsGQAXqaEligibleSeqFree<MLFloat16>(inputs));
  inputs.num_heads = 5 * inputs.kv_num_heads;
  EXPECT_TRUE(IsGQAXqaEligibleSeqFree<MLFloat16>(inputs));
}

template <typename T>
class GroupQueryAttentionFp16Bf16EligibilityTest : public testing::Test {};

using QueryTypes = testing::Types<MLFloat16, BFloat16>;
TYPED_TEST_SUITE(GroupQueryAttentionFp16Bf16EligibilityTest, QueryTypes);

TYPED_TEST(GroupQueryAttentionFp16Bf16EligibilityTest, CudnnCoreChecksEachGate) {
  const struct {
    const char* name;
    bool has_attention_bias;
    bool is_inputs_quantized;
    float softcap;
    bool use_smooth_softmax;
    bool has_head_sink;
    int local_window_size;
    bool past_kv_format_bnsh;
    bool cudnn_enabled;
    bool expected;
  } cases[] = {
      {"Supported", false, false, 0.0f, false, false, -1, true, true, true},
      {"AttentionBias", true, false, 0.0f, false, false, -1, true, true, false},
      {"QuantizedInputs", false, true, 0.0f, false, false, -1, true, true, false},
      {"Softcap", false, false, 1.0f, false, false, -1, true, true, false},
      {"SmoothSoftmax", false, false, 0.0f, true, false, -1, true, true, false},
      {"HeadSink", false, false, 0.0f, false, true, -1, true, true, false},
      {"LocalWindow", false, false, 0.0f, false, false, 128, true, true, false},
      {"CacheLayout", false, false, 0.0f, false, false, -1, false, true, false},
      {"Disabled", false, false, 0.0f, false, false, -1, true, false, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ((IsGQACudnnSdpaCoreEligibleSeqFree<TypeParam, TypeParam>(
                  test_case.has_attention_bias, test_case.is_inputs_quantized, test_case.softcap,
                  test_case.use_smooth_softmax, test_case.has_head_sink, test_case.local_window_size,
                  test_case.past_kv_format_bnsh, test_case.cudnn_enabled)),
              test_case.expected);
  }
  using OtherType = std::conditional_t<std::is_same_v<TypeParam, MLFloat16>, BFloat16, MLFloat16>;
  EXPECT_FALSE((IsGQACudnnSdpaCoreEligibleSeqFree<TypeParam, OtherType>(
      false, false, 0.0f, false, false, -1, true, true)));
}

#if USE_FLASH_ATTENTION
TYPED_TEST(GroupQueryAttentionFp16Bf16EligibilityTest, FlashChecksEachGate) {
#ifdef ORT_QUICK_BUILD
  constexpr bool supports_other_head_sizes = false;
#else
  constexpr bool supports_other_head_sizes = true;
#endif
  const struct {
    const char* name;
    int device_major;
    bool has_attention_bias;
    bool disabled;
    int64_t head_size;
    int64_t num_heads;
    int64_t kv_num_heads;
    bool expected;
  } cases[] = {
      {"Supported", 8, false, false, 128, 8, 2, true},
      {"Hopper", 9, false, false, 128, 8, 2, true},
      {"Blackwell", 12, false, false, 128, 8, 2, true},
      {"AttentionBias", 8, true, false, 128, 8, 2, false},
      {"Disabled", 8, false, true, 128, 8, 2, false},
      {"PreAmpere", 7, false, false, 128, 8, 2, false},
      {"HeadAlignment", 8, false, false, 129, 8, 2, false},
      {"HeadSizeLimit", 8, false, false, 264, 8, 2, false},
      {"HeadRatio", 8, false, false, 128, 7, 2, false},
      {"OtherHeadSize", 8, false, false, 64, 8, 2, supports_other_head_sizes},
      {"MaximumHeadSize", 8, false, false, 256, 8, 2, supports_other_head_sizes},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    cudaDeviceProp device{};
    device.major = test_case.device_major;
    EXPECT_EQ(contrib::cuda::IsGQAFlashEligibleSeqFree<TypeParam>(
                  device, test_case.has_attention_bias, test_case.disabled,
                  test_case.head_size, test_case.num_heads, test_case.kv_num_heads),
              test_case.expected);
  }
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
TYPED_TEST(GroupQueryAttentionFp16Bf16EligibilityTest, MemoryEfficientChecksEachGate) {
  constexpr bool is_half = std::is_same_v<TypeParam, MLFloat16>;
  const struct {
    const char* name;
    int32_t sm;
    bool disabled;
    bool is_inputs_quantized;
    bool has_attention_bias;
    int64_t head_size;
    bool expected;
  } cases[] = {
      {"Supported", 80, false, false, false, 128, true},
      {"Disabled", 80, true, false, false, 128, false},
      {"QuantizedInputs", 80, false, true, false, 128, false},
      {"AttentionBias", 80, false, false, true, 128, false},
      {"PreMaxwell", 49, false, false, false, 128, false},
      {"PreFp16", 52, false, false, false, 128, false},
      {"Fp16Minimum", 53, false, false, false, 128, is_half},
      {"PreBf16", 75, false, false, false, 128, is_half},
      {"Hopper", 90, false, false, false, 128, true},
      {"HeadAlignment", 80, false, false, false, 129, false},
      {"MaximumHeadSize", 80, false, false, false, 1024, true},
      {"HeadSizeLimit", 80, false, false, false, 1032, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(contrib::cuda::IsGQAMemoryEfficientEligibleSeqFree<TypeParam>(
                  test_case.sm, test_case.disabled, test_case.is_inputs_quantized,
                  test_case.has_attention_bias, test_case.head_size),
              test_case.expected);
  }
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
