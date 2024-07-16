// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef DISABLE_CONTRIB_OPS

#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/util/include/scoped_env_vars.h"
#include "gtest/gtest.h"

#include <unordered_map>
#include <string>

using onnxruntime::contrib::cuda::AttentionBackend;
using onnxruntime::contrib::cuda::AttentionKernelOptions;

namespace onnxruntime {
namespace test {

TEST(AttentionKernelOptionsTest, NonZeroValue) {
  constexpr bool force_init = true;
  int value = static_cast<int>(AttentionBackend::FLASH_ATTENTION) | static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION);
  const AttentionKernelOptions* options = AttentionKernelOptions::GetInstance(value, force_init);
  ASSERT_TRUE(options->UseFlashAttention());
  ASSERT_TRUE(options->UseEfficientAttention());
  ASSERT_FALSE(options->UseTrtFusedAttention());
  ASSERT_FALSE(options->UseUnfusedAttention());
  ASSERT_FALSE(options->UseTrtFlashAttention());
  ASSERT_FALSE(options->UseTrtCrossAttention());
  ASSERT_FALSE(options->UseTrtCausalAttention());

  value = static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION) | static_cast<int>(AttentionBackend::MATH);
  options = AttentionKernelOptions::GetInstance(value, force_init);
  ASSERT_FALSE(options->UseFlashAttention());
  ASSERT_FALSE(options->UseEfficientAttention());
  ASSERT_TRUE(options->UseTrtFusedAttention());
  ASSERT_TRUE(options->UseUnfusedAttention());
  ASSERT_FALSE(options->UseTrtFlashAttention());
  ASSERT_FALSE(options->UseTrtCrossAttention());
  ASSERT_FALSE(options->UseTrtCausalAttention());

  value = static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION);
  options = AttentionKernelOptions::GetInstance(value, force_init);
  ASSERT_FALSE(options->UseFlashAttention());
  ASSERT_FALSE(options->UseEfficientAttention());
  ASSERT_FALSE(options->UseTrtFusedAttention());
  ASSERT_FALSE(options->UseUnfusedAttention());
  ASSERT_TRUE(options->UseTrtFlashAttention());
  ASSERT_FALSE(options->UseTrtCrossAttention());
  ASSERT_FALSE(options->UseTrtCausalAttention());

  value = static_cast<int>(AttentionBackend::TRT_CROSS_ATTENTION) | static_cast<int>(AttentionBackend::TRT_CAUSAL_ATTENTION);
  options = AttentionKernelOptions::GetInstance(value, force_init);
  ASSERT_FALSE(options->UseFlashAttention());
  ASSERT_FALSE(options->UseEfficientAttention());
  ASSERT_FALSE(options->UseTrtFusedAttention());
  ASSERT_FALSE(options->UseUnfusedAttention());
  ASSERT_FALSE(options->UseTrtFlashAttention());
  ASSERT_TRUE(options->UseTrtCrossAttention());
  ASSERT_TRUE(options->UseTrtCausalAttention());

  // Test environment variables are ignored when option value is non-zero
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
          {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "256"}}};
  value = static_cast<int>(AttentionBackend::FLASH_ATTENTION);
  options = AttentionKernelOptions::GetInstance(value, force_init);
  ASSERT_TRUE(options->UseFlashAttention());
  ASSERT_FALSE(options->UseEfficientAttention());
  ASSERT_FALSE(options->UseTrtFusedAttention());
  ASSERT_FALSE(options->UseUnfusedAttention());
  ASSERT_FALSE(options->UseTrtFlashAttention());
  ASSERT_FALSE(options->UseTrtCrossAttention());
  ASSERT_FALSE(options->UseTrtCausalAttention());
  EXPECT_EQ(options->MinSeqLenForFlashAttentionPackedQkv(), 256);
}

TEST(AttentionKernelOptionsTest, ZeroValueWithEnvVar) {
  constexpr bool force_init = true;
  int value = 0;
  const AttentionKernelOptions* options = nullptr;

  // Test environment variables take effect when option value is 0
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
            {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "128"}}};
    options = AttentionKernelOptions::GetInstance(value, force_init);
    ASSERT_TRUE(options->UseFlashAttention());
    ASSERT_TRUE(options->UseEfficientAttention());
    ASSERT_TRUE(options->UseTrtFusedAttention());
    ASSERT_TRUE(options->UseUnfusedAttention());
    ASSERT_TRUE(options->UseTrtFlashAttention());
    ASSERT_TRUE(options->UseTrtCrossAttention());
    ASSERT_TRUE(options->UseTrtCausalAttention());
    ASSERT_TRUE(options->UseTrtCausalAttention());
    EXPECT_EQ(options->MinSeqLenForFlashAttentionPackedQkv(), 128);
  }

  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "0"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "0"},
            {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "64"}}};
    options = AttentionKernelOptions::GetInstance(value, force_init);
    ASSERT_FALSE(options->UseFlashAttention());
    ASSERT_FALSE(options->UseEfficientAttention());
    ASSERT_FALSE(options->UseTrtFusedAttention());
    ASSERT_TRUE(options->UseUnfusedAttention());
    ASSERT_FALSE(options->UseTrtFlashAttention());
    ASSERT_FALSE(options->UseTrtCrossAttention());
    ASSERT_FALSE(options->UseTrtCausalAttention());
    ASSERT_FALSE(options->UseTrtCausalAttention());
    EXPECT_EQ(options->MinSeqLenForFlashAttentionPackedQkv(), 64);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
