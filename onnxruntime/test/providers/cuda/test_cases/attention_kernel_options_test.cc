// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef DISABLE_CONTRIB_OPS

#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/util/include/scoped_env_vars.h"
#include "gtest/gtest.h"

#include <unordered_map>
#include <string>

using onnxruntime::AttentionKernelOptions;
using onnxruntime::contrib::attention::AttentionBackend;

namespace onnxruntime {
namespace test {

TEST(CudaEpAttentionKernelOptionsTest, NonZeroValue) {
  {
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::FLASH_ATTENTION) | static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_TRUE(options.UseFlashAttention());
    ASSERT_TRUE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  {
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION) | static_cast<int>(AttentionBackend::MATH);
    options.InitializeOnce(value, false);
    ASSERT_FALSE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_TRUE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_TRUE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  {
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::CUDNN_FLASH_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_FALSE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_TRUE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  {
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_FALSE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_TRUE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  {
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::TRT_CROSS_ATTENTION) | static_cast<int>(AttentionBackend::TRT_CAUSAL_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_FALSE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_TRUE(options.UseTrtCrossAttention());
    ASSERT_TRUE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  // Test environment variables are ignored when option value is non-zero
  // Test default min sequence lengths are zeros
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kEnableCudnnFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
            {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
            {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"}}};
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::FLASH_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_TRUE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 0);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 0);
  }

  // Test min sequence lengths can be parsed from environment variables when option value is non-zero
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
            {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "128"},
            {onnxruntime::contrib::attention::kMinSeqLenForEfficientAttentionFp32, "256"}}};
    AttentionKernelOptions options;
    int value = static_cast<int>(AttentionBackend::FLASH_ATTENTION);
    options.InitializeOnce(value, false);
    ASSERT_TRUE(options.UseFlashAttention());
    ASSERT_FALSE(options.UseEfficientAttention());
    ASSERT_FALSE(options.UseTrtFusedAttention());
    ASSERT_FALSE(options.UseCudnnFlashAttention());
    ASSERT_FALSE(options.UseUnfusedAttention());
    ASSERT_FALSE(options.UseTrtFlashAttention());
    ASSERT_FALSE(options.UseTrtCrossAttention());
    ASSERT_FALSE(options.UseTrtCausalAttention());
    EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 128);
    EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 256);
  }
}

// Test all environment variables take effect when option value is 0.
TEST(CudaEpAttentionKernelOptionsTest, DefaultOptionWithEnvVar) {
  constexpr int value = 0;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
          {onnxruntime::contrib::attention::kEnableCudnnFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "1"},
          {onnxruntime::contrib::attention::kMinSeqLenForFlashAttentionPackedQKV, "128"},
          {onnxruntime::contrib::attention::kMinSeqLenForEfficientAttentionFp32, "256"}}};
  AttentionKernelOptions options;
  options.InitializeOnce(value, false);
  ASSERT_TRUE(options.UseFlashAttention());
  ASSERT_TRUE(options.UseEfficientAttention());
  ASSERT_TRUE(options.UseTrtFusedAttention());
  ASSERT_TRUE(options.UseCudnnFlashAttention());
  ASSERT_TRUE(options.UseUnfusedAttention());
  ASSERT_TRUE(options.UseTrtFlashAttention());
  ASSERT_TRUE(options.UseTrtCrossAttention());
  ASSERT_TRUE(options.UseTrtCausalAttention());
  ASSERT_TRUE(options.UseTrtCausalAttention());
  EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(), 128);
  EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(), 256);
}

// Test default min sequence lengths when environment variables are not set.
TEST(CudaEpAttentionKernelOptionsTest, DefaultMinSeqLens) {
  constexpr int value = 0;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kEnableCudnnFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "0"},
          {onnxruntime::contrib::attention::kEnableFusedCausalAttention, "0"}}};
  AttentionKernelOptions options;
  options.InitializeOnce(value, false);
  ASSERT_FALSE(options.UseFlashAttention());
  ASSERT_FALSE(options.UseEfficientAttention());
  ASSERT_FALSE(options.UseTrtFusedAttention());
  ASSERT_FALSE(options.UseCudnnFlashAttention());
  ASSERT_TRUE(options.UseUnfusedAttention());
  ASSERT_FALSE(options.UseTrtFlashAttention());
  ASSERT_FALSE(options.UseTrtCrossAttention());
  ASSERT_FALSE(options.UseTrtCausalAttention());
  ASSERT_FALSE(options.UseTrtCausalAttention());
  EXPECT_EQ(options.MinSeqLenForFlashAttentionPackedQkv(),
            onnxruntime::contrib::attention::kDefaultMinSeqLenForFlashAttentionPackedQKV);
  EXPECT_EQ(options.MinSeqLenForEfficientAttentionFp32(),
            onnxruntime::contrib::attention::kDefaultMinSeqLenForEfficientAttentionFp32);
}

}  // namespace test
}  // namespace onnxruntime

#endif
