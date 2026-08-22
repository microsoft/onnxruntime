// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "gtest/gtest.h"

#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace test {
namespace {

webgpu::Activation MakeActivation(webgpu::ActivationKind kind, float value0, float value1 = 0.0f) {
  webgpu::Activation activation;
  activation.activation_kind_ = kind;
  activation.activation_params_.values_[0] = value0;
  activation.activation_params_.values_[1] = value1;
  return activation;
}

}  // namespace

TEST(ActivationCacheKeyTest, DistinctLeakyReluAlphasProduceDistinctKeys) {
  constexpr float kAlphaA = 1234567.5f;
  constexpr float kAlphaB = 1234567.625f;
  ASSERT_NE(kAlphaA, kAlphaB);

  const auto activation_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, kAlphaA);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, kAlphaB);

  EXPECT_NE(activation_a.ToString(), activation_b.ToString())
      << "two LeakyRelu alphas that emit different WGSL hashed to the same pipeline cache key";
}

TEST(ActivationCacheKeyTest, DistinctClipBoundsProduceDistinctKeys) {
  constexpr float kMaxA = 1000000.0f;
  constexpr float kMaxB = 1000000.0625f;
  ASSERT_NE(kMaxA, kMaxB);

  // Exercise the second activation parameter.
  const auto activation_a = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, kMaxA);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, kMaxB);

  EXPECT_NE(activation_a.ToString(), activation_b.ToString())
      << "two Clip bounds that emit different WGSL hashed to the same pipeline cache key";
}

TEST(ActivationCacheKeyTest, IdenticalActivationsProduceIdenticalKeys) {
  const auto activation_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);

  EXPECT_EQ(activation_a.ToString(), activation_b.ToString())
      << "identical activations must share a cache entry";

  const auto clip_a = MakeActivation(webgpu::ActivationKind::Clip, -6.0f, 6.0f);
  const auto clip_b = MakeActivation(webgpu::ActivationKind::Clip, -6.0f, 6.0f);

  EXPECT_EQ(clip_a.ToString(), clip_b.ToString());
}

TEST(ActivationCacheKeyTest, DistinctActivationKindsProduceDistinctKeys) {
  const auto relu = MakeActivation(webgpu::ActivationKind::Relu, 0.0f);
  const auto sigmoid = MakeActivation(webgpu::ActivationKind::Sigmoid, 0.0f);

  EXPECT_NE(relu.ToString(), sigmoid.ToString());
}

}  // namespace test
}  // namespace onnxruntime
