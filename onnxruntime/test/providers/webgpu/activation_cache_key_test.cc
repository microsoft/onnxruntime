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

TEST(ActivationCacheKeyTest, ParametersDoNotAffectTheKey) {
  const auto leaky_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto leaky_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.2f);
  EXPECT_EQ(leaky_a.CacheKey(), leaky_b.CacheKey())
      << "LeakyRelu alpha is a uniform, so it must not fork the pipeline cache";

  const auto clip_a = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, 6.0f);
  const auto clip_b = MakeActivation(webgpu::ActivationKind::Clip, -1.0f, 1.0f);
  EXPECT_EQ(clip_a.CacheKey(), clip_b.CacheKey())
      << "Clip bounds are uniforms, so they must not fork the pipeline cache";
}

TEST(ActivationCacheKeyTest, QuickGeluUnitAlphaIsKeyedSeparately) {
  const auto unit_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 1.0f);
  const auto other_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 1.702f);

  EXPECT_NE(unit_alpha.CacheKey(), other_alpha.CacheKey())
      << "QuickGelu drops the multiply at alpha == 1, which is different shader text";

  const auto another_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 0.5f);
  EXPECT_EQ(other_alpha.CacheKey(), another_alpha.CacheKey());
}

TEST(ActivationCacheKeyTest, IdenticalActivationsProduceIdenticalKeys) {
  const auto activation_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);

  EXPECT_EQ(activation_a.CacheKey(), activation_b.CacheKey())
      << "identical activations must share a cache entry";
}

TEST(ActivationCacheKeyTest, DistinctActivationKindsProduceDistinctKeys) {
  const auto relu = MakeActivation(webgpu::ActivationKind::Relu, 0.0f);
  const auto sigmoid = MakeActivation(webgpu::ActivationKind::Sigmoid, 0.0f);

  EXPECT_NE(relu.CacheKey(), sigmoid.CacheKey());
}

}  // namespace test
}  // namespace onnxruntime
