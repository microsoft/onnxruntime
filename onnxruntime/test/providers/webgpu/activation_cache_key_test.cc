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

// Activation::ToString() is the cache hint used by every program that bakes the activation into its
// WGSL, and GetActivationSnippet() writes those same parameters into the shader text with
// std::to_string. The two formatters must agree on what counts as a distinct parameter: if the key
// collapses two values that the shader distinguishes, the first program compiled is served for both
// and the second silently runs with the wrong parameter.
//
// They did not agree. std::to_string is 6 *decimal places*, while an unconfigured std::stringstream
// is 6 *significant* digits, so the pairs below -- all exactly representable in float32 -- produced
// one key and two different shaders. Activation::ToString() now sets max_digits10, which is by
// definition the shortest precision that round-trips every float.
//
// These assertions are on the formatter alone, so the test needs no GPU and runs on any CI agent,
// including those where the WebGPU end-to-end tests skip for want of an adapter.
//
// A graph-level test covers this too -- see
// WebGpuSmallMatMulConvDistinguishesActivationParamsInPipelineCache -- but only for activations that
// can amplify the difference. Colliding under 6 significant digits bounds the two parameters to
// within about 1e-5 relative, so for an activation whose output is proportional to the parameter
// (LeakyRelu: alpha * x) the outputs stay within 1e-5 relative too, against a 2e-3 relative
// tolerance. That test therefore uses HardSigmoid, where clamping to [0, 1] turns the same 1e-5 into
// a full 0-versus-1 swing.

TEST(ActivationCacheKeyTest, DistinctLeakyReluAlphasProduceDistinctKeys) {
  constexpr float kAlphaA = 1234567.5f;
  constexpr float kAlphaB = 1234567.625f;
  // Guards the premise: if these ever became the same float the assertion below would be vacuous.
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

  // Clip stores {minimum, maximum}, so this exercises the second parameter as well as the first.
  const auto activation_a = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, kMaxA);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, kMaxB);

  EXPECT_NE(activation_a.ToString(), activation_b.ToString())
      << "two Clip bounds that emit different WGSL hashed to the same pipeline cache key";
}

// Positive control. Without this the two tests above would still pass if ToString() returned
// something unconditionally unique, which would defeat the cache rather than key it correctly.
TEST(ActivationCacheKeyTest, IdenticalActivationsProduceIdenticalKeys) {
  const auto activation_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);

  EXPECT_EQ(activation_a.ToString(), activation_b.ToString())
      << "identical activations must share a cache entry";

  const auto clip_a = MakeActivation(webgpu::ActivationKind::Clip, -6.0f, 6.0f);
  const auto clip_b = MakeActivation(webgpu::ActivationKind::Clip, -6.0f, 6.0f);

  EXPECT_EQ(clip_a.ToString(), clip_b.ToString());
}

// The activation kind must remain part of the key independently of the parameters: Relu and Sigmoid
// carry no parameters at all, so only the kind can separate them.
TEST(ActivationCacheKeyTest, DistinctActivationKindsProduceDistinctKeys) {
  const auto relu = MakeActivation(webgpu::ActivationKind::Relu, 0.0f);
  const auto sigmoid = MakeActivation(webgpu::ActivationKind::Sigmoid, 0.0f);

  EXPECT_NE(relu.ToString(), sigmoid.ToString());
}

}  // namespace test
}  // namespace onnxruntime
