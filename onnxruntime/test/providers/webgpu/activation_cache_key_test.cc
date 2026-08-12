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

// Activation::ToString() is the cache hint used by every program that embeds a fused activation in
// its WGSL. The rule it has to follow is exact: it must distinguish two activations if and only if
// they generate different shader text.
//
// Now that parameters travel as uniforms, they no longer appear in the shader text, so they must NOT
// appear in the key -- otherwise every distinct alpha would compile its own identical pipeline. The
// one exception is QuickGelu at alpha == 1, where the multiply folds away and the emitted code
// really does differ, so that variant is keyed as a boolean.
//
// These assertions are on the formatter alone, so the test needs no GPU and runs on any CI agent.
// They pin the sharing rule itself; that the parameters still reach the shader by another route is
// what the end-to-end parity tests cover.

// The uniforms change: parameters are deliberately absent from the key, so two activations that
// differ only in a parameter share one compiled pipeline.
TEST(ActivationCacheKeyTest, ParametersDoNotAffectTheKey) {
  const auto leaky_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto leaky_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.2f);
  EXPECT_EQ(leaky_a.ToString(), leaky_b.ToString())
      << "LeakyRelu alpha is a uniform, so it must not fork the pipeline cache";

  // Clip stores {minimum, maximum}, so this exercises the second parameter as well as the first.
  const auto clip_a = MakeActivation(webgpu::ActivationKind::Clip, 0.0f, 6.0f);
  const auto clip_b = MakeActivation(webgpu::ActivationKind::Clip, -1.0f, 1.0f);
  EXPECT_EQ(clip_a.ToString(), clip_b.ToString())
      << "Clip bounds are uniforms, so they must not fork the pipeline cache";
}

// The one parameter that still changes the generated code. Without this, the alpha == 1 fast path
// and the general path would share a key and one would be served for the other.
TEST(ActivationCacheKeyTest, QuickGeluUnitAlphaIsKeyedSeparately) {
  const auto unit_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 1.0f);
  const auto other_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 1.702f);

  EXPECT_NE(unit_alpha.ToString(), other_alpha.ToString())
      << "QuickGelu drops the multiply at alpha == 1, which is different shader text";

  // All non-unit alphas emit the same code, so they must still share one entry.
  const auto another_alpha = MakeActivation(webgpu::ActivationKind::QuickGelu, 0.5f);
  EXPECT_EQ(other_alpha.ToString(), another_alpha.ToString());
}

// Positive control. Without this the assertions above would still pass if ToString() returned
// something unconditionally unique, which would defeat the cache rather than key it correctly.
TEST(ActivationCacheKeyTest, IdenticalActivationsProduceIdenticalKeys) {
  const auto activation_a = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);
  const auto activation_b = MakeActivation(webgpu::ActivationKind::LeakyRelu, 0.01f);

  EXPECT_EQ(activation_a.ToString(), activation_b.ToString())
      << "identical activations must share a cache entry";
}

// The activation kind must remain part of the key: Relu and Sigmoid carry no parameters at all, so
// only the kind can separate them.
TEST(ActivationCacheKeyTest, DistinctActivationKindsProduceDistinctKeys) {
  const auto relu = MakeActivation(webgpu::ActivationKind::Relu, 0.0f);
  const auto sigmoid = MakeActivation(webgpu::ActivationKind::Sigmoid, 0.0f);

  EXPECT_NE(relu.ToString(), sigmoid.ToString());
}

}  // namespace test
}  // namespace onnxruntime
