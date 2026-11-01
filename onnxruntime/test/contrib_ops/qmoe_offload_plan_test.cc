// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>

#include "contrib_ops/cuda/moe/qmoe_offload_plan.h"
#include "core/framework/tensor_shape.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace qmoe = contrib::cuda::qmoe;

namespace {

constexpr int64_t kScaleGroup = 16;

// Per-expert scale bytes of one NVFP4 layer: FC1 [2*I, H/16] and FC2 [H, I/16].
uint64_t Fc1ScaleBytes(int64_t experts, int64_t hidden, int64_t inter) {
  return static_cast<uint64_t>(experts * 2 * inter * (hidden / kScaleGroup));
}

uint64_t Fc2ScaleBytes(int64_t experts, int64_t hidden, int64_t inter) {
  return static_cast<uint64_t>(experts * hidden * (inter / kScaleGroup));
}

}  // namespace

TEST(QMoEOffloadPlanTest, StandardFp4BankShapesResolveHiddenAndIntermediateSizes) {
  // FC1 [E, 2*I, H/2], FC2 [E, H, I/2]. Reading these shapes as the legacy layout would give H=128 and I=256.
  const TensorShape fc1({4, 128, 128});
  const TensorShape fc2({4, 256, 32});
  const auto dims = qmoe::ResolveNvfp4BankDims(fc1, fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                               Fc2ScaleBytes(4, 256, 64));
  ASSERT_TRUE(dims.valid);
  EXPECT_EQ(dims.hidden, 256);
  EXPECT_EQ(dims.inter, 64);
}

TEST(QMoEOffloadPlanTest, LegacyFp4BankShapesResolveHiddenAndIntermediateSizes) {
  // FC1 [E, H, I], FC2 [E, I, H/2].
  const TensorShape fc1({4, 256, 64});
  const TensorShape fc2({4, 64, 128});
  const auto dims = qmoe::ResolveNvfp4BankDims(fc1, fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                               Fc2ScaleBytes(4, 256, 64));
  ASSERT_TRUE(dims.valid);
  EXPECT_EQ(dims.hidden, 256);
  EXPECT_EQ(dims.inter, 64);
}

TEST(QMoEOffloadPlanTest, RejectsShapesThatMatchNeitherLayout) {
  const TensorShape fc1({4, 128, 128});
  const TensorShape transposed_fc2({4, 32, 256});
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, transposed_fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);

  const TensorShape rank2({128, 128});
  const TensorShape fc2({4, 256, 32});
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(rank2, fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);

  const TensorShape other_experts({3, 256, 32});
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, other_experts, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);
}

TEST(QMoEOffloadPlanTest, RejectsScaleBanksThatDoNotHoldExactlyOneSliceForEveryExpert) {
  const TensorShape fc1({4, 128, 128});
  const TensorShape fc2({4, 256, 32});
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64) - 1,
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, fc2, kScaleGroup, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(5, 256, 64))
                   .valid);
}

TEST(QMoEOffloadPlanTest, RejectsSizesThatAreNotWholeScaleGroups) {
  const TensorShape fc1({4, 128, 128});
  const TensorShape fc2({4, 256, 32});
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, fc2, /*scale_group=*/48, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, fc2, /*scale_group=*/0, Fc1ScaleBytes(4, 256, 64),
                                          Fc2ScaleBytes(4, 256, 64))
                   .valid);
}

// The raw dimensions are multiplied (fc2_dim2 * 2, fc2_dim2 * 4, fc1_dim2 * 2) to detect the layout, so extreme values
// must be rejected before any product is formed. Under UBSan the first case below reported signed overflow.
TEST(QMoEOffloadPlanTest, RejectsExtremeDimensionsWithoutOverflowingTheLayoutChecks) {
  constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
  const auto fc1_scale_bytes = Fc1ScaleBytes(1, 128, 64);
  const auto fc2_scale_bytes = Fc2ScaleBytes(1, 128, 64);

  // FC2.dim2 large enough that fc2_dim2 * 2 (and * 4) overflows int64.
  const TensorShape fc1({1, 128, 128});
  for (const int64_t huge : {kInt64Max, kInt64Max / 2 + 1, kInt64Max / 4 + 1}) {
    EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, TensorShape({1, 256, huge}), kScaleGroup, fc1_scale_bytes,
                                            fc2_scale_bytes)
                     .valid)
        << "fc2_dim2=" << huge;
  }

  // FC1.dim2 large enough that fc1_dim2 * 2 overflows once FC1.dim1 == 4 * FC2.dim2 selects the standard layout.
  const TensorShape fc2({1, 256, 32});
  for (const int64_t huge : {kInt64Max, kInt64Max / 2 + 1}) {
    EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({1, 128, huge}), fc2, kScaleGroup, fc1_scale_bytes,
                                            fc2_scale_bytes)
                     .valid)
        << "fc1_dim2=" << huge;
  }

  // 4 * (2^62 + 8) wraps to 32 in 64-bit arithmetic, so without the early limit this impossible FC2 shape is read as
  // the standard layout with H=128 and I=16 and accepted even when no overflow checker is active.
  constexpr int64_t kWrapsToStandardLayout = (int64_t{1} << 62) + 8;
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({1, 32, 64}), TensorShape({1, 128, kWrapsToStandardLayout}),
                                          kScaleGroup, Fc1ScaleBytes(1, 128, 16), Fc2ScaleBytes(1, 128, 16))
                   .valid);

  // Dimensions that are not multiplied are rejected too, as are expert counts beyond int32.
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({1, kInt64Max, 128}), fc2, kScaleGroup, fc1_scale_bytes,
                                          fc2_scale_bytes)
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(fc1, TensorShape({1, kInt64Max, 32}), kScaleGroup, fc1_scale_bytes,
                                          fc2_scale_bytes)
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({kInt64Max, 128, 128}), TensorShape({kInt64Max, 256, 32}),
                                          kScaleGroup, fc1_scale_bytes, fc2_scale_bytes)
                   .valid);
}

// The early dimension limit must not reject a layout that the later int32 checks accept, and must stay in step with them.
TEST(QMoEOffloadPlanTest, LargestInt32HiddenAndIntermediateSizesStillResolve) {
  constexpr int64_t kLargest = (int64_t{1} << 31) - kScaleGroup;  // largest multiple of 16 not above INT32_MAX
  const auto fc1_scale_bytes = Fc1ScaleBytes(1, kLargest, kLargest);
  const auto fc2_scale_bytes = Fc2ScaleBytes(1, kLargest, kLargest);

  // Standard layout: FC1 [1, 2*I, H/2], FC2 [1, H, I/2].
  auto dims = qmoe::ResolveNvfp4BankDims(TensorShape({1, 2 * kLargest, kLargest / 2}),
                                         TensorShape({1, kLargest, kLargest / 2}), kScaleGroup, fc1_scale_bytes,
                                         fc2_scale_bytes);
  ASSERT_TRUE(dims.valid);
  EXPECT_EQ(dims.hidden, kLargest);
  EXPECT_EQ(dims.inter, kLargest);

  // Legacy layout: FC1 [1, H, I], FC2 [1, I, H/2].
  dims = qmoe::ResolveNvfp4BankDims(TensorShape({1, kLargest, kLargest}), TensorShape({1, kLargest, kLargest / 2}),
                                    kScaleGroup, fc1_scale_bytes, fc2_scale_bytes);
  ASSERT_TRUE(dims.valid);
  EXPECT_EQ(dims.hidden, kLargest);
  EXPECT_EQ(dims.inter, kLargest);

  // One scale group above INT32_MAX is rejected in both layouts.
  constexpr int64_t kTooLarge = kLargest + kScaleGroup;
  const auto big_fc1_scale_bytes = Fc1ScaleBytes(1, kTooLarge, kTooLarge);
  const auto big_fc2_scale_bytes = Fc2ScaleBytes(1, kTooLarge, kTooLarge);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({1, 2 * kTooLarge, kTooLarge / 2}),
                                          TensorShape({1, kTooLarge, kTooLarge / 2}), kScaleGroup,
                                          big_fc1_scale_bytes, big_fc2_scale_bytes)
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({1, kTooLarge, kTooLarge}),
                                          TensorShape({1, kTooLarge, kTooLarge / 2}), kScaleGroup,
                                          big_fc1_scale_bytes, big_fc2_scale_bytes)
                   .valid);
}

TEST(QMoEOffloadPlanTest, RejectsZeroOrNegativeExpertCountsAndSupportsInt4AndInt3Banks) {
  // E=0 must be rejected before per-expert division for NVFP4, INT4, and INT3.
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(TensorShape({0, 256, 128}), TensorShape({0, 128, 128}),
                                          /*scale_group=*/32, 0, 0, /*scale_elem_bytes=*/2)
                   .valid);
  EXPECT_FALSE(qmoe::ResolveInt3BankDims(TensorShape({0, 256, 96}), TensorShape({0, 256, 48}),
                                         /*scale_group=*/32, 0, 0)
                   .valid);

  // Valid prepacked INT4 (block_size=32, 2B FP16/BF16 scales) in both legacy [E,H,I]/[E,I,H/2] and standard layouts.
  const uint64_t int4_fc1_s = 8ull * 256 * (256 / 32) * 2;
  const uint64_t int4_fc2_s = 8ull * 256 * (128 / 32) * 2;
  const auto int4_legacy = qmoe::ResolveNvfp4BankDims(
      TensorShape({8, 256, 128}), TensorShape({8, 128, 128}), /*scale_group=*/32, int4_fc1_s, int4_fc2_s, 2);
  ASSERT_TRUE(int4_legacy.valid);
  EXPECT_EQ(int4_legacy.hidden, 256);
  EXPECT_EQ(int4_legacy.inter, 128);

  const auto int4_std = qmoe::ResolveNvfp4BankDims(
      TensorShape({8, 256, 128}), TensorShape({8, 256, 64}), /*scale_group=*/32, int4_fc1_s, int4_fc2_s, 2);
  ASSERT_TRUE(int4_std.valid);
  EXPECT_EQ(int4_std.hidden, 256);
  EXPECT_EQ(int4_std.inter, 128);

  // Valid raw INT3 (block_size in {32, 64, 128}): FC1 [E, 2*I, (H*3)/8], FC2 [E, H, (I*3)/8].
  for (int64_t blk : {32, 64, 128}) {
    const uint64_t int3_fc1_s = static_cast<uint64_t>(8 * 256 * (256 / blk) * 2);
    const uint64_t int3_fc2_s = static_cast<uint64_t>(8 * 256 * (128 / blk) * 2);
    const auto int3_dims = qmoe::ResolveInt3BankDims(
        TensorShape({8, 256, (256 * 3) / 8}), TensorShape({8, 256, (128 * 3) / 8}), blk, int3_fc1_s, int3_fc2_s);
    ASSERT_TRUE(int3_dims.valid) << "blk=" << blk;
    EXPECT_EQ(int3_dims.hidden, 256);
    EXPECT_EQ(int3_dims.inter, 128);
  }
}

TEST(QMoEOffloadPlanTest, Int2CanonicalAndLegacyBanksMatchScaleGeometry) {
  for (int64_t block : {32, 64, 128}) {
    const uint64_t fc1_scales = 8ull * 256 * (256 / block) * 2;
    const uint64_t fc2_scales = 8ull * 256 * (128 / block) * 2;
    for (const auto& fc2 : {TensorShape({8, 128, 64}), TensorShape({8, 256, 32})}) {
      const auto dims = qmoe::ResolveNvfp4BankDims(
          TensorShape({8, 256, 64}), fc2, block, fc1_scales, fc2_scales, 2, /*pack_factor=*/4);
      ASSERT_TRUE(dims.valid);
      EXPECT_EQ(dims.hidden, 256);
      EXPECT_EQ(dims.inter, 128);
      EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(
                       TensorShape({8, 256, 64}), fc2, block, fc1_scales - 1, fc2_scales, 2, 4)
                       .valid);
    }
  }
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(
                   TensorShape({0, 256, 64}), TensorShape({0, 256, 32}), 64, 0, 0, 2, 4)
                   .valid);
  EXPECT_FALSE(qmoe::ResolveNvfp4BankDims(
                   TensorShape({8, 256, 64}), TensorShape({8, 256, 32}), 64, 8192, 4096, 2, 3)
                   .valid);
}

namespace {

qmoe::CrossLayerSite Site(const void* session, int64_t layer, bool doorbell_pending = false) {
  qmoe::CrossLayerSite site;
  site.session = session;
  site.layer_index = layer;
  site.doorbell_pending = doorbell_pending;
  return site;
}

}  // namespace

// The doorbell registry is shared by every session. With two 2-layer sessions registered A0, A1, B0, B1, servicing
// A1 must not stage or publish anything for B0/B1: B may already have classified its routes with the old placement.
TEST(QMoEOffloadPlanTest, CrossLayerPromotionNeverTargetsAnotherSession) {
  int session_a = 0;
  int session_b = 0;
  const auto servicing_a1 = Site(&session_a, 1);

  for (const int64_t layer : {0, 1, 2, 3}) {
    EXPECT_FALSE(qmoe::IsCrossLayerLookaheadTarget(servicing_a1, Site(&session_b, layer)));
    // Whether or not B was held after classification, the other session is never a publication target.
    EXPECT_FALSE(qmoe::CanPublishCrossLayerPromotion(servicing_a1, Site(&session_b, layer, false)));
    EXPECT_FALSE(qmoe::CanPublishCrossLayerPromotion(servicing_a1, Site(&session_b, layer, true)));
  }
}

TEST(QMoEOffloadPlanTest, CrossLayerPromotionTargetsOnlyLaterLayersOfTheServicingSession) {
  int session = 0;
  const auto servicing = Site(&session, 3);

  EXPECT_TRUE(qmoe::IsCrossLayerLookaheadTarget(servicing, Site(&session, 4)));
  EXPECT_TRUE(qmoe::IsCrossLayerLookaheadTarget(servicing, Site(&session, 7)));
  // The servicing layer itself and layers that already ran this step (including the wrap into the next step).
  EXPECT_FALSE(qmoe::IsCrossLayerLookaheadTarget(servicing, Site(&session, 3)));
  EXPECT_FALSE(qmoe::IsCrossLayerLookaheadTarget(servicing, Site(&session, 2)));
  EXPECT_FALSE(qmoe::IsCrossLayerLookaheadTarget(servicing, Site(&session, 0)));
}

TEST(QMoEOffloadPlanTest, CrossLayerPromotionNeedsAKnownSession) {
  // Two nodes without a controller are not "the same session".
  EXPECT_FALSE(qmoe::IsCrossLayerLookaheadTarget(Site(nullptr, 0), Site(nullptr, 1)));
  EXPECT_FALSE(qmoe::CanPublishCrossLayerPromotion(Site(nullptr, 0), Site(nullptr, 1)));
}

TEST(QMoEOffloadPlanTest, CrossLayerPromotionIsNotPublishedOnceTheTargetHasClassified) {
  int session = 0;
  const auto servicing = Site(&session, 1);

  EXPECT_TRUE(qmoe::CanPublishCrossLayerPromotion(servicing, Site(&session, 2, /*doorbell_pending=*/false)));
  // A raised doorbell request means the target already routed with the old placement.
  EXPECT_FALSE(qmoe::CanPublishCrossLayerPromotion(servicing, Site(&session, 2, /*doorbell_pending=*/true)));
}

}  // namespace test
}  // namespace onnxruntime
