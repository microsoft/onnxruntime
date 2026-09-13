// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <limits>

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"
#include "contrib_ops/cuda/bert/unfused_attention.h"

namespace onnxruntime {
namespace test {

using contrib::cuda::GetGQACompleteWorkspaceRecipe;
using contrib::cuda::GetGQAMemoryEfficientWorkspaceRecipe;
using contrib::cuda::GetGQAUnfusedWorkspaceRecipe;
using contrib::cuda::GQABackend;
using contrib::cuda::GQAConcreteRoute;
using contrib::cuda::GQAPreprocessMode;
using contrib::cuda::GQAWorkspaceError;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::GQAXqaConfig;
using contrib::cuda::ValidateGQACompleteWorkspaceRecipe;
using contrib::cuda::ValidateGQAMemoryEfficientWorkspaceRecipe;
using contrib::cuda::ValidateGQAUnfusedWorkspaceRecipe;

namespace {

GQAWorkspaceProblem Problem() {
  GQAWorkspaceProblem problem;
  problem.qkv_element_size = 2;
  problem.cache_element_size = 2;
  problem.batch_size = 2;
  problem.sequence_length = 3;
  problem.num_heads = 4;
  problem.kv_num_heads = 2;
  problem.head_size = 64;
  problem.present_kv_cache_capacity = 5;
  return problem;
}

GQAWorkspaceProblem DecodeProblem() {
  auto problem = Problem();
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.num_heads = 8;
  problem.kv_num_heads = 2;
  problem.present_kv_cache_capacity = 512;
  return problem;
}

GQAXqaConfig XqaConfig() {
  GQAXqaConfig config;
  config.device_major = 8;
  config.device_minor = 0;
  config.multi_processor_count = 80;
  return config;
}

}  // namespace

TEST(GroupQueryAttentionMeaWorkspaceTest, HandCalculatedGqaExpansionUsesEffectiveCapacity) {
  const auto result = GetGQAMemoryEfficientWorkspaceRecipe(Problem(), 5);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  const auto& recipe = result.recipe;
  // sizeof(T)*B*Nq*C*H = 2*2*4*5*64 = 5120, separately for K and V.
  EXPECT_EQ(recipe.expanded_key_offset_bytes, 0U);
  EXPECT_EQ(recipe.expanded_key_bytes, 5120U);
  EXPECT_EQ(recipe.expanded_value_offset_bytes, 5120U);
  EXPECT_EQ(recipe.expanded_value_bytes, 5120U);
  EXPECT_EQ(recipe.output_accumulator_bytes, 0U);
  EXPECT_EQ(recipe.total_backend_bytes, 10240U);
  EXPECT_TRUE(ValidateGQAMemoryEfficientWorkspaceRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionMeaWorkspaceTest, MhaHeadsAvoidExpansionAndLargeHeadsUseFp32Accumulator) {
  auto problem = Problem();
  problem.kv_num_heads = problem.num_heads;
  auto result = GetGQAMemoryEfficientWorkspaceRecipe(problem, 5);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.total_backend_bytes, 0U);

  problem.head_size = 256;
  result = GetGQAMemoryEfficientWorkspaceRecipe(problem, 5);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.expanded_key_bytes, 0U);
  // sizeof(float)*B*S*N*H = 4*2*3*4*256 = 24576.
  EXPECT_EQ(result.recipe.output_accumulator_bytes, 24576U);
  EXPECT_EQ(result.recipe.output_accumulator_offset_bytes, 0U);
  EXPECT_EQ(result.recipe.total_backend_bytes, 24576U);
}

TEST(GroupQueryAttentionMeaWorkspaceTest, CompleteRouteUsesStagedCapacityAndIncludesPreprocessOnce) {
  auto problem = Problem();
  problem.is_windowed_kv_cache = true;
  problem.is_packed_qkv = true;
  GQAConcreteRoute route;
  route.backend = GQABackend::MemoryEfficient;
  route.preparation.preprocess_mode = GQAPreprocessMode::MemoryEfficient;

  const auto result = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  const auto& recipe = result.recipe;
  EXPECT_EQ(recipe.preparation.effective_kv_cache_capacity, 8);
  // Q + K + V = 3072 + 1536 + 1536.
  EXPECT_EQ(recipe.preparation.qkv_preprocess_bytes, 6144U);
  EXPECT_EQ(recipe.memory_efficient.expanded_key_bytes, 8192U);
  EXPECT_EQ(recipe.memory_efficient.expanded_value_bytes, 8192U);
  EXPECT_EQ(recipe.backend_offset_bytes % 256, 0U);
  EXPECT_GE(recipe.backend_offset_bytes, recipe.preparation.total_preparation_bytes);
  EXPECT_EQ(recipe.total_workspace_bytes,
            recipe.backend_offset_bytes + recipe.backend_bytes);
  EXPECT_TRUE(ValidateGQACompleteWorkspaceRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionUnfusedWorkspaceTest, HandCalculatedCombinedAllocationUsesRuntimeHeadSize) {
  const auto result = GetGQAUnfusedWorkspaceRecipe(Problem(), 5);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  const auto& recipe = result.recipe;
  // GQA runtime requires H_v == H. Q and Y are each
  // align(2*2*4*3*64,256)=3072.
  EXPECT_EQ(recipe.q_bnsh_offset_bytes, 0U);
  EXPECT_EQ(recipe.q_bnsh_bytes, 3072U);
  EXPECT_EQ(recipe.y_bnsh_offset_bytes, 3072U);
  EXPECT_EQ(recipe.y_bnsh_bytes, 3072U);
  // QK and softmax: align(4*2*4*3*5,256)=512 each.
  EXPECT_EQ(recipe.qk_offset_bytes, 6144U);
  EXPECT_EQ(recipe.qk_bytes, 512U);
  EXPECT_EQ(recipe.softmax_offset_bytes, 6656U);
  EXPECT_EQ(recipe.softmax_bytes, 512U);
  EXPECT_EQ(recipe.total_backend_bytes, 7168U);
  EXPECT_TRUE(ValidateGQAUnfusedWorkspaceRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionUnfusedWorkspaceTest, MatchesExistingWorkspaceHelperAcrossFiniteDomain) {
  for (int batch : {1, 2, 7}) {
    for (int heads : {1, 4, 16}) {
      for (int query_length : {1, 3, 64}) {
        for (int kv_length : {1, 5, 257}) {
          auto problem = Problem();
          problem.batch_size = batch;
          problem.num_heads = heads;
          problem.kv_num_heads = 1;
          problem.sequence_length = query_length;
          const auto recipe =
              GetGQAUnfusedWorkspaceRecipe(problem, kv_length);
          ASSERT_TRUE(recipe.status.IsOK()) << recipe.status.message;
          const size_t expected =
              contrib::cuda::GetUnfusedAttentionWorkspaceSize(
                  batch, heads, query_length, kv_length);
          EXPECT_EQ(
              recipe.recipe.qk_bytes + recipe.recipe.softmax_bytes, expected);
        }
      }
    }
  }
}

TEST(GroupQueryAttentionUnfusedWorkspaceTest, CompleteRootAlignsAndDoesNotOverlapPreparation) {
  auto problem = Problem();
  problem.do_rotary = true;
  GQAConcreteRoute route;
  route.backend = GQABackend::Unfused;
  route.preparation.preprocess_mode = GQAPreprocessMode::Unfused;
  route.unfused.total_sequence_length = 5;

  const auto result = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  const auto& recipe = result.recipe;
  EXPECT_EQ(recipe.preparation.qkv_preprocess_bytes, 3072U);
  EXPECT_EQ(recipe.backend_offset_bytes % 256, 0U);
  EXPECT_GE(recipe.backend_offset_bytes, recipe.preparation.total_preparation_bytes);
  EXPECT_EQ(recipe.total_workspace_bytes,
            recipe.backend_offset_bytes + recipe.unfused.total_backend_bytes);
  EXPECT_TRUE(ValidateGQACompleteWorkspaceRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionCompleteWorkspaceTest, FlashRootIncludesPreparationExactlyOnce) {
  auto problem = DecodeProblem();
  problem.num_heads = 2;
  problem.kv_num_heads = 2;
  problem.do_rotary = true;
  GQAConcreteRoute route;
  route.backend = GQABackend::Flash;
  route.preparation.preprocess_mode = GQAPreprocessMode::Flash;
  route.flash.total_sequence_length = 4096;
  route.flash.multi_processor_count = 108;

  const auto result = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.preparation.qkv_preprocess_bytes, 256U);
  EXPECT_EQ(result.recipe.backend_offset_bytes % 256, 0U);
  EXPECT_GE(
      result.recipe.backend_offset_bytes,
      result.recipe.preparation.total_preparation_bytes);
  EXPECT_EQ(
      result.recipe.total_workspace_bytes,
      result.recipe.backend_offset_bytes + result.recipe.flash.total_backend_bytes);
  EXPECT_TRUE(ValidateGQACompleteWorkspaceRecipe(result.recipe).IsOK());
}

TEST(GroupQueryAttentionCompleteWorkspaceTest, XqaRootKeepsPreparationQAndXqaRopeBuffers) {
  auto problem = DecodeProblem();
  problem.do_rotary = true;
  GQAConcreteRoute route;
  route.backend = GQABackend::Xqa;
  route.preparation.preprocess_mode = GQAPreprocessMode::Xqa;
  route.xqa = XqaConfig();

  const auto result = GetGQACompleteWorkspaceRecipe(problem, route);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.preparation.qkv_preprocess_bytes, 1024U);
  EXPECT_EQ(result.recipe.xqa.rotary_q_bytes, 1024U);
  EXPECT_EQ(result.recipe.xqa.rotary_k_bytes, 256U);
  EXPECT_EQ(result.recipe.backend_offset_bytes, 1280U);
  EXPECT_EQ(result.recipe.total_workspace_bytes,
            result.recipe.backend_offset_bytes + result.recipe.backend_bytes);
  EXPECT_TRUE(ValidateGQACompleteWorkspaceRecipe(result.recipe).IsOK());
}

TEST(GroupQueryAttentionCompleteWorkspaceTest, RejectsContradictoryFlashFastDecodeFacts) {
  auto problem = DecodeProblem();
  problem.num_heads = 2;
  problem.kv_num_heads = 2;
  GQAConcreteRoute route;
  route.backend = GQABackend::Flash;
  route.preparation.preprocess_mode = GQAPreprocessMode::Flash;
  route.preparation.use_flash_attention_fast_decode = true;
  route.flash.fast_decode = false;
  route.flash.total_sequence_length = 512;
  route.flash.multi_processor_count = 80;

  EXPECT_EQ(GetGQACompleteWorkspaceRecipe(problem, route).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionBackendWorkspaceTest, CheckedOverflowReturnsNoPartialRecipe) {
  auto problem = Problem();
  problem.batch_size = std::numeric_limits<int32_t>::max();
  problem.sequence_length = std::numeric_limits<int32_t>::max();
  problem.num_heads = std::numeric_limits<int32_t>::max();
  problem.kv_num_heads = 1;
  problem.head_size = std::numeric_limits<int32_t>::max() - 7;

  auto unfused = GetGQAUnfusedWorkspaceRecipe(
      problem, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(unfused.status.error, GQAWorkspaceError::Overflow);
  EXPECT_EQ(unfused.recipe.total_backend_bytes, 0U);

  auto mea = GetGQAMemoryEfficientWorkspaceRecipe(
      problem, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(mea.status.error, GQAWorkspaceError::Overflow);
  EXPECT_EQ(mea.recipe.total_backend_bytes, 0U);
}

TEST(GroupQueryAttentionCompleteWorkspaceTest, CudnnIsUnavailableRatherThanZero) {
  GQAConcreteRoute route;
  route.backend = GQABackend::Cudnn;
  const auto result = GetGQACompleteWorkspaceRecipe(Problem(), route);
  EXPECT_EQ(result.status.error, GQAWorkspaceError::Unavailable);
  EXPECT_EQ(result.recipe.total_workspace_bytes, 0U);
}

TEST(GroupQueryAttentionCompleteWorkspaceTest, RejectsMismatchedBackendAndPreparation) {
  GQAConcreteRoute route;
  route.backend = GQABackend::Flash;
  route.preparation.preprocess_mode = GQAPreprocessMode::Unfused;
  route.flash.total_sequence_length = 5;
  route.flash.multi_processor_count = 80;
  EXPECT_EQ(GetGQACompleteWorkspaceRecipe(Problem(), route).status.error,
            GQAWorkspaceError::InvalidArgument);
}

}  // namespace test
}  // namespace onnxruntime
