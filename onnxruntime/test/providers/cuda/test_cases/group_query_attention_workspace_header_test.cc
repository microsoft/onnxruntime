// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Keep this graph-free header first and this translation unit free of ORT and
// test-framework headers. Both the in-tree CUDA target and the plugin-internal
// target compile it.
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_bounds.h"

#include <type_traits>

namespace onnxruntime {
namespace test {

using contrib::cuda::GetGQAEffectiveWorkspaceKvLength;
using contrib::cuda::GQACompleteWorkspaceRecipe;
using contrib::cuda::GQAConcreteRoute;
using contrib::cuda::GQAFlashWorkspaceRecipe;
using contrib::cuda::GQAMemoryEfficientWorkspaceRecipe;
using contrib::cuda::GQAPreparationRecipe;
using contrib::cuda::GQAPreparationRoute;
using contrib::cuda::GQAUnfusedWorkspaceRecipe;
using contrib::cuda::GQAWorkspaceAggregate;
using contrib::cuda::GQAWorkspaceBounds;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::GQAWorkspaceStatus;
using contrib::cuda::GQAXqaWorkspaceRecipe;
using contrib::cuda::IsSupportedGQAXqaGroupSize;
using contrib::cuda::IsSupportedGQAXqaHeadSize;

static_assert(std::is_trivially_copyable_v<GQAWorkspaceProblem>);
static_assert(std::is_trivially_copyable_v<GQAWorkspaceBounds>);
static_assert(std::is_trivially_copyable_v<GQAWorkspaceAggregate>);
static_assert(std::is_trivially_copyable_v<GQAPreparationRoute>);
static_assert(std::is_trivially_copyable_v<GQAPreparationRecipe>);
static_assert(std::is_trivially_copyable_v<GQAConcreteRoute>);
static_assert(std::is_trivially_copyable_v<GQAXqaWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAFlashWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAMemoryEfficientWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAUnfusedWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQACompleteWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAWorkspaceStatus>);
static_assert(IsSupportedGQAXqaHeadSize(64));
static_assert(!IsSupportedGQAXqaHeadSize(96));
static_assert(IsSupportedGQAXqaGroupSize(5, false));
static_assert(!IsSupportedGQAXqaGroupSize(5, true));
static_assert(GetGQAEffectiveWorkspaceKvLength(257, 8, true) == 8);
static_assert(GetGQAEffectiveWorkspaceKvLength(5, 8, true) == 5);
static_assert(GetGQAEffectiveWorkspaceKvLength(257, 8, false) == 257);

void CompileGroupQueryAttentionWorkspaceHeaderInIsolation() {
  GQAWorkspaceProblem problem;
  problem.requires_separate_past_buffer = true;
  problem.past_kv_cache_capacity = 1;
  GQAWorkspaceBounds bounds;
  GQAPreparationRoute route;
  GQAPreparationRecipe recipe;
  GQAConcreteRoute concrete_route;
  GQACompleteWorkspaceRecipe complete_recipe;
  (void)problem;
  (void)bounds;
  (void)route;
  (void)recipe;
  (void)concrete_route;
  (void)complete_recipe;
}

}  // namespace test
}  // namespace onnxruntime
