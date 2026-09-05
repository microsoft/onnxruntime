// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Keep this graph-free header first and this translation unit free of ORT and
// test-framework headers. Both the in-tree CUDA target and the plugin-internal
// target compile it.
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#include <type_traits>

namespace onnxruntime {
namespace test {

using contrib::cuda::GQACompleteWorkspaceRecipe;
using contrib::cuda::GQAConcreteRoute;
using contrib::cuda::GQAFlashWorkspaceRecipe;
using contrib::cuda::GQAMemoryEfficientWorkspaceRecipe;
using contrib::cuda::GQAPreparationRecipe;
using contrib::cuda::GQAPreparationRoute;
using contrib::cuda::GQAUnfusedWorkspaceRecipe;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::GQAWorkspaceStatus;
using contrib::cuda::GQAXqaWorkspaceRecipe;

static_assert(std::is_trivially_copyable_v<GQAWorkspaceProblem>);
static_assert(std::is_trivially_copyable_v<GQAPreparationRoute>);
static_assert(std::is_trivially_copyable_v<GQAPreparationRecipe>);
static_assert(std::is_trivially_copyable_v<GQAConcreteRoute>);
static_assert(std::is_trivially_copyable_v<GQAXqaWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAFlashWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAMemoryEfficientWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAUnfusedWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQACompleteWorkspaceRecipe>);
static_assert(std::is_trivially_copyable_v<GQAWorkspaceStatus>);

void CompileGroupQueryAttentionWorkspaceHeaderInIsolation() {
  GQAWorkspaceProblem problem;
  GQAPreparationRoute route;
  GQAPreparationRecipe recipe;
  GQAConcreteRoute concrete_route;
  GQACompleteWorkspaceRecipe complete_recipe;
  (void)problem;
  (void)route;
  (void)recipe;
  (void)concrete_route;
  (void)complete_recipe;
}

}  // namespace test
}  // namespace onnxruntime
