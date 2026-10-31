// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Keep this header first and this translation unit free of ORT and test-framework
// headers. Both the in-tree CUDA target and the plugin-internal target compile it.
#include "contrib_ops/cuda/bert/packed_attention_workspace.h"

#include <type_traits>

namespace onnxruntime {
namespace test {

using contrib::cuda::GetPackedAttentionQkvMaterializationIndexWidth;
using contrib::cuda::PackedAttentionProblem;
using contrib::cuda::PackedAttentionQkvMaterializationIndexWidth;
using contrib::cuda::PackedAttentionWorkspaceRecipe;
using contrib::cuda::PackedMultiHeadAttentionProblem;

static_assert(std::is_trivially_copyable_v<PackedAttentionProblem>);
static_assert(std::is_trivially_copyable_v<PackedMultiHeadAttentionProblem>);
static_assert(std::is_trivially_copyable_v<PackedAttentionWorkspaceRecipe>);
static_assert(noexcept(GetPackedAttentionQkvMaterializationIndexWidth(4, 4)));
static_assert(GetPackedAttentionQkvMaterializationIndexWidth(4, 4) ==
              PackedAttentionQkvMaterializationIndexWidth::Vector4);
static_assert(GetPackedAttentionQkvMaterializationIndexWidth(2, 2) ==
              PackedAttentionQkvMaterializationIndexWidth::Vector2);
static_assert(GetPackedAttentionQkvMaterializationIndexWidth(1, 1) ==
              PackedAttentionQkvMaterializationIndexWidth::Scalar);

void CompilePackedAttentionWorkspaceHeaderInIsolation() {
  PackedAttentionProblem packed_attention;
  PackedMultiHeadAttentionProblem packed_mha;
  PackedAttentionWorkspaceRecipe recipe;
  (void)packed_attention;
  (void)packed_mha;
  (void)recipe;
}

}  // namespace test
}  // namespace onnxruntime
