// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/weight_layout_transform.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/weight_layout_transform_cache.h"
#include "core/providers/webgpu/nn/conv.h"  // For TransposeKernel

namespace onnxruntime {
namespace webgpu {

Status TransformWeightLayout(
    ComputeContext& context,
    const Tensor* weight,
    const std::string& weight_name,
    const std::string& format_descriptor,
    WeightLayoutTransformCache& cache,
    /*out*/ const Tensor*& transformed_weight) {
  // Check cache first
  const auto* cached = cache.GetTransformedWeight(weight_name, format_descriptor);
  if (cached != nullptr) {
    transformed_weight = cached;
    return Status::OK();
  }

  // Not in cache, need to transform

  const auto& original_shape = weight->Shape();
  auto num_dims = original_shape.NumDimensions();

  // Dispatch transformation based on format
  Tensor output_tensor;
  if (format_descriptor == "hwio") {
    // For 3D tensors, extend to 4D before transposing
    TensorShape input_shape_for_transpose = original_shape;
    if (num_dims == 3) {
      // Extend OIW [O, I, W] to OIHW [O, I, 1, W]
      TensorShapeVector extended_shape = original_shape.AsShapeVector();
      extended_shape.insert(extended_shape.begin() + 2, 1);  // Insert H=1 at position 2
      input_shape_for_transpose = TensorShape(extended_shape);
    }

    // Use existing TransposeKernel: OIHW [O,I,H,W] -> HWIO [H,W,I,O]
    // Permutation: [2, 3, 1, 0] means output[i] = input[perm[i]]
    // TransposeKernel creates the output tensor internally
    const InlinedVector<size_t> perm = {2, 3, 1, 0};
    ORT_RETURN_IF_ERROR(TransposeKernel(context, weight, input_shape_for_transpose,
                                        &output_tensor, perm));
  } else {
    // Add more format implementations here
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Format not yet implemented: ", format_descriptor);
  }

  // Add to cache
  cache.AddTransformedWeight(weight_name, format_descriptor, std::move(output_tensor));

  // Return cached tensor
  const auto* cached_result = cache.GetTransformedWeight(weight_name, format_descriptor);
  ORT_ENFORCE(cached_result != nullptr, "Failed to cache transformed weight");
  transformed_weight = cached_result;

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
