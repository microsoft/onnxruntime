// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/mean_variance_normalization.h"

#include <algorithm>
#include <optional>

#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {

namespace {
InlinedVector<int64_t> GetAxesFromAttribute(const OpKernelInfo& info) {
  const auto axes = info.GetAttrsOrDefault<int64_t>("axes", {0, 2, 3});
  // TODO handle across_channels
  return InlinedVector<int64_t>(axes.begin(), axes.end());
}

InlinedVector<int64_t> NormalizeAxes(gsl::span<const int64_t> axes, size_t rank) {
  InlinedVector<int64_t> normalized_axes{};
  normalized_axes.reserve(axes.size());
  std::copy_if(axes.begin(), axes.end(), std::back_inserter(normalized_axes),
               [rank](int64_t axis) { return IsAxisInRange(axis, static_cast<int64_t>(rank)); });
  std::transform(normalized_axes.begin(), normalized_axes.end(), normalized_axes.begin(),
                 [rank](int64_t axis) { return HandleNegativeAxis(axis, static_cast<int64_t>(rank)); });
  std::sort(normalized_axes.begin(), normalized_axes.end());
  normalized_axes.erase(std::unique(normalized_axes.begin(), normalized_axes.end()), normalized_axes.end());
  return normalized_axes;
}

std::optional<InlinedVector<size_t>> GetTransposePermutationIfNeeded(gsl::span<const int64_t> normalized_axes, size_t rank) {
  // we need to transpose if anything other than the trailing axes are specified
  // assume normalized_axes is sorted with unique values

  const bool is_transpose_needed = [&]() {
    for (size_t i = 0, num_axes = normalized_axes.size(); i < num_axes; ++i) {
      if (static_cast<size_t>(normalized_axes[i]) != rank - num_axes + i) {
        return true;
      }
    }
    return false;
  }();

  if (!is_transpose_needed) {
    return std::nullopt;
  }

  // permutation of [ { unspecified axes }, { specified axes } ]
  InlinedVector<size_t> permutation{};
  permutation.reserve(rank);

  auto specified_axis_it = normalized_axes.begin();
  for (size_t axis = 0; axis < rank; ++axis) {
    if (specified_axis_it != normalized_axes.end() &&
        axis == static_cast<size_t>(*specified_axis_it)) {
      // skip specified axis for now, add them all to the end later
      ++specified_axis_it;
    } else {
      // add unspecified axis
      permutation.push_back(axis);
    }
  }

  // add all specified axes
  std::transform(normalized_axes.begin(), normalized_axes.end(), std::back_inserter(permutation),
                 [](int64_t axis) { return static_cast<size_t>(axis); });

  return permutation;
}
}  // namespace

class MeanVarianceNormalization : public OpKernel {
 public:
  MeanVarianceNormalization(const OpKernelInfo& info)
      : OpKernel{info},
        normalize_variance_(info.GetAttrOrDefault<int64_t>("normalize_variance", int64_t{1}) == int64_t{1}),
        axes_{GetAxesFromAttribute(info)} {
  }

  Status Compute(OpKernelContext* context) const override {
    // general idea:
    // - transpose to partition into [unspecified axes, specified axes]
    // - do normalization on inner dimensions
    // - transpose back
    const auto& input = context->RequiredInput<Tensor>(0);

    const auto rank = input.Shape().GetDims().size();

    const auto normalized_axes = NormalizeAxes(axes_, rank);

    const auto transpose_permutation = GetTransposePermutationIfNeeded(normalized_axes, rank);
    const bool is_transpose_required = transpose_permutation.has_value();

    // intermediate tensors if transposing is necessary
    Tensor transposed_input;
    Tensor transposed_result;

    if (is_transpose_required) {
      AllocatorPtr alloc;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

      InlinedVector<int64_t> transposed_dims{};
      transposed_dims.reserve(rank);
      std::transform(transpose_permutation->begin(), transpose_permutation->end(), std::back_inserter(transposed_dims),
                     [&input_shape = input.Shape().GetDims()](size_t axis) { return input_shape[axis]; });
      const TensorShape transposed_shape = TensorShape::FromExistingBuffer(transposed_dims);

      transposed_input = Tensor{input.DataType(), transposed_shape, alloc};

      ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(*transpose_permutation, input, transposed_input));

      transposed_output = Tensor{input.DataType(), transposed_shape, alloc};
    }
  }

 private:
  const bool normalize_variance_;
  const InlinedVector<int64_t> axes_;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MeanVarianceNormalization,
    9,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization_1<float>);

ONNX_CPU_OPERATOR_KERNEL(
    MeanVarianceNormalization,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization_1<float>);

}  // namespace onnxruntime
