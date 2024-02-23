// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/mean_variance_normalization.h"

#include <algorithm>
#include <optional>

#include "core/common/gsl.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace {
InlinedVector<int64_t> GetAxesFromAttribute(const OpKernelInfo& info) {
  // legacy attribute that affects default axes value
  const bool across_channels = info.GetAttrOrDefault<int64_t>("across_channels", int64_t{0}) == int64_t{1};

  const auto default_axes = across_channels
                                ? std::vector<int64_t>{0, 1, 2, 3}
                                : std::vector<int64_t>{0, 2, 3};

  const auto axes = info.GetAttrsOrDefault<int64_t>("axes", default_axes);

  return InlinedVector<int64_t>(axes.begin(), axes.end());
}

// Drop out of range, make non-negative, sort, and make unique.
InlinedVector<size_t> NormalizeAxes(gsl::span<const int64_t> axes, size_t rank) {
  InlinedVector<size_t> normalized_axes{};
  normalized_axes.reserve(axes.size());
  for (int64_t axis : axes) {
    if (IsAxisInRange(axis, static_cast<int64_t>(rank))) {
      normalized_axes.push_back(
          static_cast<size_t>(HandleNegativeAxis(axis, static_cast<int64_t>(rank))));
    }
  }
  std::sort(normalized_axes.begin(), normalized_axes.end());
  normalized_axes.erase(std::unique(normalized_axes.begin(), normalized_axes.end()), normalized_axes.end());
  return normalized_axes;
}

std::optional<InlinedVector<size_t>> GetTransposePermutationIfNeeded(gsl::span<const size_t> normalized_axes, size_t rank) {
  // We need to transpose if anything other than the trailing axes are specified.
  // Assume `normalized_axes` is sorted with unique values.
  const bool is_transpose_needed = [&]() {
    for (size_t i = 0, num_axes = normalized_axes.size(); i < num_axes; ++i) {
      if (normalized_axes[i] != rank - num_axes + i) {
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
        axis == *specified_axis_it) {
      // skip specified axis for now, add them all to the end later
      ++specified_axis_it;
    } else {
      // add unspecified axis
      permutation.push_back(axis);
    }
  }

  // add all specified axes
  permutation.insert(permutation.end(), normalized_axes.begin(), normalized_axes.end());

  return permutation;
}

// Given an M x N array where N is the inner dimension, compute normalized quantities for M sets of N values.
// X is the input and Y is the output.
Status ComputeMeanVarianceNormalization2D(size_t M, size_t N,
                                          gsl::span<const float> X, gsl::span<float> Y,
                                          bool normalize_variance) {
  ORT_RETURN_IF_NOT(X.size() == M * N && X.size() == Y.size(), "X and Y must both have M * N elements.");

  const auto idx_M = narrow<Eigen::Index>(M), idx_N = narrow<Eigen::Index>(N);
  // Note: Eigen arrays have column-major storage by default, so we specify N rows x M columns.
  ConstEigenArrayMap<float> X_array(X.data(), idx_N, idx_M);
  EigenArrayMap<float> Y_array(Y.data(), idx_N, idx_M);

  // for each column, compute Y = X - E[X]
  Y_array = X_array.rowwise() - X_array.colwise().mean();

  if (normalize_variance) {
    // for each column, compute Y' = (X - E[X]) / ( E[ (X - E[X])^2 ] )^(1/2)
    //   we start with Y = X - E[X],
    //   so Y' = Y / ( E[ Y^2 ] )^(1/2)
    Y_array = (Y_array.rowwise() / (Y_array.square().colwise().mean().sqrt())).eval();
  }

  return Status::OK();
}

InlinedVector<size_t> InvertPerm(gsl::span<const size_t> perm) {
  InlinedVector<size_t> inverted_perm{};
  inverted_perm.resize(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inverted_perm[perm[i]] = i;
  }
  return inverted_perm;
}
}  // namespace

MeanVarianceNormalization::MeanVarianceNormalization(const OpKernelInfo& info)
    : OpKernel{info},
      normalize_variance_{
          // legacy attribute
          info.GetAttrOrDefault("normalize_variance", int64_t{1}) == int64_t{1}},
      axes_{GetAxesFromAttribute(info)} {
}

Status MeanVarianceNormalization::Compute(OpKernelContext* context) const {
  const auto& input = context->RequiredInput<Tensor>(0);
  const auto& input_shape = input.Shape();

  Tensor& output = context->RequiredOutput(0, input_shape);

  // approach for normalizing values across arbitrary dimensions:
  // - transpose to [unspecified axes, specified axes]
  // - do normalization of inner dimension values
  // - transpose back

  const auto rank = input_shape.GetDims().size();

  const auto normalized_axes = NormalizeAxes(axes_, rank);

  // The ONNX spec doesn't specify what to do if no axes are specified so we won't try to do anything.
  ORT_RETURN_IF(normalized_axes.empty(), "No valid axes are specified. This is not handled now.");

  if (input_shape.Size() == 0) {
    return Status::OK();
  }

  const auto transpose_permutation = GetTransposePermutationIfNeeded(normalized_axes, rank);
  const bool is_transpose_required = transpose_permutation.has_value();

  // intermediate tensors if transposing is necessary
  Tensor transposed_input;
  Tensor transposed_result;

  TensorShape compute_shape = input_shape;

  if (is_transpose_required) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    InlinedVector<int64_t> transposed_dims{};
    transposed_dims.reserve(rank);
    std::transform(transpose_permutation->begin(), transpose_permutation->end(), std::back_inserter(transposed_dims),
                   [input_dims = input.Shape().GetDims()](size_t axis) { return input_dims[axis]; });
    compute_shape = TensorShape(transposed_dims);

    transposed_input = Tensor{input.DataType(), compute_shape, alloc};

    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(*transpose_permutation, input, transposed_input));

    transposed_result = Tensor{input.DataType(), compute_shape, alloc};
  }

  const size_t num_unspecified_axes = rank - normalized_axes.size();
  const size_t M = narrow<size_t>(compute_shape.SizeToDimension(num_unspecified_axes)),
               N = narrow<size_t>(compute_shape.SizeFromDimension(num_unspecified_axes));

  const gsl::span<const float> X = is_transpose_required
                                       ? transposed_input.DataAsSpan<float>()
                                       : input.DataAsSpan<float>();
  const gsl::span<float> Y = is_transpose_required
                                 ? transposed_result.MutableDataAsSpan<float>()
                                 : output.MutableDataAsSpan<float>();

  ORT_RETURN_IF_ERROR(ComputeMeanVarianceNormalization2D(M, N, X, Y, normalize_variance_));

  if (is_transpose_required) {
    const auto inverted_permutation = InvertPerm(*transpose_permutation);
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(inverted_permutation, transposed_result, output));
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MeanVarianceNormalization,
    9,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization);

ONNX_CPU_OPERATOR_KERNEL(
    MeanVarianceNormalization,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization);

}  // namespace onnxruntime
