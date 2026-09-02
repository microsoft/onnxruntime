// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "einsum_fast_path.h"

#include <limits>

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "einsum_fast_path_kernels.h"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr size_t kFastPathRankLimit = 8;

std::vector<int64_t> GetOutputLabels(const EinsumComputePreprocessor& preprocessor) {
  const auto& label_to_output_axis = preprocessor.GetMappedSubscriptIndicesToOutputindices();
  std::vector<int64_t> output_labels(preprocessor.GetOutputDims().size(), -1);
  for (size_t label = 0; label < label_to_output_axis.size(); ++label) {
    const int64_t output_axis = label_to_output_axis[label];
    if (output_axis >= 0) {
      output_labels[onnxruntime::narrow<size_t>(output_axis)] = onnxruntime::narrow<int64_t>(label);
    }
  }
  return output_labels;
}

bool IsExactSequenceIgnoringSizeOne(gsl::span<const int64_t> actual_labels,
                                    gsl::span<const int64_t> actual_dims,
                                    gsl::span<const int64_t> expected_labels) {
  std::vector<int64_t> filtered_actual;
  filtered_actual.reserve(actual_labels.size());
  for (size_t i = 0; i < actual_labels.size(); ++i) {
    if (actual_dims[i] != 1) {
      filtered_actual.push_back(actual_labels[i]);
    }
  }

  std::vector<int64_t> filtered_expected;
  filtered_expected.reserve(expected_labels.size());
  for (const int64_t label : expected_labels) {
    auto actual_axis = std::find(actual_labels.begin(), actual_labels.end(), label);
    if (actual_axis != actual_labels.end()) {
      const size_t axis = static_cast<size_t>(std::distance(actual_labels.begin(), actual_axis));
      if (actual_dims[axis] != 1) {
        filtered_expected.push_back(label);
      }
    }
  }

  return filtered_actual == filtered_expected;
}

TensorShapeVector BuildBroadcastViewDims(gsl::span<const int64_t> input_labels,
                                         gsl::span<const int64_t> input_dims,
                                         gsl::span<const int64_t> output_labels) {
  TensorShapeVector view_dims(output_labels.size(), 1);
  for (size_t input_axis = 0; input_axis < input_labels.size(); ++input_axis) {
    const auto output_axis = std::find(output_labels.begin(), output_labels.end(), input_labels[input_axis]);
    ORT_ENFORCE(output_axis != output_labels.end());
    view_dims[static_cast<size_t>(std::distance(output_labels.begin(), output_axis))] = input_dims[input_axis];
  }
  return view_dims;
}

bool IsRepeatedLabelShapeValid(gsl::span<const int64_t> labels,
                               gsl::span<const int64_t> dims,
                               size_t num_labels) {
  std::vector<int64_t> first_extent(num_labels, -1);
  for (size_t axis = 0; axis < labels.size(); ++axis) {
    const size_t label = onnxruntime::narrow<size_t>(labels[axis]);
    if (first_extent[label] < 0) {
      first_extent[label] = dims[axis];
    } else if (first_extent[label] != dims[axis]) {
      return false;
    }
  }
  return true;
}

bool ProductFitsCublas(gsl::span<const int64_t> labels,
                       const std::vector<int64_t>& label_extents,
                       int64_t& result) {
  result = 1;
  for (const int64_t label : labels) {
    const int64_t dim = label_extents[onnxruntime::narrow<size_t>(label)];
    if (dim > 0 && result > std::numeric_limits<int>::max() / dim) {
      return false;
    }
    result *= dim;
  }
  return true;
}

std::vector<int64_t> ConcatLabels(gsl::span<const int64_t> first,
                                  gsl::span<const int64_t> second,
                                  gsl::span<const int64_t> third = {}) {
  std::vector<int64_t> result;
  result.reserve(first.size() + second.size() + third.size());
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  result.insert(result.end(), third.begin(), third.end());
  return result;
}

bool AllBatchDimsAreOne(gsl::span<const int64_t> batch_labels,
                        gsl::span<const int64_t> input_labels,
                        gsl::span<const int64_t> input_dims) {
  for (const int64_t label : batch_labels) {
    const auto axis = std::find(input_labels.begin(), input_labels.end(), label);
    if (axis != input_labels.end() &&
        input_dims[static_cast<size_t>(std::distance(input_labels.begin(), axis))] != 1) {
      return false;
    }
  }
  return true;
}

bool AllBatchDimsMatchOutput(gsl::span<const int64_t> batch_labels,
                             gsl::span<const int64_t> input_labels,
                             gsl::span<const int64_t> input_dims,
                             const std::vector<int64_t>& label_extents) {
  for (const int64_t label : batch_labels) {
    const auto axis = std::find(input_labels.begin(), input_labels.end(), label);
    if (axis == input_labels.end() ||
        input_dims[static_cast<size_t>(std::distance(input_labels.begin(), axis))] !=
            label_extents[onnxruntime::narrow<size_t>(label)]) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<Tensor> MakeTensorView(const Tensor& tensor, const TensorShapeVector& dims) {
  return Tensor::Create(tensor.DataType(), TensorShape(dims), const_cast<void*>(tensor.DataRaw()), tensor.Location());
}

std::unique_ptr<Tensor> MakeTensorView(Tensor& tensor, const TensorShapeVector& dims) {
  return Tensor::Create(tensor.DataType(), TensorShape(dims), tensor.MutableDataRaw(), tensor.Location());
}

template <typename T>
Status ExecuteReduceSum(const CudaKernel* cuda_kernel,
                        OpKernelContext* context,
                        const Tensor& input,
                        Tensor& output,
                        gsl::span<const int64_t> axes,
                        EinsumOp::EinsumCudaAssets& assets) {
  PrepareReduceMetadata metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(&input, false, axes, metadata));
  ORT_RETURN_IF_NOT(TensorShape(metadata.squeezed_output_dims) == output.Shape(),
                    "CUDA Einsum reduction fast-path output shape mismatch");
  return ReduceComputeCore<T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      assets.gpu_allocator_, cuda_kernel, input, metadata, output, CUDNN_REDUCE_TENSOR_ADD, axes,
      false, false, false, !context->GetUseDeterministicCompute(), assets.GetCudaStream(),
      assets.ort_stream_, assets.cudnn_handle_);
}

Status BuildDiagonalMetadata(const Tensor& input,
                             const EinsumFastPathPlan& plan,
                             TArray<int64_t>& input_strides,
                             TArray<int32_t>& input_axis_to_output_axis,
                             TArray<fast_divmod>& output_strides,
                             int64_t& trace_dim,
                             int64_t& trace_stride) {
  ORT_RETURN_IF_NOT(input.Shape().NumDimensions() <= kFastPathRankLimit &&
                        plan.output_dims.size() <= kFastPathRankLimit,
                    "CUDA Einsum diagonal fast path rank exceeds metadata capacity");

  TensorPitches pitches(input.Shape().GetDims());
  input_strides.SetSize(onnxruntime::narrow<int32_t>(pitches.size()));
  input_axis_to_output_axis.SetSize(onnxruntime::narrow<int32_t>(plan.input_axis_to_output_axis.size()));
  trace_dim = 0;
  trace_stride = 0;
  for (size_t axis = 0; axis < pitches.size(); ++axis) {
    input_strides[onnxruntime::narrow<int32_t>(axis)] = pitches[axis];
    input_axis_to_output_axis[onnxruntime::narrow<int32_t>(axis)] = plan.input_axis_to_output_axis[axis];
    if (plan.input_axis_to_output_axis[axis] < 0) {
      if (trace_dim == 0) {
        trace_dim = input.Shape()[axis];
      }
      trace_stride += pitches[axis];
    }
  }

  TensorPitches output_pitches(plan.output_dims);
  output_strides.SetSize(onnxruntime::narrow<int32_t>(output_pitches.size()));
  for (size_t axis = 0; axis < output_pitches.size(); ++axis) {
    ORT_RETURN_IF_NOT(output_pitches[axis] <= std::numeric_limits<int>::max(),
                      "CUDA Einsum diagonal fast path output stride exceeds int range");
    output_strides[onnxruntime::narrow<int32_t>(axis)] =
        fast_divmod(onnxruntime::narrow<int>(output_pitches[axis]));
  }

  return Status::OK();
}

}  // namespace

Status CreateEinsumFastPathPlan(const EinsumComputePreprocessor& preprocessor,
                                EinsumFastPathPlan& plan) {
  plan = EinsumFastPathPlan{};
  plan.output_dims = preprocessor.GetOutputDims();

  const auto& inputs = preprocessor.GetRawInputTensors();
  const auto& input_labels = preprocessor.GetInputSubscriptIndices();
  const auto& label_to_output_axis = preprocessor.GetMappedSubscriptIndicesToOutputindices();
  const auto& label_extents = preprocessor.GetSubscriptIndicesToDimValue();
  const size_t num_labels = preprocessor.GetNumSubscriptIndices();
  const auto output_labels = GetOutputLabels(preprocessor);

  if (inputs.size() == 1) {
    const auto& labels = input_labels[0];
    const auto dims = inputs[0]->Shape().GetDims();
    std::vector<size_t> counts(num_labels, 0);
    for (const int64_t label : labels) {
      ++counts[onnxruntime::narrow<size_t>(label)];
    }

    bool has_repeated_label = std::any_of(counts.begin(), counts.end(), [](size_t count) { return count > 1; });
    if (has_repeated_label) {
      if (labels.size() > kFastPathRankLimit || output_labels.size() > kFastPathRankLimit ||
          !IsRepeatedLabelShapeValid(labels, dims, num_labels) ||
          TensorShape(plan.output_dims).Size() > std::numeric_limits<int>::max()) {
        return Status::OK();
      }

      plan.input_axis_to_output_axis.reserve(labels.size());
      std::vector<int64_t> reduced_labels;
      for (const int64_t label : labels) {
        plan.input_axis_to_output_axis.push_back(
            onnxruntime::narrow<int32_t>(label_to_output_axis[onnxruntime::narrow<size_t>(label)]));
      }
      for (size_t label = 0; label < num_labels; ++label) {
        if (counts[label] > 0 && label_to_output_axis[label] < 0) {
          reduced_labels.push_back(onnxruntime::narrow<int64_t>(label));
        }
      }

      if (reduced_labels.empty()) {
        plan.kind = EinsumFastPathKind::Diagonal;
      } else if (reduced_labels.size() == 1 && counts[onnxruntime::narrow<size_t>(reduced_labels[0])] > 1) {
        plan.kind = EinsumFastPathKind::Trace;
        plan.trace_label = reduced_labels[0];
        for (size_t axis = 0; axis < labels.size(); ++axis) {
          if (labels[axis] != plan.trace_label && plan.input_axis_to_output_axis[axis] < 0) {
            plan.kind = EinsumFastPathKind::None;
            break;
          }
        }
      }
      return Status::OK();
    }

    for (size_t axis = 0; axis < labels.size(); ++axis) {
      if (label_to_output_axis[onnxruntime::narrow<size_t>(labels[axis])] < 0) {
        plan.reduce_axes.push_back(onnxruntime::narrow<int64_t>(axis));
      }
    }

    if (!plan.reduce_axes.empty()) {
      std::vector<int64_t> retained_labels;
      for (const int64_t label : labels) {
        if (label_to_output_axis[onnxruntime::narrow<size_t>(label)] >= 0) {
          retained_labels.push_back(label);
        }
      }
      if (retained_labels == output_labels) {
        plan.kind = EinsumFastPathKind::ReduceSum;
      }
      return Status::OK();
    }

    plan.permutation.reserve(output_labels.size());
    for (const int64_t output_label : output_labels) {
      const auto input_axis = std::find(labels.begin(), labels.end(), output_label);
      ORT_ENFORCE(input_axis != labels.end());
      plan.permutation.push_back(static_cast<size_t>(std::distance(labels.begin(), input_axis)));
    }
    plan.kind = !EinsumOp::IsTransposeRequired(labels.size(), plan.permutation) ||
                        IsExactSequenceIgnoringSizeOne(labels, dims, output_labels)
                    ? EinsumFastPathKind::Copy
                    : EinsumFastPathKind::Transpose;
    return Status::OK();
  }

  if (inputs.size() != 2) {
    return Status::OK();
  }

  const auto& lhs_labels = input_labels[0];
  const auto& rhs_labels = input_labels[1];
  const auto lhs_dims = inputs[0]->Shape().GetDims();
  const auto rhs_dims = inputs[1]->Shape().GetDims();
  std::vector<size_t> lhs_counts(num_labels, 0);
  std::vector<size_t> rhs_counts(num_labels, 0);
  for (const int64_t label : lhs_labels) ++lhs_counts[onnxruntime::narrow<size_t>(label)];
  for (const int64_t label : rhs_labels) ++rhs_counts[onnxruntime::narrow<size_t>(label)];
  if (std::any_of(lhs_counts.begin(), lhs_counts.end(), [](size_t count) { return count > 1; }) ||
      std::any_of(rhs_counts.begin(), rhs_counts.end(), [](size_t count) { return count > 1; })) {
    return Status::OK();
  }

  std::vector<int64_t> batch_labels;
  std::vector<int64_t> lhs_free_labels;
  std::vector<int64_t> rhs_free_labels;
  std::vector<int64_t> contraction_labels;
  for (const int64_t label : output_labels) {
    const size_t index = onnxruntime::narrow<size_t>(label);
    if (lhs_counts[index] && rhs_counts[index]) {
      batch_labels.push_back(label);
    } else if (lhs_counts[index]) {
      lhs_free_labels.push_back(label);
    } else if (rhs_counts[index]) {
      rhs_free_labels.push_back(label);
    } else {
      return Status::OK();
    }
  }
  for (size_t label = 0; label < num_labels; ++label) {
    if (label_to_output_axis[label] < 0) {
      if (!lhs_counts[label] || !rhs_counts[label]) {
        return Status::OK();
      }
      contraction_labels.push_back(onnxruntime::narrow<int64_t>(label));
    }
  }

  if (contraction_labels.empty()) {
    if (output_labels.size() > kFastPathRankLimit ||
        TensorShape(plan.output_dims).Size() > std::numeric_limits<int>::max() ||
        !IsExactSequenceIgnoringSizeOne(lhs_labels, lhs_dims, output_labels) ||
        !IsExactSequenceIgnoringSizeOne(rhs_labels, rhs_dims, output_labels)) {
      return Status::OK();
    }
    plan.kind = EinsumFastPathKind::Multiply;
    plan.lhs_view_dims = BuildBroadcastViewDims(lhs_labels, lhs_dims, output_labels);
    plan.rhs_view_dims = BuildBroadcastViewDims(rhs_labels, rhs_dims, output_labels);
    plan.output_view_dims = plan.output_dims;
    return Status::OK();
  }

  const auto expected_output = ConcatLabels(batch_labels, lhs_free_labels, rhs_free_labels);
  if (expected_output != output_labels) {
    return Status::OK();
  }

  std::vector<int64_t> contraction_order;
  for (const int64_t label : lhs_labels) {
    if (std::find(contraction_labels.begin(), contraction_labels.end(), label) != contraction_labels.end()) {
      contraction_order.push_back(label);
    }
  }
  std::vector<int64_t> rhs_contraction_order;
  for (const int64_t label : rhs_labels) {
    if (std::find(contraction_labels.begin(), contraction_labels.end(), label) != contraction_labels.end()) {
      rhs_contraction_order.push_back(label);
    }
  }
  if (contraction_order != rhs_contraction_order) {
    return Status::OK();
  }
  for (const int64_t label : contraction_order) {
    const auto lhs_axis = std::find(lhs_labels.begin(), lhs_labels.end(), label);
    const auto rhs_axis = std::find(rhs_labels.begin(), rhs_labels.end(), label);
    ORT_ENFORCE(lhs_axis != lhs_labels.end() && rhs_axis != rhs_labels.end());
    const int64_t expected_extent = label_extents[onnxruntime::narrow<size_t>(label)];
    if (lhs_dims[static_cast<size_t>(std::distance(lhs_labels.begin(), lhs_axis))] != expected_extent ||
        rhs_dims[static_cast<size_t>(std::distance(rhs_labels.begin(), rhs_axis))] != expected_extent) {
      return Status::OK();
    }
  }

  const auto lhs_normal = ConcatLabels(batch_labels, lhs_free_labels, contraction_order);
  const auto lhs_transposed = ConcatLabels(batch_labels, contraction_order, lhs_free_labels);
  const auto rhs_normal = ConcatLabels(batch_labels, contraction_order, rhs_free_labels);
  const auto rhs_transposed = ConcatLabels(batch_labels, rhs_free_labels, contraction_order);
  if (IsExactSequenceIgnoringSizeOne(lhs_labels, lhs_dims, lhs_normal)) {
    plan.trans_a = false;
  } else if (IsExactSequenceIgnoringSizeOne(lhs_labels, lhs_dims, lhs_transposed)) {
    plan.trans_a = true;
  } else {
    return Status::OK();
  }
  if (IsExactSequenceIgnoringSizeOne(rhs_labels, rhs_dims, rhs_normal)) {
    plan.trans_b = false;
  } else if (IsExactSequenceIgnoringSizeOne(rhs_labels, rhs_dims, rhs_transposed)) {
    plan.trans_b = true;
  } else {
    return Status::OK();
  }

  const bool lhs_batch_matches = AllBatchDimsMatchOutput(batch_labels, lhs_labels, lhs_dims, label_extents);
  const bool rhs_batch_matches = AllBatchDimsMatchOutput(batch_labels, rhs_labels, rhs_dims, label_extents);
  const bool rhs_batch_is_one = AllBatchDimsAreOne(batch_labels, rhs_labels, rhs_dims);
  if (!lhs_batch_matches || (!rhs_batch_matches && !rhs_batch_is_one)) {
    return Status::OK();
  }

  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch_count = 1;
  if (!ProductFitsCublas(batch_labels, label_extents, batch_count) ||
      !ProductFitsCublas(lhs_free_labels, label_extents, m) ||
      !ProductFitsCublas(contraction_order, label_extents, k) ||
      !ProductFitsCublas(rhs_free_labels, label_extents, n)) {
    return Status::OK();
  }

  TensorShapeVector batch_dims;
  for (const int64_t label : batch_labels) {
    const int64_t dim = label_extents[onnxruntime::narrow<size_t>(label)];
    if (dim > std::numeric_limits<int>::max()) {
      return Status::OK();
    }
    batch_dims.push_back(dim);
  }

  plan.lhs_view_dims = batch_dims;
  plan.lhs_view_dims.push_back(plan.trans_a ? k : m);
  plan.lhs_view_dims.push_back(plan.trans_a ? m : k);

  if (rhs_batch_is_one && !rhs_batch_matches) {
    plan.rhs_view_dims = {};
  } else {
    plan.rhs_view_dims = batch_dims;
  }
  plan.rhs_view_dims.push_back(plan.trans_b ? n : k);
  plan.rhs_view_dims.push_back(plan.trans_b ? k : n);

  plan.output_view_dims = batch_dims;
  plan.output_view_dims.push_back(m);
  plan.output_view_dims.push_back(n);
  plan.kind = EinsumFastPathKind::MatMul;
  return Status::OK();
}

template <typename T>
Status ExecuteEinsumFastPath(const CudaKernel* cuda_kernel,
                             OpKernelContext* context,
                             const std::vector<const Tensor*>& inputs,
                             const EinsumFastPathPlan& plan,
                             EinsumOp::EinsumCudaAssets& assets) {
  Tensor& output = *context->Output(0, plan.output_dims);
  switch (plan.kind) {
    case EinsumFastPathKind::Copy:
      return EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy(*inputs[0], output, &assets);
    case EinsumFastPathKind::Transpose:
      return EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose(
          plan.permutation, *inputs[0], output, nullptr, &assets);
    case EinsumFastPathKind::ReduceSum:
      return ExecuteReduceSum<T>(cuda_kernel, context, *inputs[0], output, plan.reduce_axes, assets);
    case EinsumFastPathKind::Diagonal:
    case EinsumFastPathKind::Trace: {
      TArray<int64_t> input_strides;
      TArray<int32_t> input_axis_to_output_axis;
      TArray<fast_divmod> output_strides;
      int64_t trace_dim = 0;
      int64_t trace_stride = 0;
      ORT_RETURN_IF_ERROR(BuildDiagonalMetadata(*inputs[0], plan, input_strides, input_axis_to_output_axis,
                                                output_strides, trace_dim, trace_stride));
      if (plan.kind == EinsumFastPathKind::Diagonal) {
        return LaunchEinsumDiagonal(assets.GetCudaStream(), inputs[0]->DataRaw(), output.MutableDataRaw(),
                                    inputs[0]->DataType()->Size(), onnxruntime::narrow<size_t>(output.Shape().Size()),
                                    input_strides, input_axis_to_output_axis, output_strides);
      }
      return LaunchEinsumTrace(assets.GetCudaStream(), inputs[0]->DataRaw(), output.MutableDataRaw(),
                               inputs[0]->DataType()->Size(), onnxruntime::narrow<size_t>(output.Shape().Size()),
                               trace_dim, trace_stride, input_strides, input_axis_to_output_axis, output_strides);
    }
    case EinsumFastPathKind::Multiply: {
      auto lhs_view = MakeTensorView(*inputs[0], plan.lhs_view_dims);
      auto rhs_view = MakeTensorView(*inputs[1], plan.rhs_view_dims);
      auto output_view = MakeTensorView(output, plan.output_view_dims);
      BinaryElementwisePreparation preparation;
      ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(
          lhs_view.get(), rhs_view.get(), output_view.get(), &preparation));
      using CudaT = typename ToCudaType<T>::MappedType;
      Impl_Mul<CudaT>(assets.GetCudaStream(), preparation.output_rank_or_simple_broadcast,
                      &preparation.lhs_padded_strides, reinterpret_cast<const CudaT*>(lhs_view->Data<T>()),
                      &preparation.rhs_padded_strides, reinterpret_cast<const CudaT*>(rhs_view->Data<T>()),
                      &preparation.fdm_output_strides, preparation.fdm_H, preparation.fdm_C,
                      reinterpret_cast<CudaT*>(output_view->MutableData<T>()),
                      onnxruntime::narrow<size_t>(output_view->Shape().Size()));
      return CUDA_CALL(cudaGetLastError());
    }
    case EinsumFastPathKind::MatMul: {
      auto lhs_view = MakeTensorView(*inputs[0], plan.lhs_view_dims);
      auto rhs_view = MakeTensorView(*inputs[1], plan.rhs_view_dims);
      auto output_view = MakeTensorView(output, plan.output_view_dims);
      return FuncMatMul<T>(cuda_kernel, context, lhs_view.get(), rhs_view.get(), 1.0f,
                           plan.trans_a, plan.trans_b, false, false, output_view.get());
    }
    case EinsumFastPathKind::None:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attempted to execute an ineligible CUDA Einsum fast path");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown CUDA Einsum fast path");
}

template Status ExecuteEinsumFastPath<float>(
    const CudaKernel*, OpKernelContext*, const std::vector<const Tensor*>&,
    const EinsumFastPathPlan&, EinsumOp::EinsumCudaAssets&);
template Status ExecuteEinsumFastPath<double>(
    const CudaKernel*, OpKernelContext*, const std::vector<const Tensor*>&,
    const EinsumFastPathPlan&, EinsumOp::EinsumCudaAssets&);
template Status ExecuteEinsumFastPath<MLFloat16>(
    const CudaKernel*, OpKernelContext*, const std::vector<const Tensor*>&,
    const EinsumFastPathPlan&, EinsumOp::EinsumCudaAssets&);

}  // namespace cuda
}  // namespace onnxruntime
