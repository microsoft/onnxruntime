// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "non_max_suppression.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "non_max_suppression_impl.h"
#include "core/providers/cuda/tensor/concat_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10, 10,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(2)
        .InputMemoryType<OrtMemTypeCPUInput>(3)
        .InputMemoryType<OrtMemTypeCPUInput>(4),
    NonMaxSuppression);

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(2)
        .InputMemoryType<OrtMemTypeCPUInput>(3)
        .InputMemoryType<OrtMemTypeCPUInput>(4),
    NonMaxSuppression);

Status NonMaxSuppression::ComputeInternal(OpKernelContext* ctx) const {
  PrepareContext pc;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, pc));

  int64_t max_output_boxes_per_class = 0;
  float iou_threshold = .0f;
  float score_threshold = .0f;

  ORT_RETURN_IF_ERROR(GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold));

  if (0 == pc.num_boxes_ || 0 == max_output_boxes_per_class) {
    ctx->Output(0, {0, 3});
    return Status::OK();
  }

  // TODO: use cub::DeviceSegmentedRadixSort::SortPairsDescending instead of cub::DeviceRadixSort::SortPairsDescending
  //       to deal with multi batch/class parallelly

  std::vector<std::tuple<IAllocatorUniquePtr<void>, int>> all_selected_indices;
  int total_num_saved_outputs = 0;

  // safe downcast max_output_boxes_per_class to int as cub::DeviceSelect::Flagged() does not support int64_t
  int int_max_output_boxes_per_class = max_output_boxes_per_class > std::numeric_limits<int>::max()
                                           ? std::numeric_limits<int>::max()
                                           : static_cast<int>(max_output_boxes_per_class);

  for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
    for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
      IAllocatorUniquePtr<void> d_selected_indices{};
      IAllocatorUniquePtr<void> h_number_selected_ptr{AllocateBufferOnCPUPinned<void>(sizeof(int))};
      auto* h_number_selected = static_cast<int*>(h_number_selected_ptr.get());

      ORT_RETURN_IF_ERROR(NonMaxSuppressionImpl(
          Stream(),
          [this](size_t bytes) { return GetScratchBuffer<void>(bytes); },
          pc,
          GetCenterPointBox(),
          batch_index,
          class_index,
          int_max_output_boxes_per_class,
          iou_threshold,
          score_threshold,
          d_selected_indices,
          h_number_selected));

      int num_saved_outputs = *h_number_selected;
      if (num_saved_outputs > 0) {
        all_selected_indices.emplace_back(std::move(d_selected_indices), num_saved_outputs);
        total_num_saved_outputs += num_saved_outputs;
      }
    }
  }

  if (total_num_saved_outputs == 0) {
    ctx->Output(0, {0, 3});
  } else {
    // concatenate outputs
    const int last_dim = 3;
    const int num_elements = last_dim * total_num_saved_outputs;
    Tensor* output = ctx->Output(0, {static_cast<int64_t>(total_num_saved_outputs), last_dim});
    ORT_ENFORCE(output != nullptr);
    int64_t* dst = output->MutableData<int64_t>();
    size_t count = all_selected_indices.size();

    CudaAsyncBuffer<const void*> input_ptr(this, count);
    CudaAsyncBuffer<int64_t> concat_sizes_gpu(this, count);
    CudaAsyncBuffer<int64_t> concat_sizes_range_gpu(this, count);
    CudaAsyncBuffer<int64_t> axis_dimension_input_output_mapping_gpu(this, total_num_saved_outputs);

    int index = 0;
    for (size_t i = 0; i < count; i++) {
      auto& it = all_selected_indices[i];
      auto src = std::get<0>(it).get();
      auto size = std::get<1>(it);

      input_ptr.CpuPtr()[i] = src;
      concat_sizes_gpu.CpuPtr()[i] = size;
      concat_sizes_range_gpu.CpuPtr()[i] = (i == 0) ? size : size + concat_sizes_range_gpu.CpuPtr()[i - 1];
      for (int j = 0; j < size; j++) {
        axis_dimension_input_output_mapping_gpu.CpuPtr()[index++] = i;
      }
    }

    concat_sizes_gpu.CopyToGpu();
    axis_dimension_input_output_mapping_gpu.CopyToGpu();
    concat_sizes_range_gpu.CopyToGpu();
    input_ptr.CopyToGpu();

    ORT_RETURN_IF_ERROR(ConcatImpl(Stream(),
                                   sizeof(int64_t),
                                   num_elements,
                                   last_dim,
                                   concat_sizes_gpu.GpuPtr(),
                                   concat_sizes_range_gpu.GpuPtr(),
                                   axis_dimension_input_output_mapping_gpu.GpuPtr(),
                                   dst,
                                   input_ptr.GpuPtr(),
                                   static_cast<size_t>(num_elements)));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
