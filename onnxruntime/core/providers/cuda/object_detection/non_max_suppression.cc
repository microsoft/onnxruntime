// Copyright (c) Microsoft Corporation. All rights reserved. 
// Licensed under the MIT License. 

#include "non_max_suppression.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "non_max_suppression_impl.h"

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
  auto ret = PrepareCompute(ctx, pc);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  int64_t max_output_boxes_per_class = 0;
  float iou_threshold = .0f;
  float score_threshold = .0f;

  ret = GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  if (0 == pc.num_boxes_ || 0 == max_output_boxes_per_class) {
    ctx->Output(0, {0, 3});
    return Status::OK();
  }

  // TODO: use cub::DeviceSegmentedRadixSort::SortPairsDescending instead of cub::DeviceRadixSort::SortPairsDescending
  //       to deal with multi batch/class parallelly

  std::vector<std::tuple<IAllocatorUniquePtr<void>, int>> all_selected_indices;
  int total_num_saved_outputs = 0;

  for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
    for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
      IAllocatorUniquePtr<void> d_selected_indices{};
      IAllocatorUniquePtr<void> h_number_selected_ptr{AllocateBufferOnCPUPinned<void>(sizeof(int))};
      auto* h_number_selected = static_cast<int*>(h_number_selected_ptr.get());

      ORT_RETURN_IF_ERROR(NonMaxSuppressionImpl(
        [this](size_t bytes){ return GetScratchBuffer<void>(bytes); },
        pc,
        GetCenterPointBox(),
        batch_index,
        class_index,
        max_output_boxes_per_class,
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

  if (total_num_saved_outputs == 0){
    ctx->Output(0, {0, 3});
  } else {
    const auto last_dim = 3;
    Tensor* output = ctx->Output(0, {static_cast<int64_t>(total_num_saved_outputs), last_dim});
    ORT_ENFORCE(output != nullptr);
    int64_t *dst = output->MutableData<int64_t>();
    for (auto& it: all_selected_indices) {
      void *src = std::get<0>(it).get();
      size_t count = std::get<1>(it) * last_dim;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, count * sizeof(int64_t), cudaMemcpyDeviceToDevice));
      dst += count;
    }
  }

  return Status::OK();
}

}  // namespace cuda
};  // namespace onnxruntime
