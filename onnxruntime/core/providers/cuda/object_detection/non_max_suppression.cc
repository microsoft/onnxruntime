// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "non_max_suppression.h"
#include "non_max_suppression_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    KernelDefBuilder(),
    cuda::NonMaxSuppression);

namespace nms_helpers {
Status GetThresholdsFromInputs(const PrepareContext& pc,
                               int64_t& max_output_boxes_per_class,
                               float& iou_threshold,
                               float& score_threshold) {
  // XXX: Should we apply async copy?
  if (pc.max_output_boxes_per_class_ != nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&max_output_boxes_per_class, pc.max_output_boxes_per_class_,
                                    sizeof(int64_t), cudaMemcpyDeviceToHost));
    max_output_boxes_per_class = std::max(max_output_boxes_per_class, 0ll);
  }

  if (pc.iou_threshold_ != nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&iou_threshold, pc.iou_threshold_,
                                    sizeof(float), cudaMemcpyDeviceToHost));
    ORT_RETURN_IF_NOT((iou_threshold >= 0 && iou_threshold <= 1.f), "iou_threshold must be in range [0, 1].");
  }

  if (pc.score_threshold_ != nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&score_threshold, pc.score_threshold_,
                                    sizeof(float), cudaMemcpyDeviceToHost));
  }
  return Status::OK();
}
}  // namespace nms_helpers

using namespace nms_helpers;

Status NonMaxSuppression::ComputeInternal(OpKernelContext* ctx) const {

  PrepareContext pc;
  auto ret = PrepareCompute(ctx, pc);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  int64_t max_output_boxes_per_class = 0;
  float iou_threshold = .0f;
  float score_threshold = .0f;
  ret = GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  if (0 == max_output_boxes_per_class) {
    ctx->Output(0, {0, 3});
    return Status::OK();
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
