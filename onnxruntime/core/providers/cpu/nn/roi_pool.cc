// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/roi_pool.h"
#include <cmath>

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    MaxRoiPool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    RoiPool<float>);

template <>
Status RoiPool<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* R = context->Input<Tensor>(1);
  if (X == nullptr || R == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

  int batch_size = static_cast<int>(X->Shape()[0]);
  int channels = static_cast<int>(X->Shape()[1]);
  int height = static_cast<int>(X->Shape()[2]);
  int width = static_cast<int>(X->Shape()[3]);
  int num_rois = static_cast<int>(R->Shape()[0]);

  // Each ROI is of the form [batch_index x1 y1 x2 y2]
  ORT_ENFORCE(R->Shape()[1] == 5);

  Tensor* Y = context->Output(0, {num_rois, channels, pooled_height_, pooled_width_});

  const auto* Xdata = X->template Data<float>();
  const auto* rois = R->template Data<float>();

  auto* Ydata = Y->template MutableData<float>();

  for (int n = 0; n < num_rois; n++) {
    int roi_batch_id = static_cast<int>(rois[0]);
    int roi_start_w = static_cast<int>(std::round(rois[1] * spatial_scale_));
    int roi_start_h = static_cast<int>(std::round(rois[2] * spatial_scale_));
    int roi_end_w = static_cast<int>(std::round(rois[3] * spatial_scale_));
    int roi_end_h = static_cast<int>(std::round(rois[4] * spatial_scale_));
    ORT_ENFORCE(roi_batch_id >= 0);
    ORT_ENFORCE(roi_batch_id < batch_size);

    // Force malformed ROIs to be 1x1
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);

    const float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height_);
    const float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width_);

    const float* batch_data = Xdata + roi_batch_id * X->Shape().SizeFromDimension(1);

    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(std::floor(static_cast<float>(ph) * bin_size_h));
          int wstart = static_cast<int>(std::floor(static_cast<float>(pw) * bin_size_w));
          int hend = static_cast<int>(std::ceil(static_cast<float>(ph + 1) * bin_size_h));
          int wend = static_cast<int>(std::ceil(static_cast<float>(pw + 1) * bin_size_w));

          // Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart + roi_start_h, 0), height);
          hend = std::min(std::max(hend + roi_start_h, 0), height);
          wstart = std::min(std::max(wstart + roi_start_w, 0), width);
          wend = std::min(std::max(wend + roi_start_w, 0), width);

          const int pool_index = static_cast<int>(ph * pooled_width_ + pw);

          // Define an empty pooling region to be zero
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          Ydata[pool_index] = is_empty ? 0 : std::numeric_limits<float>::lowest();

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              Ydata[pool_index] = std::max(batch_data[index], Ydata[pool_index]);
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += X->Shape().SizeFromDimension(2);
      Ydata += Y->Shape().SizeFromDimension(2);
    }
    // Increment ROI data pointer
    rois += R->Shape().SizeFromDimension(1);
  }

  return Status::OK();
}

}  // namespace onnxruntime
