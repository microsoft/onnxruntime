/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "contrib_ops/cpu/crop_and_resize.h"

#include <cmath>
#include "core/util/math_cpuonly.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/object_detection/roialign.h"

using namespace onnxruntime::concurrency;

namespace onnxruntime {
namespace contrib {

#define ADD_TYPED_CROPANDRESIZE_OP(data_type)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      CropAndResize,                                                     \
      kMSDomain,                                                         \
      1,                                                                 \
      data_type,                                                         \
      kCpuExecutionProvider,                                             \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>()), \
      CropAndResize<data_type>);

ADD_TYPED_CROPANDRESIZE_OP(float);

template <typename T>
void CropAndResizeForward(const TensorShape& output_shape,
                          const T* bottom_data,
                          float extrapolation_value,
                          int64_t height,
                          int64_t width,
                          const T* bottom_rois,
                          int64_t num_roi_cols,
                          T* top_data,
                          const std::string& mode,
                          const int32_t* batch_indices_ptr,
                          ThreadPool* ttp) {
  int64_t n_rois = output_shape[0];
  int64_t channels = output_shape[1];
  int64_t pooled_height = output_shape[2];
  int64_t pooled_width = output_shape[3];

  ThreadPool::TryBatchParallelFor(ttp, static_cast<int32_t>(n_rois), [&](ptrdiff_t n) {
    int64_t index_n = n * channels * pooled_width * pooled_height;

    const T* offset_bottom_rois = bottom_rois + n * num_roi_cols;
    const auto roi_batch_ind = batch_indices_ptr[n];

    T roi_start_w = offset_bottom_rois[1];
    T roi_start_h = offset_bottom_rois[0];
    T roi_end_w = offset_bottom_rois[3];
    T roi_end_h = offset_bottom_rois[2];

    T height_scale = (pooled_height > 1)
                         ? (roi_end_h - roi_start_h) * (height - 1) / (pooled_height - 1)
                         : 0;
    T width_scale = (pooled_width > 1)
                        ? (roi_end_w - roi_start_w) * (width - 1) / (pooled_width - 1)
                        : 0;

    for (auto ph = 0; ph < pooled_height; ph++) {
      T in_y = static_cast<T>((pooled_height > 1)
                                  ? roi_start_h * (height - 1) + ph * height_scale
                                  : 0.5 * (roi_start_h + roi_end_h) * (height - 1));
      if (ph == pooled_height - 1) {
        in_y = static_cast<T>((pooled_height > 1)
                                  ? roi_end_h * (height - 1)
                                  : 0.5 * (roi_start_h + roi_end_h) * (height - 1));
      }
      if (ph == 0) {
        in_y = static_cast<T>((pooled_height > 1)
                                  ? roi_start_h * (height - 1)
                                  : 0.5 * (roi_start_h + roi_end_h) * (height - 1));
      }
      if (in_y < 0 || in_y > height - 1) {
        for (int64_t pw = 0; pw < pooled_width; pw++) {
          for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            int64_t index = index_n_c + ph * pooled_width + pw;
            top_data[index] = extrapolation_value;
          }
        }
        continue;
      }

      const int top_y_index = static_cast<int>(floorf(static_cast<float>(in_y)));
      const int bottom_y_index = static_cast<int>(ceilf(static_cast<float>(in_y)));
      const float y_lerp = static_cast<float>(in_y - top_y_index);

      for (auto pw = 0; pw < pooled_width; pw++) {
        T in_x = static_cast<T>((pooled_width > 1)
                                    ? roi_start_w * (width - 1) + pw * width_scale
                                    : 0.5 * (roi_start_w + roi_end_w) * (width - 1));
        if (pw == pooled_width - 1) {
          in_x = static_cast<T>((pooled_width > 1)
                                    ? roi_end_w * (width - 1)
                                    : 0.5 * (roi_start_w + roi_end_w) * (width - 1));
        }
        if (pw == 0) {
          in_x = static_cast<T>((pooled_width > 1)
                                    ? roi_start_w * (width - 1)
                                    : 0.5 * (roi_start_w + roi_end_w) * (width - 1));
        }
        if (in_x < 0 || in_x > width - 1) {
          for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            int64_t index = index_n_c + ph * pooled_width + pw;
            top_data[index] = extrapolation_value;
          }
          continue;
        }

        T output_val = extrapolation_value;
        if (mode == "bilinear") {
          const int left_x_index = static_cast<int>(floorf(static_cast<float>(in_x)));
          const int right_x_index = static_cast<int>(ceilf(static_cast<float>(in_x)));
          const float x_lerp = static_cast<float>(in_x - left_x_index);
          auto top_left_index = top_y_index * width + left_x_index;
          auto top_right_index = top_y_index * width + right_x_index;
          auto bottom_left_index = bottom_y_index * width + left_x_index;
          auto bottom_right_index = bottom_y_index * width + right_x_index;

          for (auto c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            int64_t index = index_n_c + ph * pooled_width + pw;
            const T* offset_bottom_data =
                bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
            const float top_left(static_cast<float>(offset_bottom_data[top_left_index]));
            const float top_right(static_cast<float>(offset_bottom_data[top_right_index]));
            const float bottom_left(static_cast<float>(offset_bottom_data[bottom_left_index]));
            const float bottom_right(static_cast<float>(offset_bottom_data[bottom_right_index]));
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
            output_val = top + (bottom - top) * y_lerp;
            top_data[index] = output_val;
          }
        } else {  // mode == "nearest"
          const int closest_x_index = static_cast<int>(roundf(static_cast<float>(in_x)));
          const int closest_y_index = static_cast<int>(roundf(static_cast<float>(in_y)));
          auto closest_index = closest_y_index * width + closest_x_index;

          for (auto c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            int64_t index = index_n_c + ph * pooled_width + pw;
            const T* offset_bottom_data =
                bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
            top_data[index] = static_cast<float>(offset_bottom_data[closest_index]);
          }
        }
      }  // for pw
    }    // for ph
  }, 0);    // for n
}

template <typename T>
Status CropAndResize<T>::Compute(OpKernelContext* context) const {
  // X
  const auto* X_ptr = context->Input<Tensor>(0);
  // rois
  const auto* rois_ptr = context->Input<Tensor>(1);
  // batch indices
  const auto* batch_indices_ptr = context->Input<Tensor>(2);
  // crop size
  const auto* crop_size_ptr = context->Input<Tensor>(3);
  if (!crop_size_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null crop_size_ptr");
  }

  const auto& x_dims = X_ptr->Shape();
  const auto& rois_dims = rois_ptr->Shape();
  const auto& batch_indices_dims = batch_indices_ptr->Shape();
  const auto& crop_size_dims = crop_size_ptr->Shape();

  //validate crop_size
  if (crop_size_dims.NumDimensions() != 1) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Number of dimensions for crop size should be exactly 1");
  }

  auto channels = x_dims[1];
  auto num_rois = batch_indices_dims[0];
  auto num_roi_cols = rois_dims[1];
  auto crop_size_data = crop_size_ptr->Data<int32_t>();
  auto crop_height = crop_size_data[0];
  auto crop_width = crop_size_data[1];

  auto status = CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr);
  if (status != Status::OK()) {
    return status;
  }

  TensorShape Y_shape = {num_rois, channels, crop_height, crop_width};
  auto& Y = *context->Output(0, Y_shape);
  CropAndResizeForward<T>(
      Y_shape,
      X_ptr->Data<T>(),
      extrapolation_value_,
      x_dims[2],  // height
      x_dims[3],  // width
      rois_ptr->Data<T>(),
      num_roi_cols,
      Y.template MutableData<T>(),
      mode_,
      batch_indices_ptr->Data<int32_t>(),
      context->GetOperatorThreadPool());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
