/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "roialign.h"
#include "core/util/math_cpuonly.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/platform/threadpool.h"

using namespace onnxruntime::concurrency;

namespace onnxruntime {
namespace contrib {
const int64_t EXPECTED_NUM_ROI_DIMS = 2;
const int64_t EXPECTED_SECOND_ROI_DIM = 5;

#define ADD_TYPED_ROIALIGN_OP(data_type)                                  \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                      \
      ROIAlign,                                                           \
      1,                                                                  \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      ROIAlign<data_type>);

ADD_TYPED_ROIALIGN_OP(float);
ADD_TYPED_ROIALIGN_OP(double);

namespace {
template <typename T>
struct PreCalc {
  int64_t pos1;
  int64_t pos2;
  int64_t pos3;
  int64_t pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t iy_upper,
    const int64_t ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int64_t roi_bin_grid_h,
    int64_t roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int64_t pre_calc_index = 0;
  for (int64_t ph = 0; ph < pooled_height; ph++) {
    for (int64_t pw = 0; pw < pooled_width; pw++) {
      for (int64_t iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                         static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int64_t ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                           static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int64_t y_low = static_cast<int64_t>(y);
          int64_t x_low = static_cast<int64_t>(x);
          int64_t y_high;
          int64_t x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = static_cast<T>(1.) - ly, hx = static_cast<T>(1.) - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(
    int64_t nthreads,
    const T* bottom_data,
    float spatial_scale,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    const std::string& mode,
    const ThreadPool* ttp) {
  int64_t n_rois = nthreads / channels / pooled_width / pooled_height;

  std::function<void(int32_t)> work_object = [&](int32_t n) {
    int64_t index_n = n * channels * pooled_width * pooled_height;

    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    const T roi_batch_ind = offset_bottom_rois[0];
    offset_bottom_rois++;

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int64_t roi_bin_grid_h = (sampling_ratio > 0)
                                 ? sampling_ratio
                                 : static_cast<int64_t>(ceil(roi_height / pooled_height));  // e.g., = 2
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(roi_width / pooled_width));

    // We do average (integral) pooling inside a bin
    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    for (int64_t c = 0; c < channels; c++) {
      int64_t index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_bottom_data =
          bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
      int64_t pre_calc_index = 0;

      for (int64_t ph = 0; ph < pooled_height; ph++) {
        for (int64_t pw = 0; pw < pooled_width; pw++) {
          int64_t index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          if (mode == "avg") {  // avg pooling
            for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc<T> pc = pre_calc[pre_calc_index];
                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                              pc.w2 * offset_bottom_data[pc.pos2] +
                              pc.w3 * offset_bottom_data[pc.pos3] +
                              pc.w4 * offset_bottom_data[pc.pos4];

                pre_calc_index += 1;
              }
            }
            output_val /= count;
          } else {  // max pooling
            bool max_flag = false;
            for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc<T> pc = pre_calc[pre_calc_index];
                if (!max_flag) {
                  output_val = pc.w1 * offset_bottom_data[pc.pos1];
                  max_flag = true;
                } else {
                  output_val = std::max(std::max(std::max(output_val, pc.w2 * offset_bottom_data[pc.pos2]),
                                                 pc.w3 * offset_bottom_data[pc.pos3]),
                                        pc.w4 * offset_bottom_data[pc.pos4]);
                }

                pre_calc_index += 1;
              }
            }
          }

          top_data[index] = output_val;
        }  // for pw
      }    // for ph
    }      // for c
  };       // for n
  const_cast<ThreadPool*>(ttp)->ParallelFor((int32_t)n_rois, work_object);
}
}  // namespace

template <typename T>
Status ROIAlign<T>::Compute(OpKernelContext* context) const {
  const Tensor* X_ptr = context->Input<Tensor>(0);
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null input X ptr");
  }

  const Tensor* rois_ptr = context->Input<Tensor>(1);
  if (!rois_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null rois_ptr");
  }

  auto& x_dims = X_ptr->Shape();

  auto& rois_dims = rois_ptr->Shape();

  // validate rois_dims
  if (rois_dims.NumDimensions() != EXPECTED_NUM_ROI_DIMS) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Number of dimensions for rois should be exactly 2");
  }
  if (rois_dims[1] != EXPECTED_SECOND_ROI_DIM) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Second dimension for rois should be exactly 5");
  }

  auto& Y = *context->Output(0, {rois_dims[0], x_dims[1], pooled_h_, pooled_w_});
  int64_t output_size = Y.Shape().Size();
  ROIAlignForward<T>(
      output_size,
      X_ptr->Data<T>(),
      spatial_scale_,
      x_dims[1],
      x_dims[2],
      x_dims[3],
      pooled_h_,
      pooled_w_,
      sampling_ratio_,
      rois_ptr->Data<T>(),
      rois_dims[1],
      Y.template MutableData<T>(),
      mode_,
      static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool());

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
