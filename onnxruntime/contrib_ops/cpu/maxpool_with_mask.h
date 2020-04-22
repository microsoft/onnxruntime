// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*
* Highly specialized code, only works for TP3 L1
*/

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace contrib {

class MaxpoolWithMask : public OpKernel, public PoolBase {
 public:
  MaxpoolWithMask(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* M = context->Input<Tensor>(1);

    const TensorShape& x_shape = X->Shape();
    const TensorShape& m_shape = M->Shape();
    ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

    //TODO: fix this checker later
    //ONNXRUNTIME_RETURN_IF_NOT((x_shape[2] == m_shape[2]) && (x_shape[3] == m_shape[3]), " Input shape and mask shape mismatch: ", x_shape, " vs ", m_shape);

    std::vector<int64_t> pads = pool_attrs_.pads;
    std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

    std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
    Tensor* Y = context->Output(0, TensorShape(output_dims));

    const float* X_data = X->template Data<float>();
    const int32_t* M_data = M->template Data<int32_t>();
    float* Y_data = Y->template MutableData<float>();

    // The main loop
    int64_t channels = x_shape[1];
    int64_t height = x_shape[2];
    int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
    int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
    int64_t pooled_height = output_dims[2];
    int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
    int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;

    switch (kernel_shape.size()) {
      case 1: {
        int64_t x_step = height;
        int64_t y_step = pooled_height;
        const int64_t total_channels = x_shape[0] * channels;
        const int64_t total_mask_channels = m_shape[0] * m_shape[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t c = 0; c < total_channels; ++c) {
          const float* x_d = X_data + c * x_step;
          const int32_t* m_d = M_data + (c * x_step) % total_mask_channels;
          float* y_d = Y_data + c * y_step;
          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            float Yh = std::numeric_limits<float>::lowest();
            for (int64_t h = hstart; h < hend; ++h) {
              if (h >= 0 && m_d[h] == 0) break;  // if mask == 0, stop
              if (x_d[h] > Yh) {
                Yh = x_d[h];
              }
            }
            y_d[ph] = Yh;
          }
        }

        break;
      }

      case 2: {
        int64_t x_step = height * width;
        int64_t y_step = pooled_height * pooled_width;
        const int64_t total_channels = x_shape[0] * channels;
        const int64_t total_mask_channels = m_shape[0] * m_shape[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t c = 0; c < total_channels; ++c) {
          const float* x_d = X_data + c * x_step;
          const int32_t* m_d = M_data + (c * x_step) % total_mask_channels;
          float* y_d = Y_data + c * y_step;

          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
              int64_t wstart = pw * stride_w() - pads[1];
              int64_t wend = std::min(wstart + kernel_shape[1], width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
              const int64_t pool_index = ph * pooled_width + pw;
              float Yh = std::numeric_limits<float>::lowest();
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  const int64_t input_index = h * width + w;
                  if (input_index > 0 && m_d[input_index] == 0) break;  // if mask == 0, break
                  if (x_d[input_index] > Yh) {
                    Yh = x_d[input_index];
                  }
                }
              }
              y_d[pool_index] = Yh;
            }
          }
        }
        break;
      }
      case 3: {
        int64_t x_step = height * width * depth;
        int64_t y_step = pooled_height * pooled_width * pooled_depth;
        const int64_t total_channels = x_shape[0] * channels;
        const int64_t total_mask_channels = m_shape[0] * m_shape[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t c = 0; c < total_channels; ++c) {
          const float* x_d = X_data + c * x_step;
          const int32_t* m_d = M_data + (c * x_step) % total_mask_channels;
          float* y_d = Y_data + c * y_step;

          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
              int64_t wstart = pw * stride_w() - pads[1];
              int64_t wend = std::min(wstart + kernel_shape[1], width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
              for (int64_t pd = 0; pd < pooled_depth; ++pd) {
                int64_t dstart = pd * stride_d() - pads[2];
                int64_t dend = std::min(dstart + kernel_shape[2], depth);
                dstart = std::max(dstart, static_cast<int64_t>(0));
                const int64_t pool_index =
                    ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                float Yh = std::numeric_limits<float>::lowest();
                for (int64_t h = hstart; h < hend; ++h) {
                  for (int64_t w = wstart; w < wend; ++w) {
                    for (int64_t d = dstart; d < dend; ++d) {
                      const int64_t input_index = h * width * depth + w * depth + d;
                      if (input_index > 0 && m_d[input_index] == 0) break;  // if mask == 0, break
                      if (x_d[input_index] > Yh) {
                        Yh = x_d[input_index];
                      }
                    }
                  }
                }
                y_d[pool_index] = Yh;
              }
            }
          }
        }
        break;
      }
      default:
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported pooling size : ");
    }

    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
