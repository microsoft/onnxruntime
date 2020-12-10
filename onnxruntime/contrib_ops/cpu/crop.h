// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl"

namespace onnxruntime {
namespace contrib {

class CropBase {
 protected:
  CropBase(const OpKernelInfo& info)
      : border_(info.GetAttrsOrDefault<int64_t>("border")),
        scale_(info.GetAttrsOrDefault<int64_t>("scale")) {
  }

  Status ValidateInput(const Tensor* X) const {
    if (border_.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Attribute border needs to be specified with four border elements, got ", border_.size());
    }

    const auto& dims = X->Shape().GetDims();

    if (dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input is expected to have four dimensions corresponding to [N,C,H,W], got ", dims.size(), " input dimensions instead");
    }

    const int64_t H = dims[2];
    const int64_t W = dims[3];

    // find the cropped region, and copy it to the destination matrix
    int64_t leftBorder = border_[0],
            topBorder = border_[1],
            rightBorder = border_[2],
            bottomBorder = border_[3];

    if (H < topBorder + bottomBorder) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input's height (", H, ") needs to be greater than or equal to the topBorder (", topBorder, ") + bottomBorder (", bottomBorder, ")");
    }

    if (W < leftBorder + rightBorder) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input's width (", W, ") needs to be greater than or equal to the leftBorder (", leftBorder, ") + rightBorder (", rightBorder, ")");
    }

    // scale = (height, width)
    if (!scale_.empty()) {
      int64_t bottomLimit = topBorder + scale_[0];
      int64_t rightLimit = leftBorder + scale_[1];

      if (H < bottomLimit) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input's height (", H, ") needs to be greater than or equal to the topBorder (", topBorder, ") + scale_[0] (", scale_[0], ")");
      }

      if (W < rightLimit) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input's width (", W, ") needs to be greater than or equal to the leftBorder (", leftBorder, ") + scale_[1] (", scale_[1], ")");
      }
    }

    return Status::OK();
  }

  const std::vector<int64_t> border_;  // (leftBorder, topBorder, rightBorder, bottomBorder)
  const std::vector<int64_t> scale_;   // (height, width)
};

template <typename T>
class Crop final : public CropBase, public OpKernel {
 public:
  Crop(const OpKernelInfo& info) : CropBase(info), OpKernel(info) {
  }

  common::Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    ORT_RETURN_IF_ERROR(ValidateInput(X));

    const auto& dims = X->Shape().GetDims();
    const int64_t N = dims[0];
    const int64_t C = dims[1];
    const int64_t H = dims[2];
    const int64_t W = dims[3];

    // find the cropped region, and copy it to the destination matrix
    int64_t leftBorder = border_[0],
            topBorder = border_[1],
            rightBorder = border_[2],
            bottomBorder = border_[3];

    int64_t bottomLimit = H - bottomBorder;
    int64_t rightLimit = W - rightBorder;

    // scale = (height, width)
    if (!scale_.empty()) {
      bottomLimit = topBorder + scale_[0];
      rightLimit = leftBorder + scale_[1];
    }

    Tensor* Y = context->Output(0, TensorShape({N, C, bottomLimit - topBorder, rightLimit - leftBorder}));
    const T* Xdata = X->template Data<T>();
    T* Ydata = Y->template MutableData<T>();

    int64_t dest_idx = 0;
    int64_t HW = H * W;
    int64_t CHW = C * HW;
    int64_t nCHW;
    int64_t nCHW_p_cHW;
    int64_t nCHW_p_cHW_p_hW;
    int64_t source_idx;
    for (int64_t n = 0; n < N; ++n) {
      nCHW = n * CHW;
      for (int64_t c = 0; c < C; ++c) {
        nCHW_p_cHW = nCHW + c * HW;
        for (int64_t h = topBorder; h < bottomLimit; ++h) {
          nCHW_p_cHW_p_hW = nCHW_p_cHW + h * W;
          for (int64_t w = leftBorder; w < rightLimit; ++w) {
            source_idx = nCHW_p_cHW_p_hW + w;
            Ydata[dest_idx++] = Xdata[source_idx];
          }
        }
      }
    }
    return Status::OK();
  }
};

}  // namespace contrib
}  //namespace onnxruntime
