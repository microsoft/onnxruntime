// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

#include "gsl/gsl"
namespace onnxruntime {
template <typename T>
class MeanVarianceNormalization_0 : public OpKernel {
 public:
  MeanVarianceNormalization_0(const OpKernelInfo& info, bool old_attr = true) : OpKernel(info) {
    if (old_attr) {
      ORT_ENFORCE(info.GetAttr<int64_t>("across_channels", &across_channels_).IsOK());
      ORT_ENFORCE(info.GetAttr<int64_t>("normalize_variance", &normalize_variance_).IsOK());
    }
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    const auto dims = X->Shape().GetDims();

    if (dims.size() < 4) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input is expected to have four dimensions corresponding to [N,C,H,W]");
    }

    const int64_t N = dims[0];
    const int64_t C = dims[1];
    const int64_t H = dims[2];
    const int64_t W = dims[3];

    Tensor* Y = context->Output(0, {N, C, H, W});
    const T* Xdata = X->template Data<T>();
    T* Ydata = Y->template MutableData<T>();

    const int64_t sample_size = H * W;
    Eigen::Array<float, Eigen::Dynamic, 1> mean(C, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> var(C, 1);
    mean.setZero();
    var.setZero();

    ConstEigenArrayMap<T> X_arr(Xdata, sample_size, N * C);
    for (int nc = 0; nc < N * C; ++nc) {
      mean(nc % C) += X_arr.col(nc).sum();
    }
    mean /= gsl::narrow_cast<T>(N * sample_size);
    for (int64_t nc = 0; nc < N * C; ++nc) {
      var(nc % C) += (X_arr.col(nc) - mean(nc % C)).matrix().squaredNorm();
    }
    var /= gsl::narrow_cast<T>(N * sample_size);

    Eigen::Array<T, Eigen::Dynamic, 1> inv_std;
    EigenArrayMap<T> Y_arr(Ydata, sample_size, N * C);

    if (across_channels_) {
      // m_c = sum(m_i) / n
      float global_mean = mean.mean();

      // var_c = [(var_1 + (m_1 - m_c)^2) + ...  + (var_n + (m_n - m_c)^2)] / n
      //       = [sum(var_i) + squared_norm(m_i - m_c)] / n
      float global_var = ((mean - global_mean).matrix().squaredNorm() + var.sum()) / C;

      // For across channels we can directly use eigen because global_mean and global_var
      // are just scalars.
      if (!normalize_variance_) {
        Y_arr = X_arr - global_mean;
      } else {
        float inv_std_scalar = 1 / std::sqrt(global_var);
        Y_arr = (X_arr - global_mean) * inv_std_scalar;
      }
    } else {
      if (!normalize_variance_) {
        // inv_std = 1
        for (int64_t nc = 0; nc < N * C; ++nc) {
          // y = (x - mean)
          Y_arr.col(nc) = (X_arr.col(nc) - mean(nc % C));
        }
      } else {
        inv_std = var.sqrt().inverse();
        for (int64_t nc = 0; nc < N * C; ++nc) {
          // y = (x - mean) * (inv_std)
          Y_arr.col(nc) = (X_arr.col(nc) - mean(nc % C)) * inv_std(nc % C);
        }
      }
    }
    return Status::OK();
  }

 protected:
  int64_t across_channels_;
  int64_t normalize_variance_;
};

template <typename T>
class MeanVarianceNormalization_1 final : public MeanVarianceNormalization_0<T> {
 public:
  MeanVarianceNormalization_1(const OpKernelInfo& info) : MeanVarianceNormalization_0<T>(info, false) {
    std::vector<int64_t> axes;
    if (!info.GetAttrs("axes", axes).IsOK()) {
      axes = {0, 2, 3};
    }
    if (find(axes.begin(), axes.end(), 1) != axes.end()) {
      this->across_channels_ = true;
    } else {
      this->across_channels_ = false;
    }
    this->normalize_variance_ = 1;
  }
};

}  //namespace onnxruntime
