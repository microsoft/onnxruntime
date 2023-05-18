// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "ml_common.h"
#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace ml {

// code shared by SVMClassifier and SVMRegressor
class SVMCommon {
 protected:
  SVMCommon(const OpKernelInfo& info)
      : kernel_type_(MakeKernel(info.GetAttrOrDefault<std::string>("kernel_type", "LINEAR"))) {
    std::vector<float> kernel_params;
    ORT_ENFORCE(info.GetAttrs<float>("kernel_params", kernel_params).IsOK());

    if (!kernel_params.empty()) {
      gamma_ = kernel_params[0];
      coef0_ = kernel_params[1];
      degree_ = kernel_params[2];
    }
  }

  void set_kernel_type(KERNEL new_kernel_type) { kernel_type_ = new_kernel_type; }
  KERNEL get_kernel_type() const { return kernel_type_; }

  template <typename T>
  void batched_kernel_dot(const gsl::span<const T> a, const gsl::span<const T> b,
                          ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          float scalar_C,
                          const gsl::span<T> out,
                          concurrency::ThreadPool* threadpool) const {
    assert(a.size() == size_t(m * k) && b.size() == size_t(k * n) && out.size() == size_t(m * n));

    if (kernel_type_ == KERNEL::RBF) {
      T* cur_out = out.data();
      const T* cur_batch = a.data();

      // each batch has 'k' features
      for (int64_t batch = 0; batch < m; ++batch) {
        const T* cur_support_vector = b.data();

        // broadcast the support vectors against the k features in each batch. output is one value per support vector
        for (int64_t support_vector = 0; support_vector < n; ++support_vector) {
          T sum = 0.f;
          const T* cur_input = cur_batch;

          for (int64_t feature = 0; feature < k; ++feature) {
            T val = *cur_input++ - *cur_support_vector++;
            sum += val * val;
          }

          *cur_out++ = std::exp(-gamma_ * sum);
        }

        cur_batch += k;  // move to start of next batch
      }
    } else {
      float alpha = 1.f;
      float beta = 1.f;
      static const TensorShape shape_C({1});
      float c = scalar_C;  // scalar_C is used for LINEAR in the GEMM

      if (kernel_type_ != KERNEL::LINEAR) {
        // kernel_type_ == POLY or SIGMOID
        alpha = gamma_;
        c = coef0_;
      }

      onnxruntime::Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                                        m, n, k,
                                        alpha, a.data(), b.data(), beta,
                                        c != 0.f ? &c : nullptr, &shape_C,
                                        out.data(),
                                        threadpool);

      if (kernel_type_ == KERNEL::POLY) {
        auto map_out = EigenVectorArrayMap<T>(out.data(), out.size());
        if (degree_ == 2)
          map_out = map_out.square();
        else if (degree_ == 3)
          map_out = map_out.cube();
        else
          map_out = map_out.pow(degree_);

      } else if (kernel_type_ == KERNEL::SIGMOID) {
        MlasComputeTanh(out.data(), out.data(), out.size());
      }
    }
  }

 private:
  KERNEL kernel_type_;
  float gamma_{0.f};
  float coef0_{0.f};
  float degree_{0.f};
};

class SVMClassifier final : public OpKernel, private SVMCommon {
  using SVMCommon::batched_kernel_dot;
  using SVMCommon::get_kernel_type;
  using SVMCommon::set_kernel_type;

 public:
  SVMClassifier(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeImpl(OpKernelContext& ctx, gsl::span<const float> x_data, const TensorShape& x_shape) const;

  bool weights_are_all_positive_;
  ptrdiff_t feature_count_;
  ptrdiff_t class_count_;
  ptrdiff_t vector_count_;
  bool using_strings_;
  std::vector<int64_t> vectors_per_class_;
  std::vector<int64_t> starting_vector_;
  std::vector<float> rho_;
  std::vector<float> proba_;
  std::vector<float> probb_;
  std::vector<float> coefficients_;
  std::vector<float> support_vectors_;
  std::vector<int64_t> classlabels_ints_;
  std::vector<std::string> classlabels_strings_;
  POST_EVAL_TRANSFORM post_transform_;
  SVM_TYPE mode_;  // how are we computing SVM? 0=LibSVC, 1=LibLinear
};

}  // namespace ml
}  // namespace onnxruntime
