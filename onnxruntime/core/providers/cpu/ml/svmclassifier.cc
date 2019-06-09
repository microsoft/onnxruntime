// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/svmclassifier.h"

namespace onnxruntime {
namespace ml {

#define ADD_IN_TYPE_SVM_CLASSIFIER_OP(in_type)                                                                                                                                                    \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      SVMClassifier,                                                                                                                                                                              \
      1,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      SVMClassifier<in_type>);

ADD_IN_TYPE_SVM_CLASSIFIER_OP(float);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(double);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(int64_t);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(int32_t);

template <typename T>
SVMClassifier<T>::SVMClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      SVMCommon<T>(info),
      vectors_per_class_(info.GetAttrsOrDefault<int64_t>("vectors_per_class")),
      proba_(info.GetAttrsOrDefault<float>("prob_a")),
      probb_(info.GetAttrsOrDefault<float>("prob_b")),
      support_vectors_(info.GetAttrsOrDefault<float>("support_vectors")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttrs<float>("rho", rho_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());

  // prob_a and prob_b are optional for Z output
  ORT_ENFORCE(proba_.size() == probb_.size());

  // one of these should be valid
  ORT_ENFORCE(info.GetAttrs<std::string>("classlabels_strings", classlabels_strings_).IsOK() ||
              info.GetAttrs<int64_t>("classlabels_ints", classlabels_ints_).IsOK());

  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  using_strings_ = false;
  if (!classlabels_strings_.empty()) {
    using_strings_ = true;
    class_count_ = classlabels_strings_.size();
  } else if (!classlabels_ints_.empty()) {
    class_count_ = classlabels_ints_.size();
  } else {
    class_count_ = 1;
  }
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size() / class_count_;  //liblinear mode
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }
  ORT_ENFORCE(!classlabels_strings_.empty() || !classlabels_ints_.empty());
  ORT_ENFORCE(proba_.size() == probb_.size());
  ORT_ENFORCE(!coefficients_.empty());
  weights_are_all_positive_ = true;
  for (int64_t i = 0; i < static_cast<int64_t>(coefficients_.size()); i++) {
    if (coefficients_[i] < 0) {
      weights_are_all_positive_ = false;
      break;
    }
  }
}

template <typename LabelType>
int _set_score_svm(Tensor* Y, float max_weight, const int64_t maxclass, const int64_t n,
                   POST_EVAL_TRANSFORM post_transform_, const std::vector<float>& proba_, bool weights_are_all_positive_,
                   const std::vector<LabelType>& classlabels, LabelType posclass, LabelType negclass) {
  int write_additional_scores = -1;
  auto output_data = Y->template MutableData<LabelType>();
  if (classlabels.size() == 2) {
    write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
    if (proba_.empty()) {
      if (weights_are_all_positive_ && max_weight >= 0.5)
        output_data[n] = classlabels[1];
      else if (max_weight > 0 && !weights_are_all_positive_)
        output_data[n] = classlabels[1];
      else
        output_data[n] = classlabels[maxclass];
    } else {
      output_data[n] = classlabels[maxclass];
    }
  } else if (max_weight > 0) {
    output_data[n] = posclass;
  } else {
    output_data[n] = negclass;
  }
  return write_additional_scores;
}

template <typename T>
Status SVMClassifier<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N}));

  int64_t nb_columns = class_count_;
  if (proba_.empty() && vector_count_ > 0) {
    if (class_count_ > 2)
      nb_columns = class_count_ * (class_count_ - 1) / 2;
    else
      nb_columns = 2;
  }

  std::vector<int64_t> dims{N, nb_columns};
  Tensor* Z = ctx->Output(1, TensorShape(dims));

  const T* x_data = X->template Data<T>();
  int64_t zindex = 0;

  for (int64_t n = 0; n < N; n++)  //for each example
  {
    int64_t current_weight_0 = n * stride;
    int64_t maxclass = -1;
    std::vector<float> decisions;
    std::vector<float> scores;
    std::vector<float> kernels;
    std::vector<int64_t> votes;

    if (vector_count_ == 0 && mode_ == SVM_TYPE::SVM_LINEAR) {
      for (int64_t j = 0; j < class_count_; j++) {  //for each class
        auto val = kernel_dot(x_data, current_weight_0, coefficients_, feature_count_ * j,
                              feature_count_, get_kernel_type());
        val += rho_[0];
        scores.push_back(val);
      }
    } else {
      if (vector_count_ == 0)
        return Status(common::ONNXRUNTIME, common::FAIL, "No support vectors.");
      int evals = 0;

      for (int64_t j = 0; j < vector_count_; j++) {
        auto val = kernel_dot(x_data, current_weight_0, support_vectors_, feature_count_ * j,
                              feature_count_, get_kernel_type());
        kernels.push_back(val);
      }
      votes.resize(class_count_, 0);
      for (int64_t i = 0; i < class_count_; i++) {        // for each class
        for (int64_t j = i + 1; j < class_count_; j++) {  // for each class
          double sum = 0;
          int64_t start_index_i = starting_vector_[i];  // *feature_count_;
          int64_t start_index_j = starting_vector_[j];  // *feature_count_;

          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t class_j_support_count = vectors_per_class_[j];

          int64_t pos1 = (vector_count_) * (j - 1);
          int64_t pos2 = (vector_count_) * (i);
          const float* val1 = &(coefficients_[pos1 + start_index_i]);
          const float* val2 = &(kernels[start_index_i]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = &(coefficients_[pos2 + start_index_j]);
          val2 = &(kernels[start_index_j]);
          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[evals];
          scores.push_back(static_cast<float>(sum));
          ++(votes[sum > 0 ? i : j]);
          ++evals;  //index into rho
        }
      }
    }

    if (!proba_.empty() && mode_ == SVM_TYPE::SVM_SVC) {
      //compute probabilities from the scores
      int64_t num = class_count_ * class_count_;
      std::vector<float> probsp2(num, 0.f);
      std::vector<float> estimates(class_count_, 0.f);
      int64_t index = 0;
      for (int64_t i = 0; i < class_count_; ++i) {
        int64_t p1 = i * class_count_ + i + 1;
        int64_t p2 = (i + 1) * class_count_ + i;
        for (int64_t j = i + 1; j < class_count_; ++j, ++index) {
          float val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
          float val2 = std::max(val1, 1.0e-7f);
          val2 = std::min(val2, 1 - 1.0e-7f);
          probsp2[p1] = val2;
          probsp2[p2] = 1 - val2;
          ++p1;
          p2 += class_count_;
        }
      }
      multiclass_probability(class_count_, probsp2, estimates);
      // copy probabilities back into scores
      scores.resize(estimates.size());
      std::copy(estimates.begin(), estimates.end(), scores.begin());
    }

    float max_weight = 0;
    if (!votes.empty()) {
      auto it_maxvotes = std::max_element(votes.begin(), votes.end());
      maxclass = std::distance(votes.begin(), it_maxvotes);
    } else {
      auto it_max_weight = std::max_element(scores.begin(), scores.end());
      maxclass = std::distance(scores.begin(), it_max_weight);
      max_weight = *it_max_weight;
    }

    // write top class
    // onnx specs expects one column per class.
    int write_additional_scores = -1;
    if (rho_.size() == 1) {
      if (using_strings_) {
        write_additional_scores = _set_score_svm<std::string>(
            Y, max_weight, maxclass, n, post_transform_, proba_,
            weights_are_all_positive_, classlabels_strings_, "1", "0");
      } else {
        write_additional_scores = _set_score_svm<int64_t>(
            Y, max_weight, maxclass, n, post_transform_, proba_,
            weights_are_all_positive_, classlabels_ints_, 1, 0);
      }
    } else {  //multiclass
      if (using_strings_) {
        Y->template MutableData<std::string>()[n] = classlabels_strings_[maxclass];
      } else {
        Y->template MutableData<int64_t>()[n] = classlabels_ints_[maxclass];
      }
    }

    write_scores(scores, post_transform_, zindex, Z, write_additional_scores);
    zindex += scores.size();
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
