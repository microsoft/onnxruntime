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
  ONNXRUNTIME_ENFORCE(info.GetAttrs<float>("rho", rho_).IsOK());
  ONNXRUNTIME_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());

  // prob_a and prob_b are optional for Z output
  ONNXRUNTIME_ENFORCE(proba_.size() == probb_.size());

  // one of these should be valid
  ONNXRUNTIME_ENFORCE(info.GetAttrs<std::string>("classlabels_strings", classlabels_strings_).IsOK() ||
                      info.GetAttrs<int64_t>("classlabels_ints", classlabels_ints_).IsOK());

  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  using_strings_ = false;
  if (classlabels_strings_.size() > 0) {
    using_strings_ = true;
    class_count_ = classlabels_strings_.size();
  } else if (classlabels_ints_.size() > 0) {
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
  ONNXRUNTIME_ENFORCE(classlabels_strings_.size() > 0 || classlabels_ints_.size() > 0);
  ONNXRUNTIME_ENFORCE(proba_.size() == probb_.size());
  ONNXRUNTIME_ENFORCE(coefficients_.size() > 0);
  weights_are_all_positive_ = true;
  for (int64_t i = 0; i < static_cast<int64_t>(coefficients_.size()); i++) {
    if (coefficients_[i] < 0) {
      weights_are_all_positive_ = false;
      break;
    }
  }
}

#define SETSCORESVM(typlabels, classlabels, posclass, negclass)                     \
  if (classlabels.size() == 2) {                                                    \
    write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0; \
    if (proba_.size() == 0) {                                                       \
      if (weights_are_all_positive_ && maxweight >= 0.5)                            \
        Y->template MutableData<typlabels>()[n] = classlabels[1];                   \
      else if (maxweight > 0 && !weights_are_all_positive_)                         \
        Y->template MutableData<typlabels>()[n] = classlabels[1];                   \
      else                                                                          \
        Y->template MutableData<typlabels>()[n] = classlabels[maxclass];            \
    } else {                                                                        \
      Y->template MutableData<typlabels>()[n] = classlabels[maxclass];              \
    }                                                                               \
  } else if (maxweight > 0) {                                                       \
    Y->template MutableData<typlabels>()[n] = posclass;                             \
  } else {                                                                          \
    Y->template MutableData<typlabels>()[n] = negclass;                             \
  }

template <typename T>
Status SVMClassifier<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N}));
  std::vector<int64_t> dims;
  int64_t nc = (proba_.size() > 0 || vector_count_ == 0)
                   ? class_count_
                   : (class_count_ > 2 ? class_count_ * (class_count_ - 1) / 2 : 2);
  dims = {static_cast<int64_t>(N), static_cast<int64_t>(nc)};
  Tensor* Z = ctx->Output(1, TensorShape(dims));


  const auto* x_data = X->template Data<T>();
  int64_t zindex = 0;

  for (int64_t n = 0; n < N; n++)  //for each example
  {
    int64_t current_weight_0 = n * stride;
    int64_t maxclass = -1;
    std::vector<float> decisions;
    std::vector<float> scores;
    std::vector<float> kernels;
    std::vector<int64_t> votes;
    float sum;

    if (vector_count_ == 0 && mode_ == SVM_TYPE::SVM_LINEAR) {
      // This was in the original code but it does not appear in libsvm or scikit-learn.
      for (int64_t j = 0; j < class_count_; j++) {  //for each class
        float val = kernel_dot(x_data, current_weight_0, coefficients_, feature_count_ * j,
                               feature_count_, get_kernel_type());
        val += rho_[0];
        scores.push_back(val);
      }
    } else {
      if (vector_count_ == 0)
        return Status(common::ONNXRUNTIME, common::FAIL, "No support vectors.");
      int evals = 0;

      for (int64_t j = 0; j < vector_count_; j++) {
        float val = kernel_dot(x_data, current_weight_0, support_vectors_, feature_count_ * j,
                               feature_count_, get_kernel_type());
        kernels.push_back(val);
      }
      votes.resize(class_count_, 0);
      for (int64_t i = 0; i < class_count_; i++) {        // for each class
        for (int64_t j = i + 1; j < class_count_; j++) {  // for each class
          sum = 0;
          int64_t start_index_i = starting_vector_[i];  // *feature_count_;
          int64_t start_index_j = starting_vector_[j];  // *feature_count_;

          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t class_j_support_count = vectors_per_class_[j];

          int64_t pos1 = (vector_count_) * (j - 1);
          int64_t pos2 = (vector_count_) * (i);
          float* val1 = (float*)&(coefficients_[pos1 + start_index_i]);
          float* val2 = (float*)&(kernels[start_index_i]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = (float*)&(coefficients_[pos2 + start_index_j]);
          val2 = (float*)&(kernels[start_index_j]);
          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[evals];
          scores.push_back(sum);
          ++(votes[sum > 0 ? i : j]);
          ++evals;  //index into rho
        }
      }
    }

    if (proba_.size() > 0 && mode_ == SVM_TYPE::SVM_SVC) {
      //compute probabilities from the scores
      int64_t num = class_count_ * class_count_;
      std::vector<float> probsp2(num, 0.f);
      std::vector<float> estimates(class_count_, 0.f);
      int64_t index = 0;
      int64_t p1, p2;
      for (int64_t i = 0; i < class_count_; ++i) {
        p1 = i * class_count_ + i + 1;
        p2 = (i + 1) * class_count_ + i;
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
      //copy probabilities back into scores
      scores.resize(estimates.size());
      std::copy(estimates.begin(), estimates.end(), scores.begin());
#if false
      // Normalization OVR as implemented in scikit-learn.
    } else if (proba_.size() == 0) {
      // Applies function first part of _ovr_decision_function (scikit-learn).
      // ONNX specs imposes one column per class. Libsvm does not do it, scikit-learn does.
      // If OVR_NORM is defined the function also applies normalisation as
      // scikit-learn would do in function _ovr_decision_function.
      // This method has a major drawback because the scores depends on the other observations
      // due to a rescaling based on a maximum obtained for all predictions
      // (observations are not independant).
      /*
      for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            k += 1 

        max_confidences = sum_of_confidences.max()
        min_confidences = sum_of_confidences.min()

        if max_confidences == min_confidences:
            return votes

        eps = np.finfo(sum_of_confidences.dtype).eps
        max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
        scale = (0.5 - eps) / max_abs_confidence
        return votes + sum_of_confidences * scale
      */
      std::vector<float> conf(class_count_, 0.f);
      float* ps = &(scores[0]);
      for (int64_t i = 0; i < class_count_; ++i) {
        for (int64_t j = i + 1; j < class_count_; ++j, ++ps) {
          conf[i] += *ps;
          conf[j] -= *ps;
        }
      }

      scores = conf;
#endif
    }

    int64_t maxvotes = 0;
    double maxweight = 0;
    if (votes.size() > 0) {
      auto it_maxvotes = std::max_element(votes.begin(), votes.end());
      maxclass = std::distance(votes.begin(), it_maxvotes);
      maxvotes = *it_maxvotes;
    } else {
      auto it_maxweight = std::max_element(scores.begin(), scores.end());
      maxclass = std::distance(scores.begin(), it_maxweight);
      maxweight = *it_maxweight;
    }

    // write top class
    // onnx specs expects one column per class.
    int write_additional_scores = -1;
    if (rho_.size() == 1) {
      if (using_strings_) {
        SETSCORESVM(std::string, classlabels_strings_, "1", "0")
      } else {
        SETSCORESVM(int64_t, classlabels_ints_, 1, 0)
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
