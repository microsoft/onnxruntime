// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/svmclassifier.h"
#include "core/platform/threadpool.h"
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    SVMClassifier,
    1,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<float>(),
                                  DataTypeImpl::GetTensorType<double>(),
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}),
    SVMClassifier);

SVMClassifier::SVMClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      SVMCommon(info),
      vectors_per_class_(info.GetAttrsOrDefault<int64_t>("vectors_per_class")),
      proba_(info.GetAttrsOrDefault<float>("prob_a")),
      probb_(info.GetAttrsOrDefault<float>("prob_b")),
      support_vectors_(info.GetAttrsOrDefault<float>("support_vectors")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_THROW_IF_ERROR(info.GetAttrs<float>("rho", rho_));
  ORT_THROW_IF_ERROR(info.GetAttrs<float>("coefficients", coefficients_));

  // prob_a and prob_b are optional for Z output
  ORT_ENFORCE(proba_.size() == probb_.size());

  // one of these should be valid
  ORT_ENFORCE(info.GetAttrs<std::string>("classlabels_strings", classlabels_strings_).IsOK() ||
              info.GetAttrs<int64_t>("classlabels_ints", classlabels_ints_).IsOK());

  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (size_t i = 0; i < vectors_per_class_.size(); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += narrow<ptrdiff_t>(vectors_per_class_[i]);
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
    feature_count_ = support_vectors_.size() / vector_count_;  // length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size() / class_count_;  // liblinear mode
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }

  ORT_ENFORCE(classlabels_strings_.size() > 0 || classlabels_ints_.size() > 0);
  ORT_ENFORCE(proba_.size() == probb_.size());
  ORT_ENFORCE(coefficients_.size() > 0);
  weights_are_all_positive_ = std::all_of(coefficients_.cbegin(), coefficients_.cend(),
                                          [](float value) { return value >= 0.f; });
}

template <typename LabelType>
static void ChooseClass(Tensor& output, const int64_t output_idx, float max_weight, const int64_t maxclass,
                        bool have_proba, bool weights_are_all_positive,
                        const std::vector<LabelType>& classlabels,
                        const LabelType& posclass, const LabelType& negclass) {
  LabelType& output_data = *(output.MutableData<LabelType>() + output_idx);

  if (classlabels.size() == 2) {
    if (!have_proba) {
      if (weights_are_all_positive && max_weight >= 0.5)
        output_data = classlabels[1];
      else if (max_weight > 0 && !weights_are_all_positive)
        output_data = classlabels[1];
      else
        output_data = classlabels[onnxruntime::narrow<size_t>(maxclass)];
    } else {
      output_data = classlabels[onnxruntime::narrow<size_t>(maxclass)];
    }
  } else if (max_weight > 0) {
    output_data = posclass;
  } else {
    output_data = negclass;
  }
}

Status SVMClassifier::Compute(OpKernelContext* ctx) const {
  Status status = Status::OK();
  const auto& X = *ctx->Input<Tensor>(0);
  const auto& x_shape = X.Shape();

  AllocatorPtr allocator;
  auto element_type = X.GetElementType();
  gsl::span<const float> x_data;
  float* tmp_data = nullptr;

  if (element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    x_data = X.DataAsSpan<float>();
  } else {
    // need to cast the input to float so we can use the fast GEMM implementations
    auto num_elements = onnxruntime::narrow<size_t>(x_shape.Size());

    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
    tmp_data = static_cast<float*>(allocator->AllocArray(num_elements, sizeof(float)));

    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        auto in_vector = ConstEigenVectorMap<double>(X.Data<double>(), num_elements);
        auto output_vector = EigenVectorMap<float>(tmp_data, num_elements);
        output_vector = in_vector.cast<float>();
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        auto in_vector = ConstEigenVectorMap<int32_t>(X.Data<int32_t>(), num_elements);
        auto output_vector = EigenVectorMap<float>(tmp_data, num_elements);
        output_vector = in_vector.cast<float>();
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        auto in_vector = ConstEigenVectorMap<int64_t>(X.Data<int64_t>(), num_elements);
        auto output_vector = EigenVectorMap<float>(tmp_data, num_elements);
        output_vector = in_vector.cast<float>();
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type of ", element_type);
    }

    x_data = gsl::make_span<const float>(tmp_data, num_elements);
  }

  status = ComputeImpl(*ctx, x_data, x_shape);

  if (element_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    allocator->Free(tmp_data);
  }

  return status;
}

Status SVMClassifier::ComputeImpl(OpKernelContext& ctx,
                                  gsl::span<const float> x_data, const TensorShape& x_shape) const {
  concurrency::ThreadPool* threadpool = ctx.GetOperatorThreadPool();

  const auto num_batches = SafeInt<int32_t>(x_shape.NumDimensions() == 1 ? 1 : x_shape[0]);

  // Total number of classifiers comparing pairs between the classes
  // e.g. if you have A, B C and D classes, the number of classifiers to compare between each pair is 6
  //      with AB, AC, AD, BC, BD and CD
  const int64_t num_classifiers = class_count_ * (class_count_ - 1) / 2;  // == (class_count_-1)!
  const int64_t class_count_squared = class_count_ * class_count_;
  const bool have_proba = proba_.size() > 0;

  int64_t final_scores_per_batch = class_count_;
  if (mode_ == SVM_TYPE::SVM_SVC && !have_proba) {
    if (class_count_ > 2)
      final_scores_per_batch = num_classifiers;
    else
      final_scores_per_batch = 2;
  }

  // Input shapes
  // X: [num_batches, feature_count_] where features could be coefficients or support vectors
  // coefficients_: if linear [class_count, feature_count]
  //                else      [num_classes - 1, vector_count_]
  // support_vectors_ : [vector_count_, feature_count_]

  // both outputs are required so can't be nullptr
  Tensor& Y = *ctx.Output(0, {num_batches});
  Tensor& Z = *ctx.Output(1, {num_batches, final_scores_per_batch});

  auto final_scores = Z.MutableDataAsSpan<float>();

  std::vector<float> kernels_data;
  std::vector<int64_t> votes_data;

  std::vector<float> classifier_scores_data;
  std::vector<float> probsp2_data;

  if (mode_ == SVM_TYPE::SVM_SVC && have_proba) {
    probsp2_data.resize(num_batches * class_count_squared, 0.f);
  }

  int write_additional_scores = -1;
  int64_t num_scores_per_batch = class_count_;

  if (mode_ == SVM_TYPE::SVM_SVC && !have_proba) {
    num_scores_per_batch = num_classifiers;
    if (class_count_ <= 2) {
      write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
    }
  }

  if (mode_ == SVM_TYPE::SVM_LINEAR) {
    // scores_data.resize(num_batches * class_count_);
    // auto out = gsl::make_span<float>(scores_data.data(), scores_data.size());

    // combine the coefficients with the input data and apply the kernel type
    batched_kernel_dot<float>(x_data, coefficients_, num_batches, class_count_, feature_count_, rho_[0], final_scores,
                              threadpool);

  } else {
    gsl::span<float> classifier_scores;

    // if we have one classifier, are writing directly to the final buffer,
    // and will add an additional score in the results, leave a space between each classifier score so that
    // we can parallelize the batch processing below.
    int64_t num_slots_per_iteration = write_additional_scores >= 0 ? 2 : num_classifiers;

    if (have_proba) {
      // we will write num_batches * num_classifiers scores first, and transform those to num_batches * class_count_,
      // so need to use a separate buffer for the first scoring.
      classifier_scores_data.resize(num_batches * num_classifiers);
      classifier_scores = gsl::make_span<float>(classifier_scores_data.data(), classifier_scores_data.size());
    } else {
      // we will write directly to the final scores buffer
      // num_scores_per_batch = num_classifiers;
      // assert(num_scores_per_batch == final_scores_per_batch);
      classifier_scores = final_scores;
    }

    kernels_data.resize(num_batches * vector_count_);
    votes_data.resize(num_batches * class_count_, 0);

    auto kernels_span = gsl::make_span<float>(kernels_data.data(), kernels_data.size());
    auto votes_span = gsl::make_span<int64_t>(votes_data.data(), votes_data.size());

    // combine the input data with the support vectors and apply the kernel type
    // output is {num_batches, vector_count_}
    batched_kernel_dot<float>(x_data, support_vectors_, num_batches, vector_count_, feature_count_, 0.f, kernels_span,
                              threadpool);

    for (int64_t n = 0; n < num_batches; n++) {
      // reduce scores from kernels using coefficients, taking into account the varying number of support vectors
      // per class.
      // coefficients: [num_classes - 1, vector_count_]
      //
      // e.g. say you have 3 classes, with 3 x 3 coefficients
      //
      // AA AB AC
      // BA BB BC
      // CA CB CC
      //
      // you can remove the diagonal line of items comparing a class with itself leaving one less row.
      //
      // BA AB AC
      // CA CB BC
      //
      // for each class there is a coefficient per support vector, and a class has one or more support vectors.
      //
      // Combine the scores for the two combinations for two classes with their coefficient.
      // e.g. AB combines with BA.
      // If A has 3 support vectors and B has 2, there's a 3x2 block for AB and a 2x3 block for BA to combine

      auto cur_kernels = kernels_span.subspan(n * SafeInt<size_t>(vector_count_), onnxruntime::narrow<size_t>(vector_count_));
      auto cur_scores = classifier_scores.subspan(n * SafeInt<size_t>(num_slots_per_iteration), onnxruntime::narrow<size_t>(num_classifiers));
      auto cur_votes = votes_span.subspan(n * SafeInt<size_t>(class_count_), onnxruntime::narrow<size_t>(class_count_));
      auto scores_iter = cur_scores.begin();

      size_t classifier_idx = 0;
      for (int64_t i = 0; i < class_count_ - 1; i++) {
        int64_t start_index_i = starting_vector_[onnxruntime::narrow<size_t>(i)];  // start of support vectors for class i
        int64_t class_i_support_count = vectors_per_class_[onnxruntime::narrow<size_t>(i)];
        int64_t i_coeff_row_offset = vector_count_ * i;

        for (int64_t j = i + 1; j < class_count_; j++) {
          int64_t start_index_j = starting_vector_[onnxruntime::narrow<size_t>(j)];  // start of support vectors for class j
          int64_t class_j_support_count = vectors_per_class_[onnxruntime::narrow<size_t>(j)];
          int64_t j_coeff_row_offset = vector_count_ * (j - 1);

          double sum = 0;

          const float* val1 = &(coefficients_[j_coeff_row_offset + SafeInt<size_t>(start_index_i)]);
          const float* val2 = &(cur_kernels[onnxruntime::narrow<size_t>(start_index_i)]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = &(coefficients_[i_coeff_row_offset + SafeInt<size_t>(start_index_j)]);
          val2 = &(cur_kernels[onnxruntime::narrow<size_t>(start_index_j)]);

          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[classifier_idx++];

          *scores_iter++ = static_cast<float>(sum);
          ++(cur_votes[onnxruntime::narrow<size_t>(sum > 0 ? i : j)]);
        }
      }
    }
  }

  auto finalize_batch = [this, &final_scores, final_scores_per_batch,
                         have_proba, &probsp2_data, class_count_squared,
                         &classifier_scores_data, num_classifiers, &votes_data, &Y,
                         num_scores_per_batch, write_additional_scores](ptrdiff_t idx) {
    int n = SafeInt<int32_t>(idx);  // convert to a usable sized type
    auto cur_scores = final_scores.subspan(n * SafeInt<size_t>(final_scores_per_batch), onnxruntime::narrow<size_t>(final_scores_per_batch));

    if (mode_ == SVM_TYPE::SVM_SVC && have_proba) {
      auto probsp2 = gsl::make_span<float>(probsp2_data.data() + (n * class_count_squared), onnxruntime::narrow<size_t>(class_count_squared));

      float* classifier_scores = classifier_scores_data.data() + (n * num_classifiers);

      size_t index = 0;
      for (int64_t i = 0; i < class_count_ - 1; ++i) {
        int64_t p1 = i * class_count_ + i + 1;
        int64_t p2 = (i + 1) * class_count_ + i;
        for (int64_t j = i + 1; j < class_count_; ++j, ++index) {
          float val1 = sigmoid_probability(classifier_scores[index], proba_[index], probb_[index]);
          float val2 = std::max(val1, 1.0e-7f);
          val2 = std::min(val2, 1 - 1.0e-7f);
          probsp2[onnxruntime::narrow<size_t>(p1)] = val2;
          probsp2[onnxruntime::narrow<size_t>(p2)] = 1 - val2;
          ++p1;
          p2 += class_count_;
        }
      }

      // expand scores from num_classifiers to class_count_
      multiclass_probability(class_count_, probsp2, cur_scores);
    }

    float max_weight = 0;
    int64_t maxclass = -1;
    if (votes_data.size() > 0) {
      auto votes = gsl::make_span<int64_t>(votes_data.data() + (n * class_count_), onnxruntime::narrow<size_t>(class_count_));
      auto it_maxvotes = std::max_element(votes.begin(), votes.end());
      maxclass = std::distance(votes.begin(), it_maxvotes);
    } else {
      auto it_max_weight = std::max_element(cur_scores.begin(), cur_scores.end());
      maxclass = std::distance(cur_scores.begin(), it_max_weight);
      max_weight = *it_max_weight;
    }

    // write top class
    // onnx specs expects one column per class.
    if (num_classifiers == 1) {  // binary case
      if (using_strings_) {
        ChooseClass<std::string>(Y, n, max_weight, maxclass, have_proba, weights_are_all_positive_,
                                 classlabels_strings_, "1", "0");
      } else {
        ChooseClass<int64_t>(Y, n, max_weight, maxclass, have_proba, weights_are_all_positive_,
                             classlabels_ints_, 1, 0);
      }
    } else {  // multiclass
      if (using_strings_) {
        Y.MutableData<std::string>()[n] = classlabels_strings_[onnxruntime::narrow<size_t>(maxclass)];
      } else {
        Y.MutableData<int64_t>()[n] = classlabels_ints_[onnxruntime::narrow<size_t>(maxclass)];
      }
    }

    // write the score for this batch
    // as we parallelize the batch processing we want to update the final scores for each batch in the separate threads
    batched_update_scores_inplace<float>(cur_scores, 1, num_scores_per_batch, post_transform_,
                                         write_additional_scores, true, nullptr);
  };

  // TODO: Refine this rough metric to choose when to parallelize.
  if (num_batches > 512) {
    concurrency::ThreadPool::TryBatchParallelFor(threadpool, num_batches, finalize_batch, -1);
  } else {
    {
      for (ptrdiff_t i = 0; i < num_batches; ++i) {
        finalize_batch(i);
      }
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
