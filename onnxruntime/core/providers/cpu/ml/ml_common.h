// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace ml {  // name space for onnx.ml operators

enum class OUTPUT_MODE {
  TOPCLASS,
  TOPCLASS_ANDSCORE,
  ALL_SCORES
};

enum class NODE_MODE {
  BRANCH_LEQ,
  BRANCH_LT,
  BRANCH_GTE,
  BRANCH_GT,
  BRANCH_EQ,
  BRANCH_NEQ,
  LEAF
};

static inline NODE_MODE MakeTreeNodeMode(const std::string& input) {
  if (input == "BRANCH_LEQ") {
    return NODE_MODE::BRANCH_LEQ;
  }
  if (input == "LEAF") {
    return NODE_MODE::LEAF;
  }
  if (input == "BRANCH_LT") {
    return NODE_MODE::BRANCH_LT;
  }
  if (input == "BRANCH_GTE") {
    return NODE_MODE::BRANCH_GTE;
  }
  if (input == "BRANCH_GT") {
    return NODE_MODE::BRANCH_GT;
  }
  if (input == "BRANCH_EQ") {
    return NODE_MODE::BRANCH_EQ;
  }
  return NODE_MODE::BRANCH_NEQ;
}

enum class POST_EVAL_TRANSFORM {
  NONE,
  LOGISTIC,
  SOFTMAX,
  SOFTMAX_ZERO,
  PROBIT
};

static inline POST_EVAL_TRANSFORM MakeTransform(const std::string& input) {
  if (input == "NONE") {
    return POST_EVAL_TRANSFORM::NONE;
  }
  if (input == "LOGISTIC") {
    return POST_EVAL_TRANSFORM::LOGISTIC;
  }
  if (input == "SOFTMAX") {
    return POST_EVAL_TRANSFORM::SOFTMAX;
  }
  if (input == "SOFTMAX_ZERO") {
    return POST_EVAL_TRANSFORM::SOFTMAX_ZERO;
  }
  return POST_EVAL_TRANSFORM::PROBIT;
}

enum class AGGREGATE_FUNCTION {
  AVERAGE,
  SUM,
  MIN,
  MAX
};

static inline AGGREGATE_FUNCTION MakeAggregateFunction(const std::string& input) {
  if (input == "AVERAGE") {
    return AGGREGATE_FUNCTION::AVERAGE;
  }
  if (input == "SUM") {
    return AGGREGATE_FUNCTION::SUM;
  }
  if (input == "MIN") {
    return AGGREGATE_FUNCTION::MIN;
  }
  return AGGREGATE_FUNCTION::MAX;
}

enum class CAST_TO {
  TO_FLOAT,
  TO_STRING,
  TO_INT64
};

static inline CAST_TO MakeCast(const std::string& input) {
  if (input == "TO_FLOAT") {
    return CAST_TO::TO_FLOAT;
  }
  if (input == "TO_STRING") {
    return CAST_TO::TO_STRING;
  }
  if (input == "TO_INT64") {
    return CAST_TO::TO_INT64;
  }
  ORT_THROW("Invalid CAST_TO value of ", input, " Expected TO_FLOAT, TO_STRING or TO_INT64");
}

enum PACK_MAP {
  DENSE,
  SPARSE
};

static inline PACK_MAP MakePack(const std::string& input) {
  if (input == "DENSE") {
    return PACK_MAP::DENSE;
  }
  if (input == "SPARSE") {
    return PACK_MAP::SPARSE;
  }
  ORT_THROW("Invalid PACK_MAP value of ", input, " Expected DENSE or SPARSE");
}

enum KERNEL {
  LINEAR,
  POLY,
  RBF,
  SIGMOID
};

static inline KERNEL MakeKernel(const std::string& input) {
  if (input == "LINEAR") {
    return KERNEL::LINEAR;
  }
  if (input == "POLY") {
    return KERNEL::POLY;
  }
  if (input == "RBF") {
    return KERNEL::RBF;
  }
  return KERNEL::SIGMOID;
}

enum NORMALIZE {
  NMAX,
  L1,
  L2
};

static inline NORMALIZE MakeNormalize(const std::string& input) {
  if (input == "MAX") {
    return NORMALIZE::NMAX;
  }
  if (input == "L1") {
    return NORMALIZE::L1;
  }
  if (input == "L2") {
    return NORMALIZE::L2;
  }
  ORT_THROW("Invalid normalize value of ", input);
}

enum class SVM_TYPE {
  SVM_LINEAR,
  SVM_SVC
};

static inline float ErfInv(float x) {
  float sgn = x < 0 ? -1.0f : 1.0f;
  x = (1 - x) * (1 + x);
  float log = std::log(x);
  float v = 2 / (3.14159f * 0.147f) + 0.5f * log;
  float v2 = 1 / (0.147f) * log;
  float v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}

//https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf
static inline void multiclass_probability(int64_t classcount, const std::vector<float>& r, std::vector<float>& p) {
  int64_t sized2 = classcount * classcount;
  std::vector<float> Q;
  std::vector<float> Qp;
  for (int64_t k = 0; k < sized2; k++) {
    Q.push_back(0);
  }
  for (int64_t k = 0; k < classcount; k++) {
    Qp.push_back(0);
  }
  float eps = 0.005f / static_cast<float>(classcount);
  for (int64_t i = 0; i < classcount; i++) {
    p[i] = 1.0f / static_cast<float>(classcount);  // Valid if k = 1
    for (int64_t j = 0; j < i; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = Q[j * classcount + i];
    }
    for (int64_t j = i + 1; j < classcount; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = -r[j * classcount + i] * r[i * classcount + j];
    }
  }
  for (int64_t loop = 0; loop < 100; loop++) {
    // stopping condition, recalculate QP,pQP for numerical accuracy
    float pQp = 0;
    for (int64_t i = 0; i < classcount; i++) {
      Qp[i] = 0;
      for (int64_t j = 0; j < classcount; j++) {
        Qp[i] += Q[i * classcount + j] * p[j];
      }
      pQp += p[i] * Qp[i];
    }
    float max_error = 0;
    for (int64_t i = 0; i < classcount; i++) {
      float error = std::fabs(Qp[i] - pQp);
      if (error > max_error) {
        max_error = error;
      }
    }
    if (max_error < eps) break;

    for (int64_t i = 0; i < classcount; i++) {
      float diff = (-Qp[i] + pQp) / Q[i * classcount + i];
      p[i] += diff;
      pQp = (pQp + diff * (diff * Q[i * classcount + i] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
      for (int64_t j = 0; j < classcount; j++) {
        Qp[j] = (Qp[j] + diff * Q[i * classcount + j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
    }
  }
}

static const float ml_sqrt2 = 1.41421356f;

static inline float ComputeLogistic(float val) {
  float v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}

static inline float ComputeProbit(float val) {
  return ml_sqrt2 * ErfInv(2 * val - 1);
}

static inline float sigmoid_probability(float score, float proba, float probb) {
  float val = score * proba + probb;
  return 1 - ComputeLogistic(val);  // ref: https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
}

static inline void ComputeSoftmax(const gsl::span<float>& values) {
  // TODO: Replace this with usage of code in Softmax operator

  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float this_sum = 0.f;
  for (float& value : values) {
    value = std::exp(value - v_max);
    this_sum += value;
  }
  for (float& value : values)
    value /= this_sum;
}

static inline void ComputeSoftmax(std::vector<float>& values) {
  auto span = gsl::make_span(values);
  ComputeSoftmax(span);
}

//this function skips zero values (since exp(0) is non zero)
static inline void ComputeSoftmaxZero(const gsl::span<float>& values) {
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float exp_neg_v_max = std::exp(-v_max);
  float this_sum = 0.f;
  for (float& value : values) {
    if (value > 0.0000001f || value < -0.0000001f) {
      value = std::exp(value - v_max);
      this_sum += value;
    } else {
      value *= exp_neg_v_max;
    }
  }
  for (float& value : values)
    value /= this_sum;
}

static inline void ComputeSoftmaxZero(std::vector<float>& values) {
  auto span = gsl::make_span(values);
  ComputeSoftmaxZero(span);
}

template <typename T>
static void write_scores(std::vector<T>& scores, POST_EVAL_TRANSFORM post_transform,
                         T* Z, int add_second_class) {
  if (scores.size() >= 2) {
    switch (post_transform) {
      case POST_EVAL_TRANSFORM::PROBIT:
        for (auto it = scores.cbegin(); it != scores.cend(); ++it, ++Z)
          *Z = static_cast<T>(ComputeProbit(static_cast<float>(*it)));
        break;
      case POST_EVAL_TRANSFORM::LOGISTIC:
        for (auto it = scores.cbegin(); it != scores.cend(); ++it, ++Z)
          *Z = static_cast<T>(ComputeLogistic(static_cast<float>(*it)));
        break;
      case POST_EVAL_TRANSFORM::SOFTMAX:
        ComputeSoftmax(scores);
        memcpy(Z, scores.data(), scores.size() * sizeof(T));
        break;
      case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
        ComputeSoftmaxZero(scores);
        memcpy(Z, scores.data(), scores.size() * sizeof(T));
        break;
      default:
      case POST_EVAL_TRANSFORM::NONE:
        memcpy(Z, scores.data(), scores.size() * sizeof(T));
        break;
    }
  } else if (scores.size() == 1) {  //binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      scores[0] = static_cast<T>(ComputeProbit(static_cast<float>(scores[0])));
      *Z = scores[0];
    } else {
      switch (add_second_class) {
        case 0:  //0=all positive weights, winning class is positive
          scores.push_back(scores[0]);
          scores[0] = 1.f - scores[0];  //put opposite score in positive slot
          *Z = scores[0];
          *(Z + 1) = scores[1];
          break;
        case 1:  //1 = all positive weights, winning class is negative
          scores.push_back(scores[0]);
          scores[0] = 1.f - scores[0];  //put opposite score in positive slot
          *Z = scores[0];
          *(Z + 1) = scores[1];
          break;
        case 2:
        case 3:  //2 = mixed weights, winning class is positive
          if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
            scores.push_back(static_cast<T>(ComputeLogistic(static_cast<float>(scores[0]))));
            scores[0] = static_cast<T>(ComputeLogistic(static_cast<float>(-scores[0])));
          } else {
            scores.push_back(scores[0]);
            scores[0] = -scores[0];
          }
          *Z = scores[0];
          *(Z + 1) = scores[1];
          break;
        default:
          *Z = scores[0];
          break;
      }
    }
  }
}

template <typename T>
static void write_scores(std::vector<T>& scores, POST_EVAL_TRANSFORM post_transform, int64_t write_index, Tensor* Z,
                         int add_second_class) {
  T* out_p = Z->template MutableData<T>() + write_index;
  size_t len;
  if (!IAllocator::CalcMemSizeForArray(scores.size(), sizeof(T), &len)) {
    ORT_THROW("length overflow");
  }
  write_scores(scores, post_transform, out_p, add_second_class);
}

// TODO: Starting with just the pieces needed for LinearRegressor from write_scores (see above).
//       Will see what can be sensibly added to a batched in-place update of the scores for LinearClassifier, the SVM*
//       and TreeEnsemble* ops when updating those.
//       Attempted to parallelize the calculations if the number of scores to process was large, but no clear benefit
//       was seen from testing with the arbitrary values of 1000 scores per threads.
template <typename T>
void batched_update_scores_inplace(gsl::span<T> scores, int64_t num_batches_in, int64_t batch_size,
                                   POST_EVAL_TRANSFORM post_transform, int add_second_class,
                                   concurrency::ThreadPool* threadpool) {
  if (batch_size < 1)
    return;

  SafeInt<int32_t> num_batches(num_batches_in);
  SafeInt<int32_t> num_scores = num_batches * batch_size;
  SafeInt<int32_t> expected_num_scores = num_scores * (batch_size == 1 && add_second_class >= 0 ? 2 : 1);
  ORT_ENFORCE(scores.size() == static_cast<size_t>(expected_num_scores));

  ORT_UNUSED_PARAMETER(threadpool);  // TBD whether we need to parallelize code here

  // convert from span to pointer for efficiency. we've checked scores.size() matches num_scores so don't need the
  // extra checking/overhead from using operator[] for each access
  T* s = scores.data();
  const T* s_end = s + static_cast<int32_t>(num_scores);

  if (batch_size > 1) {
    switch (post_transform) {
      case POST_EVAL_TRANSFORM::PROBIT: {
        while (s < s_end) {
          *s = ComputeProbit(*s);
          ++s;
        }
        break;
      }
      case POST_EVAL_TRANSFORM::LOGISTIC: {
        MlasComputeLogistic(s, s, scores.size());
        break;
      }
      case POST_EVAL_TRANSFORM::SOFTMAX: {
        while (s < s_end) {
          gsl::span<float> scores_for_batch(s, s + batch_size);
          ComputeSoftmax(scores_for_batch);
          s += batch_size;
        }
        break;
      }
      case POST_EVAL_TRANSFORM::SOFTMAX_ZERO: {
        while (s < s_end) {
          gsl::span<float> scores_for_batch(s, s + batch_size);
          ComputeSoftmaxZero(scores_for_batch);
          s += batch_size;
        }
        break;
      }
      case POST_EVAL_TRANSFORM::NONE:
      default:
        break;
    }
  } else {  // binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      while (s < s_end) {
        *s = ComputeProbit(*s);
        ++s;
      }
    } else if (add_second_class >= 0) {
      // in this case we have a buffer that holds 2x scores. the actual scores are at the start of the buffer,
      // and for each score we need 2 entries.
      // process the scores from the back to the front so we don't need a separate buffer.
      std::function<void(const float score, float* output)> update_scores;

      switch (add_second_class) {
        case 0:
        case 1:
          update_scores = [](const float score, float* output) {
            *output++ = 1.f - score;
            *output = score;
          };
          break;

        case 2:  //2 = mixed weights, winning class is positive
        case 3:  //3 = mixed weights, winning class is negative
          if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
            update_scores = [](const float score, float* output) {
              *output++ = ComputeLogistic(-score);
              *output = ComputeLogistic(score);
            };
          } else {
            update_scores = [](const float score, float* output) {
              *output++ = -score;
              *output = score;
            };
          }
          break;

        default:
          ORT_THROW("Unexpected value for 'add_second_class' of ", add_second_class);
      }

      const float* cur_in = s_end;
      float* cur_out = &*scores.end();
      while (cur_in > s) {
        --cur_in;
        cur_out -= 2;
        update_scores(*cur_in, cur_out);
      }
    }
  }
}
}  // namespace ml
}  // namespace onnxruntime
