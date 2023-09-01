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
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace ml {  // name space for onnx.ml operators

enum class OUTPUT_MODE {
  TOPCLASS,
  TOPCLASS_ANDSCORE,
  ALL_SCORES
};

enum NODE_MODE : uint8_t {
  LEAF = 1,
  BRANCH_LEQ = 2,
  BRANCH_LT = 4,
  BRANCH_GTE = 6,
  BRANCH_GT = 8,
  BRANCH_EQ = 10,
  BRANCH_NEQ = 12
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
  float v = 2 / (static_cast<float>(M_PI) * 0.147f) + 0.5f * log;
  float v2 = 1 / (0.147f) * log;
  float v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}

// https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf
static inline void multiclass_probability(int64_t classcount,
                                          const gsl::span<const float>& r,
                                          const gsl::span<float>& p) {
  auto safe_int_classcount = SafeInt<size_t>(classcount);
  size_t sized2 = safe_int_classcount * classcount;
  std::vector<float> Q;
  std::vector<float> Qp;
  Q.assign(sized2, 0.f);
  Qp.assign(safe_int_classcount, 0.f);

  float eps = 0.005f / static_cast<float>(classcount);

  for (size_t i = 0; i < safe_int_classcount; i++) {
    p[i] = 1.0f / onnxruntime::narrow<float>(classcount);  // Valid if k = 1
    for (size_t j = 0; j < i; j++) {
      Q[i * safe_int_classcount + i] += r[j * safe_int_classcount + i] * r[j * safe_int_classcount + i];
      Q[i * safe_int_classcount + j] = Q[j * safe_int_classcount + i];
    }
    for (size_t j = i + 1; j < safe_int_classcount; j++) {
      Q[i * safe_int_classcount + i] += r[j * safe_int_classcount + i] * r[j * safe_int_classcount + i];
      Q[i * safe_int_classcount + j] = -r[j * safe_int_classcount + i] * r[i * safe_int_classcount + j];
    }
  }

  for (size_t loop = 0; loop < 100; loop++) {
    // stopping condition, recalculate QP,pQP for numerical accuracy
    float pQp = 0;
    for (size_t i = 0; i < safe_int_classcount; i++) {
      Qp[i] = 0;
      for (size_t j = 0; j < safe_int_classcount; j++) {
        Qp[i] += Q[i * safe_int_classcount + j] * p[j];
      }
      pQp += p[i] * Qp[i];
    }

    float max_error = 0;
    for (size_t i = 0; i < safe_int_classcount; i++) {
      float error = std::fabs(Qp[i] - pQp);
      if (error > max_error) {
        max_error = error;
      }
    }

    if (max_error < eps)
      break;

    for (size_t i = 0; i < safe_int_classcount; i++) {
      float diff = (-Qp[i] + pQp) / Q[i * safe_int_classcount + i];
      p[i] += diff;
      pQp = (pQp + diff * (diff * Q[i * safe_int_classcount + i] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
      for (size_t j = 0; j < safe_int_classcount; j++) {
        Qp[j] = (Qp[j] + diff * Q[i * safe_int_classcount + j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
    }
  }
}

static constexpr float ml_sqrt2 = static_cast<float>(M_SQRT2);

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

template <typename T>
static inline void ComputeSoftmax(gsl::span<T>& values) {
  // TODO: Replace this with usage of code in Softmax operator

  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (auto it = values.begin(); it != values.end(); ++it) {
    if (static_cast<float>(*it) > v_max)
      v_max = static_cast<float>(*it);
  }
  float this_sum = 0.f;
  for (auto it = values.begin(); it != values.end(); ++it) {
    *it = std::exp(static_cast<float>(*it) - v_max);
    this_sum += static_cast<float>(*it);
  }
  for (auto it = values.begin(); it != values.end(); ++it)
    *it = static_cast<float>(*it) / this_sum;
}

// this function skips zero values (since exp(0) is non zero)
template <typename T>
static inline void ComputeSoftmaxZero(gsl::span<T>& values) {
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (auto it = values.begin(); it != values.end(); ++it) {
    if (static_cast<float>(*it) > v_max)
      v_max = static_cast<float>(*it);
  }
  float exp_neg_v_max = std::exp(-v_max);
  float this_sum = 0.f;
  for (auto it = values.begin(); it != values.end(); ++it) {
    if (static_cast<float>(*it) > 0.0000001f || static_cast<float>(*it) < -0.0000001f) {
      *it = std::exp(static_cast<float>(*it) - v_max);
      this_sum += static_cast<float>(*it);
    } else {
      *it = *it * exp_neg_v_max;
    }
  }
  for (auto it = values.begin(); it != values.end(); ++it)
    *it = *it / this_sum;
}

template <typename T, typename IT>
static void write_scores(InlinedVector<IT>& scores, POST_EVAL_TRANSFORM post_transform,
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
      case POST_EVAL_TRANSFORM::SOFTMAX: {
        auto span = gsl::make_span(scores);
        ComputeSoftmax(span);
        for (auto it = scores.begin(); it != scores.end(); ++it, ++Z)
          *Z = static_cast<T>(*it);
        break;
      }
      case POST_EVAL_TRANSFORM::SOFTMAX_ZERO: {
        auto span = gsl::make_span(scores);
        ComputeSoftmaxZero(span);
        for (auto it = scores.begin(); it != scores.end(); ++it, ++Z)
          *Z = static_cast<T>(*it);
        break;
      }
      default:
      case POST_EVAL_TRANSFORM::NONE:
        for (auto it = scores.begin(); it != scores.end(); ++it, ++Z)
          *Z = static_cast<T>(*it);
        break;
    }
  } else if (scores.size() == 1) {  // binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      scores[0] = ComputeProbit(static_cast<float>(scores[0]));
      *Z = static_cast<T>(scores[0]);
    } else {
      switch (add_second_class) {
        case 0:  // 0=all positive weights, winning class is positive
          scores.push_back(scores[0]);
          scores[0] = 1 - scores[0];  // put opposite score in positive slot
          *Z = static_cast<T>(scores[0]);
          *(Z + 1) = static_cast<T>(scores[1]);
          break;
        case 1:  // 1 = all positive weights, winning class is negative
          scores.push_back(scores[0]);
          scores[0] = 1 - scores[0];  // put opposite score in positive slot
          *Z = static_cast<T>(scores[0]);
          *(Z + 1) = static_cast<T>(scores[1]);
          break;
        case 2:
        case 3:  // 2 = mixed weights, winning class is positive
          if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
            scores.resize(2);
            scores[1] = static_cast<T>(ComputeLogistic(static_cast<float>(scores[0])));
            scores[0] = static_cast<T>(ComputeLogistic(static_cast<float>(-scores[0])));
          } else {
            scores.push_back(scores[0]);
            scores[0] = -scores[0];
          }
          *Z = static_cast<T>(scores[0]);
          *(Z + 1) = static_cast<T>(scores[1]);
          break;
        default:
          *Z = static_cast<T>(scores[0]);
          break;
      }
    }
  }
}

template <typename T>
static void write_scores(InlinedVector<T>& scores, POST_EVAL_TRANSFORM post_transform, int64_t write_index, Tensor* Z,
                         int add_second_class) {
  T* out_p = Z->MutableData<T>() + write_index;
  size_t len;
  if (!IAllocator::CalcMemSizeForArray(scores.size(), sizeof(T), &len)) {
    ORT_THROW("length overflow");
  }
  write_scores(scores, post_transform, out_p, add_second_class);
}

// TODO: Update TreeEnsemble* ops to use this instead of write_scores if possible.
//       Attempted to parallelize the calculations if the number of scores to process was large, but no clear benefit
//       was seen from testing with the arbitrary values of 1000 scores per threads.
template <typename T>
void batched_update_scores_inplace(gsl::span<T> scores, int64_t num_batches_in, int64_t batch_size,
                                   POST_EVAL_TRANSFORM post_transform,
                                   int add_second_class, bool have_space_for_second_class,
                                   concurrency::ThreadPool* threadpool) {
  if (batch_size < 1)
    return;

  SafeInt<int32_t> num_batches(num_batches_in);
  SafeInt<int32_t> num_scores = num_batches * batch_size;
  SafeInt<int32_t> expected_num_scores = num_scores * (batch_size == 1 && add_second_class >= 0 ? 2 : 1);
  ORT_ENFORCE(scores.size() == static_cast<size_t>(expected_num_scores));

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
        bool use_mlas = true;
        // if there are less than 8 items in each batch it may be slower to use mlas.
        // currently MlasComputeSoftmax adds threads on 16K blocks of work.
        // for smaller batches it takes more threads to counter some of the overhead.
        switch (batch_size) {
          case 1:
            use_mlas = false;  // mlas is mildly slower
            break;
          case 2:
            use_mlas = num_scores >= 32 * 1024;
            break;
          case 3:
          case 4:
            use_mlas = num_scores >= 16 * 1024;
            break;
          default:
            // toss up if num_scores is low (<200), but the more scores to process the larger the win by mlas
            break;
        }

        if (use_mlas) {
          MlasComputeSoftmax(s, s, num_batches, onnxruntime::narrow<size_t>(batch_size), false, threadpool);
        } else {
          while (s < s_end) {
            gsl::span<float> scores_for_batch(s, s + batch_size);
            ComputeSoftmax(scores_for_batch);
            s += batch_size;
          }
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

        case 2:  // 2 = mixed weights, winning class is positive
        case 3:  // 3 = mixed weights, winning class is negative
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

      if (have_space_for_second_class) {
        // forward iteration as there's a gap between each score to write into
        float* cur_score = scores.data();
        for (int i = 0; i < num_batches; ++i) {
          update_scores(*cur_score, cur_score);
          cur_score += 2;
        }
      } else {
        // reverse iteration as the scores are packed together and each score needs to be expanded to two
        const float* cur_in = s_end;
        float* cur_out = scores.data() + scores.size();
        while (cur_in > s) {
          --cur_in;
          cur_out -= 2;
          update_scores(*cur_in, cur_out);
        }
      }
    }
  }
}
}  // namespace ml
}  // namespace onnxruntime
