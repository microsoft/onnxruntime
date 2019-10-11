// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/util/softmax.h"

namespace onnxruntime {
namespace ml {  // name space for onnx.ml operators

enum class OUTPUT_MODE { TOPCLASS,
                         TOPCLASS_ANDSCORE,
                         ALL_SCORES };

enum class NODE_MODE { BRANCH_LEQ,
                       BRANCH_LT,
                       BRANCH_GTE,
                       BRANCH_GT,
                       BRANCH_EQ,
                       BRANCH_NEQ,
                       LEAF };

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

enum class POST_EVAL_TRANSFORM { NONE,
                                 LOGISTIC,
                                 SOFTMAX,
                                 SOFTMAX_ZERO,
                                 PROBIT };

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

enum class AGGREGATE_FUNCTION { AVERAGE,
                                SUM,
                                MIN,
                                MAX };

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

enum class CAST_TO { TO_FLOAT,
                     TO_STRING,
                     TO_INT64 };

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

enum PACK_MAP { DENSE,
                SPARSE };

static inline PACK_MAP MakePack(const std::string& input) {
  if (input == "DENSE") {
    return PACK_MAP::DENSE;
  }
  if (input == "SPARSE") {
    return PACK_MAP::SPARSE;
  }
  ORT_THROW("Invalid PACK_MAP value of ", input, " Expected DENSE or SPARSE");
}

enum KERNEL { LINEAR,
              POLY,
              RBF,
              SIGMOID };

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

enum NORMALIZE { NMAX,
                 L1,
                 L2 };

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

enum class SVM_TYPE { SVM_LINEAR,
                      SVM_SVC };

template <typename T>
T ErfInv(T x) {
  T sgn = x < 0 ? -1.0f : 1.0f;
  x = (1 - x) * (1 + x);
  T log = std::log(x);
  T v = 2 / (static_cast<T>(M_PI) * 0.147f) + 0.5f * log;
  T v2 = 1 / (0.147f) * log;
  T v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}

// https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf
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
    if (max_error < eps)
      break;

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

// y = \frac{1}{1+e^{-x}} , x \in R
template <typename T>
inline T ComputeLogistic(T val) {
  T v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}

// It assumes val is in [0,1]
template <typename T>
inline T ComputeProbit(T val) {
  return static_cast<T>(M_SQRT2) * ErfInv(2 * val - 1);
}

static inline float sigmoid_probability(float score, float proba, float probb) {
  float val = score * proba + probb;
  //ref: https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
  return 1 - ComputeLogistic(val);
}

/**
 * if scores.size() == 1 and post_transform != POST_EVAL_TRANSFORM::PROBIT and
 * add_second_class is in [0, 3], output 2 value else output scores.size()
 * values
 */
template <typename T>
void write_scores(std::vector<T>& scores, POST_EVAL_TRANSFORM post_transform, int64_t write_index, Tensor* Z,
                  int add_second_class) {
  if (scores.empty())
    return;
  T* out_p = Z->template MutableData<T>() + write_index;
  if (scores.size() == 1) {
    write_binary_scores(scores[0], post_transform, add_second_class, out_p);
    if (post_transform != POST_EVAL_TRANSFORM::PROBIT && add_second_class >= 0 && add_second_class <= 3) {
      scores.push_back(0);
    }
  } else
    write_scores(scores.data(), scores.size(), post_transform, out_p);
}

/**
 * if post_transform != POST_EVAL_TRANSFORM::PROBIT and add_second_class is in
 * [0, 3], output 2 value else output 1 value
 * if post_transform == POST_EVAL_TRANSFORM::PROBIT, it assumes sc is in [0,1]
 */
template <typename T>
void write_binary_scores(T sc, POST_EVAL_TRANSFORM post_transform, int add_second_class, T* out_p) {
  if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
    *out_p = ComputeProbit(sc);
    return;
  }
  switch (add_second_class) {
    case 0:  // 0=all positive weights, winning class is positive
    case 1:
      *out_p++ = 1.f - sc;  // put opposite score in positive slot
      *out_p = (sc);
      break;
    case 2:  // 2 = mixed weights, winning class is positive
      if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
        *out_p++ = ComputeLogistic(-sc);
        *out_p = (ComputeLogistic(sc));  // ml_logit(scores[k]);
      } else {
        *out_p++ = -sc;
        *out_p = (sc);
      }
      break;
    case 3:  // 3 = mixed weights, winning class is negative
      if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
        *out_p++ = ComputeLogistic(-sc);
        *out_p = (ComputeLogistic(sc));  // ml_logit(scores[k]);
      } else {
        *out_p++ = sc;
        *out_p = (-sc);
      }
      break;
    default:
      *out_p = sc;
  }
}

template <typename T>
void write_scores(T* scores, size_t scores_len, POST_EVAL_TRANSFORM post_transform, T* out_p) {
  switch (post_transform) {
    case POST_EVAL_TRANSFORM::PROBIT:
      for (size_t i = 0; i != scores_len; ++i) out_p[i] = ComputeProbit(scores[i]);
      break;
    case POST_EVAL_TRANSFORM::LOGISTIC:
      for (size_t i = 0; i != scores_len; ++i) out_p[i] = ComputeLogistic(scores[i]);
      break;
    case POST_EVAL_TRANSFORM::SOFTMAX: {
      ComputeSoftmax(scores, scores_len, out_p);
    } break;
    case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
      ComputeSoftmaxZero(scores, scores_len, out_p);
      break;
    default:
    case POST_EVAL_TRANSFORM::NONE: {
      size_t len;
      if (!IAllocator::CalcMemSizeForArray(scores_len, sizeof(T), &len)) {
        ORT_THROW("length overflow");
      }
      memcpy(out_p, scores, len);
    } break;
  }
}

}  // namespace ml
}  // namespace onnxruntime
