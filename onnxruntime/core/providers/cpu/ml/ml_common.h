// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

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
  } else if (input == "LEAF") {
    return NODE_MODE::LEAF;
  } else if (input == "BRANCH_LT") {
    return NODE_MODE::BRANCH_LT;
  } else if (input == "BRANCH_GTE") {
    return NODE_MODE::BRANCH_GTE;
  } else if (input == "BRANCH_GT") {
    return NODE_MODE::BRANCH_GT;
  } else if (input == "BRANCH_EQ") {
    return NODE_MODE::BRANCH_EQ;
  } else {
    return NODE_MODE::BRANCH_NEQ;
  }
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
  } else if (input == "LOGISTIC") {
    return POST_EVAL_TRANSFORM::LOGISTIC;
  } else if (input == "SOFTMAX") {
    return POST_EVAL_TRANSFORM::SOFTMAX;
  } else if (input == "SOFTMAX_ZERO") {
    return POST_EVAL_TRANSFORM::SOFTMAX_ZERO;
  } else {
    return POST_EVAL_TRANSFORM::PROBIT;
  }
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
  } else if (input == "SUM") {
    return AGGREGATE_FUNCTION::SUM;
  } else if (input == "MIN") {
    return AGGREGATE_FUNCTION::MIN;
  } else {
    return AGGREGATE_FUNCTION::MAX;
  }
}

enum class CAST_TO {
  TO_FLOAT,
  TO_STRING,
  TO_INT64
};

static inline CAST_TO MakeCast(const std::string& input) {
  if (input == "TO_FLOAT") {
    return CAST_TO::TO_FLOAT;
  } else if (input == "TO_STRING") {
    return CAST_TO::TO_STRING;
  } else if (input == "TO_INT64") {
    return CAST_TO::TO_INT64;
  } else {
    ORT_THROW("Invalid CAST_TO value of ", input, " Expected TO_FLOAT, TO_STRING or TO_INT64");
  }
}

enum PACK_MAP {
  DENSE,
  SPARSE
};

static inline PACK_MAP MakePack(const std::string& input) {
  if (input == "DENSE") {
    return PACK_MAP::DENSE;
  } else if (input == "SPARSE") {
    return PACK_MAP::SPARSE;
  } else {
    ORT_THROW("Invalid PACK_MAP value of ", input, " Expected DENSE or SPARSE");
  }
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
  } else if (input == "POLY") {
    return KERNEL::POLY;
  } else if (input == "RBF") {
    return KERNEL::RBF;
  } else {
    return KERNEL::SIGMOID;
  }
}

enum NORMALIZE {
  NMAX,
  L1,
  L2
};

static inline NORMALIZE MakeNormalize(const std::string& input) {
  if (input == "MAX") {
    return NORMALIZE::NMAX;
  } else if (input == "L1") {
    return NORMALIZE::L1;
  } else if (input == "L2") {
    return NORMALIZE::L2;
  } else {
    ORT_THROW("Invalid normalize value of ", input);
  }
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

static inline void ComputeSoftmax(std::vector<float>& values) {
  std::vector<float> newscores;
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float this_sum = 0.f;
  for (float value : values) {
    float val2 = std::exp(value - v_max);
    this_sum += val2;
    newscores.push_back(val2);
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values.size()); k++) {
    values[k] = newscores[k] / this_sum;
  }
}

//this function skips zero values (since exp(0) is non zero)
static inline void ComputeSoftmaxZero(std::vector<float>& values) {
  std::vector<float> newscores;
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float exp_neg_v_max = std::exp(-v_max);
  float this_sum = 0.f;
  for (float value : values) {
    if (value > 0.0000001f || value < -0.0000001f) {
      float val2 = std::exp(value - v_max);
      this_sum += val2;
      newscores.push_back(val2);
    } else {
      newscores.push_back(value * exp_neg_v_max);
    }
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values.size()); k++) {
    values[k] = newscores[k] / this_sum;
  }
}

template <typename T>
void write_scores(std::vector<T>& scores, POST_EVAL_TRANSFORM post_transform, int64_t write_index, Tensor* Z,
                  int add_second_class) {
  if (post_transform == POST_EVAL_TRANSFORM::PROBIT && scores.size() == 1) {
    scores[0] = ComputeProbit(scores[0]);
  } else if (scores.size() >= 2) {  //multiclass
    if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
      for (float& score : scores) {
        score = ComputeLogistic(score);
      }
    } else if (post_transform == POST_EVAL_TRANSFORM::SOFTMAX) {
      ComputeSoftmax(scores);
    } else if (post_transform == POST_EVAL_TRANSFORM::SOFTMAX_ZERO) {
      ComputeSoftmaxZero(scores);
    }
  } else {                                              //binary case
    if (add_second_class == 0 && scores.size() == 1) {  //0=all positive weights, winning class is positive
      scores.push_back(scores[0]);
      scores[0] = 1.f - scores[0];                             //put opposite score in positive slot
    } else if (add_second_class == 1 && scores.size() == 1) {  //1 = all positive weights, winning class is negative
      scores.push_back(scores[0]);
      scores[0] = 1.f - scores[0];                             //put opposite score in positive slot
    } else if (add_second_class == 2 && scores.size() == 1) {  //2 = mixed weights, winning class is positive
      if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
        scores.push_back(ComputeLogistic(scores[0]));
        scores[0] = ComputeLogistic(-scores[0]);
      } else {
        scores.push_back(scores[0]);
        scores[0] = -scores[0];
      }
    } else if (add_second_class == 3 && scores.size() == 1) {  //3 = mixed weights, winning class is negative
      if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
        scores.push_back(ComputeLogistic(scores[0]));
        scores[0] = ComputeLogistic(-scores[0]);
      } else {
        scores.push_back(-scores[0]);
      }
    }
  }
  T* out_p = Z->template MutableData<T>() + write_index;
  size_t len;
  if (!IAllocator::CalcMemSizeForArray(scores.size(), sizeof(T), &len)) {
    ORT_THROW("length overflow");
  }
  memcpy(out_p, scores.data(), len);
}

}  // namespace ml
}  // namespace onnxruntime
