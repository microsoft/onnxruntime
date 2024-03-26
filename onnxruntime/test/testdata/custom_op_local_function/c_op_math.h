#pragma once

#include "c_op_common_parameters.h"
#include <cfloat>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace onnx_c_ops {

#define is_a_ge_zero_and_a_lt_b(a, b) (static_cast<uint64_t>(a) < static_cast<uint64_t>(b))

#define InlinedVector std::vector
#define InlinedHashSet std::unordered_set

#if defined(_WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#elif defined(__MACOSX__) || defined(__APPLE__)

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See
// https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
  union {
    uint64_t u;
    double f;
  } ieee754;
  ieee754.f = x;
  return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) + ((unsigned)ieee754.u != 0) > 0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#if !defined(__APPLE__)
#ifndef _SSIZE_T_DEFINED
typedef int64_t ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif

inline float ErfInv(float x) {
  float sgn = x < 0 ? -1.0f : 1.0f;
  x = (1 - x) * (1 + x);
  float log = std::log(x);
  float v = 2 / (3.14159f * 0.147f) + 0.5f * log;
  float v2 = 1 / (0.147f) * log;
  float v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}

inline double ErfInv(double x) {
  double sgn = x < 0 ? -1.0 : 1.0;
  x = (1 - x) * (1 + x);
  double log = std::log(x);
  double v = 2. / (3.14159f * 0.147f) + 0.5f * log;
  double v2 = 1. / (0.147f) * log;
  double v3 = std::sqrt(v * v - v2) - v;
  return sgn * std::sqrt(v3);
}

inline float ComputeLogistic(float val) {
  float v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}

inline double ComputeLogistic(double val) {
  double v = 1. / (1. + std::exp(-std::abs(val)));
  return (val < 0) ? (1. - v) : v;
}

#define ml_sqrt2 1.41421356f

template <class NTYPE> inline NTYPE ComputeProbit(NTYPE val) {
  return ml_sqrt2 * ErfInv(val * 2 - 1);
}

template <class NTYPE> inline NTYPE sigmoid_probability(NTYPE score, NTYPE proba, NTYPE probb) {
  // ref:
  // https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
  NTYPE val = score * proba + probb;
  return 1 - ComputeLogistic(val);
}

template <typename NTYPE> inline void ComputeSoftmax(NTYPE *begin, NTYPE *end) {
  NTYPE v_max = -FLT_MAX;
  NTYPE *it;
  for (it = begin; it != end; ++it) {
    if (*it > v_max)
      v_max = *it;
  }
  NTYPE this_sum = 0;
  for (it = begin; it != end; ++it) {
    *it = std::exp(*it - v_max);
    this_sum += *it;
  }
  for (it = begin; it != end; ++it)
    *it /= this_sum;
}

template <typename NTYPE> inline void ComputeSoftmax(std::vector<NTYPE> &values) {
  ComputeSoftmax(values.data(), values.data() + values.size());
}

template <typename NTYPE> inline void ComputeSoftmaxZero(NTYPE *begin, NTYPE *end) {
  NTYPE v_max = -std::numeric_limits<NTYPE>::max();
  NTYPE *it;
  for (it = begin; it != end; ++it) {
    if (*it > v_max)
      v_max = *it;
  }
  NTYPE exp_neg_v_max = std::exp(-v_max);
  NTYPE this_sum = (NTYPE)0;
  for (it = begin; it != end; ++it) {
    if (*it > 0.0000001f || *it < -0.0000001f) {
      *it = std::exp(*it - v_max);
      this_sum += *it;
    } else {
      *it *= exp_neg_v_max;
    }
  }
  for (it = begin; it != end; ++it)
    *it /= this_sum;
}

template <typename NTYPE> inline void ComputeSoftmaxZero(std::vector<NTYPE> &values) {
  ComputeSoftmaxZero(values.data(), values.data() + values.size());
}

template <typename NTYPE, typename T>
std::size_t write_scores(std::vector<NTYPE> &scores, POST_EVAL_TRANSFORM post_transform, T *Z,
                         int add_second_class) {
  if ((scores.size() == 1) && add_second_class) {
    scores.push_back(scores[0]);
    scores[1] = 0.f;
    return write_scores(1, scores.data(), post_transform, Z, add_second_class);
  }
  return write_scores(scores.size(), scores.data(), post_transform, Z, add_second_class);
}

template <typename NTYPE, typename T>
std::size_t write_scores(std::size_t n_classes, NTYPE *scores,
                         POST_EVAL_TRANSFORM post_transform, T *Z, int add_second_class) {
  if (n_classes >= 2) {
    NTYPE *end = scores + n_classes;
    switch (post_transform) {
    case POST_EVAL_TRANSFORM::PROBIT:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = ComputeProbit((T)*it);
      break;
    case POST_EVAL_TRANSFORM::LOGISTIC:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = ComputeLogistic((T)*it);
      break;
    case POST_EVAL_TRANSFORM::SOFTMAX:
      ComputeSoftmax(scores, end);
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
      ComputeSoftmaxZero(scores, end);
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    default:
    case POST_EVAL_TRANSFORM::NONE:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    }
  } else if (n_classes == 1) { // binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      scores[0] = ComputeProbit((T)scores[0]);
      *Z = scores[0];
    } else {
      switch (add_second_class) {
      case 0: // 0=all positive weights, winning class is positive
        scores[1] = (T)scores[0];
        scores[0] = 1.f - (T)scores[0]; // put opposite score in positive slot
        *Z = (T)scores[0];
        *(Z + 1) = (T)scores[1];
        ++n_classes;
        break;
      case 1: // 1 = all positive weights, winning class is negative
        scores[1] = (T)scores[0];
        scores[0] = 1.f - (T)scores[0]; // put opposite score in positive slot
        *Z = (T)scores[0];
        *(Z + 1) = (T)scores[1];
        ++n_classes;
        break;
      case 2:
      case 3: // 2 = mixed weights, winning class is positive
        if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
          scores[1] = ComputeLogistic((T)scores[0]); // ml_logit(scores[k]);
          scores[0] = ComputeLogistic((T)(-scores[0]));
        } else {
          scores[1] = (T)scores[0];
          scores[0] = (T)(-scores[0]);
        }
        *Z = scores[0];
        *(Z + 1) = scores[1];
        ++n_classes;
        break;
      default:
        *Z = scores[0];
        break;
      }
    }
  }
  return n_classes;
}

template <typename NTYPE, typename T>
std::size_t write_scores2(NTYPE *scores, POST_EVAL_TRANSFORM post_transform, T *Z,
                          int /* add_second_class */) {
  switch (post_transform) {
  case POST_EVAL_TRANSFORM::PROBIT:
    Z[0] = ComputeProbit(scores[0]);
    Z[1] = ComputeProbit(scores[1]);
    break;
  case POST_EVAL_TRANSFORM::LOGISTIC:
    Z[0] = ComputeLogistic(scores[0]);
    Z[1] = ComputeLogistic(scores[1]);
    break;
  case POST_EVAL_TRANSFORM::SOFTMAX:
    ComputeSoftmax(scores, scores + 2);
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
    ComputeSoftmaxZero(scores, scores + 2);
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  default:
  case POST_EVAL_TRANSFORM::NONE:
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  }
  return 2;
}

template <typename T, T b> constexpr T roundUpPow2(T a) { return (a + (b - 1)) & (~(b - 1)); }

} // namespace onnx_c_ops
