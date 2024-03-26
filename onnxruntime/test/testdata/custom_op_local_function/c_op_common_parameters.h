#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_c_ops {

enum class POST_EVAL_TRANSFORM {
  NONE = 0,
  LOGISTIC = 1,
  SOFTMAX = 2,
  SOFTMAX_ZERO = 3,
  PROBIT = 4
};

POST_EVAL_TRANSFORM to_POST_EVAL_TRANSFORM(const std::string &value);

enum NODE_MODE : uint8_t {
  LEAF = 1,
  BRANCH_LEQ = 2,
  BRANCH_LT = 4,
  BRANCH_GTE = 6,
  BRANCH_GT = 8,
  BRANCH_EQ = 10,
  BRANCH_NEQ = 12
};

NODE_MODE to_NODE_MODE(const std::string &value);

const char *to_str(NODE_MODE mode);

enum class AGGREGATE_FUNCTION { AVERAGE, SUM, MIN, MAX };

AGGREGATE_FUNCTION to_AGGREGATE_FUNCTION(const std::string &input);

enum class SVM_TYPE { SVM_LINEAR = 1, SVM_SVC = 2 };

SVM_TYPE to_SVM_TYPE(const std::string &value);

enum KERNEL { LINEAR, POLY, RBF, SIGMOID };

KERNEL to_KERNEL(const std::string &value);

enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

StorageOrder to_StorageOrder(const std::string &value);

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

AutoPadType to_AutoPadType(const std::string &value);

} // namespace onnx_c_ops
