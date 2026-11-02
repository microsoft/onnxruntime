#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnxruntime::test::ep_op_conformance {

struct Comparison {
  double absolute{1e-5};
  double relative{1e-4};
  bool nan_equal{true};
};

inline void ValidateComparison(const Comparison& policy) {
  if (!std::isfinite(policy.absolute) || !std::isfinite(policy.relative) ||
      policy.absolute < 0 || policy.relative < 0) {
    throw std::runtime_error("Comparison tolerances must be finite and nonnegative");
  }
  if (policy.absolute > std::numeric_limits<float>::max() ||
      policy.relative > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Comparison tolerances must fit float32");
  }
}

inline void CompareFloats(const std::vector<float>& expected, const float* actual,
                          size_t actual_count, const Comparison& policy) {
  ValidateComparison(policy);
  if (expected.size() != actual_count) {
    throw std::runtime_error("Output element count mismatch");
  }
  for (size_t index = 0; index < expected.size(); ++index) {
    const double reference = expected[index];
    const double observed = actual[index];
    bool matches;
    if (std::isnan(reference) || std::isnan(observed)) {
      matches = policy.nan_equal && std::isnan(reference) && std::isnan(observed);
    } else if (std::isinf(reference) || std::isinf(observed)) {
      matches = reference == observed;
    } else {
      const float relative_tolerance = static_cast<float>(policy.relative) * std::abs(expected[index]);
      const float tolerance = static_cast<float>(policy.absolute) + relative_tolerance;
      matches = std::abs(reference - observed) <= static_cast<double>(tolerance);
    }
    if (!matches) {
      throw std::runtime_error("Output mismatch at element " + std::to_string(index));
    }
  }
}

}  // namespace onnxruntime::test::ep_op_conformance
