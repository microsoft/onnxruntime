#include "comparison.h"

#include <functional>
#include <iostream>
#include <limits>

namespace {

void ExpectFailure(const std::function<void()>& action) {
  bool failed = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  if (!failed) {
    throw std::runtime_error("Expected comparison to fail");
  }
}

}  // namespace

int main() {
  using namespace onnxruntime::test::ep_op_conformance;
  try {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values{1, -2, infinity, -infinity, nan};
    CompareFloats(values, values.data(), values.size(), {});
    CompareFloats({}, nullptr, 0, {});
    const std::vector<float> nearby{1.00005f};
    CompareFloats({1}, nearby.data(), 1, {});
    ExpectFailure([&] { CompareFloats({1}, nearby.data(), 1, {0, 0, true}); });
    ExpectFailure([&] { CompareFloats({-infinity}, &infinity, 1, {}); });
    ExpectFailure([&] { CompareFloats({nan}, &nan, 1, {0, 0, false}); });
    ExpectFailure([&] { CompareFloats({1}, &nan, 1, {}); });
    ExpectFailure([&] { CompareFloats({1}, values.data(), 2, {}); });
    ExpectFailure([&] { CompareFloats({}, nullptr, 0, {-1, 0, true}); });
    ExpectFailure([&] { CompareFloats({}, nullptr, 0, {nan, 0, true}); });
    std::cout << "10 comparison checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}