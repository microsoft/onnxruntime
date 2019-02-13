#pragma once
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

class BrainSliceTestor : public OpTester {
 public:
  BrainSliceTestor(const char* op, int opset_version = 7, const char* domain = onnxruntime::kOnnxDomain) : OpTester(op, opset_version, domain) {}
  void CompareWithCPU(double rtol, double atol);
};

}  // namespace test
}  // namespace onnxruntime