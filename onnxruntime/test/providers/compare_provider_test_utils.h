// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/constants.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

class CompareOpTester : public OpTester {
 public:
  CompareOpTester(const char* op,
                  int opset_version = 9,
                  const char* domain = onnxruntime::kOnnxDomain)
      : OpTester(op, opset_version, domain) {}

  void CompareWithCPU(const std::string& target_provider_type,
                      double per_sample_tolerance = 1e-4,
                      double relative_per_sample_tolerance = 1e-4,
                      const bool need_cpu_cast = false);
};

}  // namespace test
}  // namespace onnxruntime
