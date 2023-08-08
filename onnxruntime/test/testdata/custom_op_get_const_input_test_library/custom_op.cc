// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "custom_op.h"
#include "common.h"

void simple_assert(const bool cond, const std::string& text) {
  if (!cond) {
#ifndef ORT_NO_EXCEPTIONS
    throw std::runtime_error(text);
#else
    std::cerr << text << std::endl;
    std::terminate();
#endif
  }
}

TestCustomKernel::TestCustomKernel(const OrtKernelInfo* info) {
  Ort::ConstKernelInfo kinfo(info);
  int is_constant = 0;

  // Get weights (constant inputs) from kernel info
  for (size_t i = 0; i < kinfo.GetInputCount(); i++) {
    Ort::ConstValue const_input = kinfo.GetTensorConstantInput(i, &is_constant);
    if (is_constant) {
      const float* value = const_input.GetTensorData<float>();
      simple_assert(value[0] == 1.0, "wrong value");
      simple_assert(value[1] == 2.0, "wrong value");
      simple_assert(value[2] == 3.0, "wrong value");
      simple_assert(value[3] == 4.0, "wrong value");
    }
  }
  simple_assert(is_constant == 1, "should be constant input");
}

void TestCustomKernel::Compute(OrtKernelContext* context) {
  ORT_UNUSED_PARAMETER(context);
}

//
// TestCustomOp
//

TestCustomOp::TestCustomOp() {}

void* TestCustomOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  ORT_UNUSED_PARAMETER(api);
#ifdef _WIN32
#pragma warning(disable : 26409)
#endif
  return new TestCustomKernel(info);
}
