// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include <algorithm>
#include <numeric>

namespace onnxruntime {
namespace test {

#if USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#else
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#endif

class NvtxOpTester {
  std::vector<int64_t> in_shape_;
  std::vector<float> in_data_;
  std::vector<int64_t> out_shape_;
  std::vector<float> out_data_;

  std::string operation_;
  int64_t nelements_;

 public:
  NvtxOpTester(std::string operation, std::vector<int64_t> in_shape) : 
	  in_shape_(in_shape), operation_(operation) {
		  
    // element count
    nelements_ = std::accumulate(
        in_shape_.cbegin(),
        in_shape_.cend(),
        1LL, std::multiplies<int64_t>());

    FillInputs();
    ComputeInternal();
  }

  void FillInputs() {
    srand(10);

    in_data_.resize(nelements_);
    for (int64_t i = 0; i < nelements_; i++)
      in_data_[i] = -5.0f + 10.0f * ((float)rand() / float(RAND_MAX));
  }

  void ComputeInternal() {
    out_data_.resize(nelements_);
    out_shape_ = in_shape_;

    for (int64_t i = 0; i < nelements_; i++)
      out_data_[i] = in_data_[i];
  }

  void RunComparison() {
    OpTester tester(operation_.c_str(), 1, onnxruntime::kMSDomain);

    tester.AddInput<float>("input", in_shape_, in_data_);
    tester.AddOutput<float>("output", out_shape_, out_data_);

    std::vector<std::unique_ptr<IExecutionProvider>> ep;
    #ifdef USE_CUDA
      ep.push_back(DefaultCudaExecutionProvider());
    #elif USE_ROCM
      ep.push_back(DefaultRocmExecutionProvider());
    #endif
    
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep);
  }
};

TEST(NvtxOpTest, NvtxPushTest) {
  NvtxOpTester test("NvtxPush", {2, 2, 2});
  test.RunComparison();
}

TEST(NvtxOpTest, NvtxPopTest) {
  NvtxOpTester test("NvtxPop", {2, 2, 2});
  test.RunComparison();
}

}  // namespace test
}  // namespace onnxruntime
