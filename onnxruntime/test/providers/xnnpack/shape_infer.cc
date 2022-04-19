// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/xnnpack/schema/xnnpack_onnx_defs.h>
#include <gtest/gtest.h>

namespace {
struct ComputeOutputSizeValidTestArg {
  ptrdiff_t input_size;
  uint32_t stride;
  ptrdiff_t filter_size;
  uint32_t dilation_rate;
  ptrdiff_t output_size;
};
class ComputeOutputSizeValidTest : public testing::TestWithParam<ComputeOutputSizeValidTestArg> {};

struct ComputeOutputSizeSameTestArg {
  ptrdiff_t input_size;
  uint32_t stride;
  ptrdiff_t output_size;
};

class ComputeOutputSizeSameTest : public testing::TestWithParam<ComputeOutputSizeSameTestArg> {};

}  // namespace

namespace onnxruntime {
namespace test {
TEST_P(ComputeOutputSizeValidTest, test1) {
  const ComputeOutputSizeValidTestArg& arg = this->GetParam();
  ptrdiff_t output_size;
  xnnpack::OnnxStatus st =
      xnnpack::ComputeOutputSizeValid(arg.input_size, arg.stride, arg.filter_size, arg.dilation_rate, &output_size);
  if (arg.output_size == -1) {
    ASSERT_FALSE(st.IsOK());
  } else {
    ASSERT_EQ(arg.output_size, output_size);
  }
}
static ComputeOutputSizeValidTestArg test_values[] = {{0, 0, 0, 0, -1},
                                                      {0, 0, 1, 0, -1},
                                                      {0, 255, 1, 0, -1},
                                                      {-1, 255, 1, 0, -1},
                                                      {5, 1, 2, 1, 4},
#if defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64)
                                                      {-35459249995777, 255, 9223336852482686975, 0, -1},
                                                      {0, 255, 3138168558370481227, 2132846677, -1},
                                                      {0, 255, 3964167988, 2329959947, -1},
                                                      {9223372035353181953, 4286578687, 809508867, 0, -1}
#endif
};
INSTANTIATE_TEST_SUITE_P(ComputeOutputSizeValidTest1, ComputeOutputSizeValidTest, testing::ValuesIn(test_values));

TEST_P(ComputeOutputSizeSameTest, test1) {
  const ComputeOutputSizeSameTestArg& arg = this->GetParam();
  ptrdiff_t output_size;
  xnnpack::OnnxStatus st = xnnpack::ComputeOutputSizeSame(arg.input_size, arg.stride, &output_size);
  if (arg.output_size == -1) {
    ASSERT_FALSE(st.IsOK());
  } else {
    ASSERT_EQ(arg.output_size, output_size);
  }
}
static ComputeOutputSizeSameTestArg test_values2[] = {
    {0, 0, -1}, {0, 0, -1}, {0, 255, -1}, {-1, 255, -1}, {std::numeric_limits<ptrdiff_t>::max(), 255, -1}, {5, 1, 5},
#if defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64)

#endif
};
INSTANTIATE_TEST_SUITE_P(ComputeOutputSizeSameTest1, ComputeOutputSizeSameTest, testing::ValuesIn(test_values2));
}  // namespace test
}  // namespace onnxruntime