// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(const vector<T>& input,
                    const vector<int64_t>& dims,
                    const vector<float>& output,
                    const std::string& norm,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expect_error_message = "") {
  OpTester test("Normalizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("norm", norm);

  test.AddInput("X", dims, input);
  test.AddOutput("Y", dims, output);

  // output 'norm' so if a test fails we know which one
  std::cout << "norm=" << norm << "\n";
  test.Run(expect_result, expect_error_message);
}

template <typename T>
static void RunTests(const vector<T>& input,
                     const vector<int64_t>& dims,
                     const vector<float>& max_output,
                     const vector<float>& l1_output,
                     const vector<float>& l2_output) {
  RunTest(input, dims, max_output, "MAX");
  RunTest(input, dims, l1_output, "L1");
  RunTest(input, dims, l2_output, "L2");
}

/*
Test values from this script, which is based on functions in
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/data.py

scikit-learn is a Python module for machine learning built on top of SciPy and
distributed under the 3-Clause BSD license. See https://github.com/scikit-learn/scikit-learn.
This material is licensed under the BSD License (see https://github.com/scikit-learn/scikit-learn/blob/master/COPYING);

import numpy as np

def _handle_zeros_in_scale(scale):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        scale[scale == 0.0] = 1.0
        return scale


def normalize(X, norm='l2', axis=1):
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if norm == 'l1':
        norms = np.abs(X).sum(axis=axis).astype(np.float32)
    elif norm == 'l2':
        norms = np.sqrt((X * X).sum(axis=axis)).astype(np.float32)
    elif norm == 'max':
        norms = np.max(X, axis=axis).astype(np.float32)

    norms = _handle_zeros_in_scale(norms)

    x_float = X.astype(np.float32)

    # special case single dimension input
    if np.isscalar(norms) == 1:
        x_float /= norms
    else:
        x_float /= norms[:, np.newaxis]

    return x_float


def RunNormalize(X, axis=1):
    print("Normalizing")
    print(X)
    print("\nmax")
    y = normalize(X, 'max', axis)
    print(y)
    print("\nL1")
    y = normalize(X, 'l1', axis)
    print(y)
    print("\nL2")
    y = normalize(X, 'l2', axis)
    print(y)
    print("\n\n")


x = np.array([-1, 0, 1]).astype(np.float32)
RunNormalize(x, 0)

np.random.seed(123)
x = np.random.randn(2, 3).astype(np.double)
RunNormalize(x)

x = (100 * np.random.randn(2, 3, 4)).astype(np.int)
RunNormalize(x, 1)

*/
TEST(Normalizer, SingleDimensionFloat) {
  std::vector<int64_t> dims = {3};
  std::vector<float> input = {-1.f, 0.f, 1.f};

  std::vector<float> max_output{-1.0f, 0.f, 1.0f};
  std::vector<float> l1_output{-0.5f, 0.f, 0.5f};
  std::vector<float> l2_output{-0.70710677f, 0.f, 0.70710677f};

  RunTests(input, dims, max_output, l1_output, l2_output);
}

TEST(Normalizer, TwoDimensionFloat) {
  std::vector<int64_t> dims = {2, 3};
  std::vector<float> input = {-1.0856306f, 0.99734545f, 0.2829785f,
                              -1.50629471f, -0.57860025f, 1.65143654f};

  std::vector<float> max_output{-1.0885202f, 1.f, 0.2837317f,
                                -0.91211176f, -0.35036176f, 1.f};

  std::vector<float> l1_output{-0.45885524f, 0.42154038f, 0.11960436f,
                               -0.40314806f, -0.15485784f, 0.44199413f};

  std::vector<float> l2_output{-0.7232126f, 0.6643998f, 0.18851127f,
                               -0.65239084f, -0.25059736f, 0.7152532f};

  RunTests(input, dims, max_output, l1_output, l2_output);
}

TEST(Normalizer, TwoDimensionDouble) {
  std::vector<int64_t> dims = {2, 3};
  std::vector<double> input = {-1.0856306, 0.99734545, 0.2829785,
                               -1.50629471, -0.57860025f, 1.65143654};

  std::vector<float> max_output{-1.0885202f, 1.f, 0.2837317f,
                                -0.91211176f, -0.35036176f, 1.f};

  std::vector<float> l1_output{-0.45885524f, 0.42154038f, 0.11960436f,
                               -0.40314806f, -0.15485784f, 0.44199413f};

  std::vector<float> l2_output{-0.7232126f, 0.6643998f, 0.18851127f,
                               -0.65239084f, -0.25059736f, 0.7152532f};

  RunTests(input, dims, max_output, l1_output, l2_output);
}

#if defined(_M_AMD64) || defined(__x86_64__)
TEST(Normalizer, TwoDimensionInt) {
  std::vector<int64_t> dims = {3, 2};
  std::vector<int32_t> input = {-242, -42,
                                126, -86,
                                -67, -9};

  std::vector<float> max_output{5.7619047f, 1.f,
                                1.f, -0.6825397f,
                                7.4444447f, 1.f};

  std::vector<float> l1_output{-0.85211265f, -0.14788732f,
                               0.5943396f, -0.4056604f,
                               -0.8815789f, -0.11842106f};

  std::vector<float> l2_output{-0.98527145f, -0.17099753f,
                               0.82594985f, -0.56374353f,
                               -0.9910982f, -0.13313259f};

  RunTests(input, dims, max_output, l1_output, l2_output);
}
#endif

TEST(Normalizer, InvalidNorm) {
  std::vector<int64_t> dims = {3};
  std::vector<float> input = {-1.f, 0.f, 1.f};
  std::vector<float> output{-1.0f, 0.f, 1.0f};

  RunTest(input, dims, output, "InvalidNormValue", OpTester::ExpectResult::kExpectFailure);
}

}  // namespace test
}  // namespace onnxruntime
