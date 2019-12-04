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

TEST(Normalizer, ThreeDimensionInt32) {
  std::vector<int64_t> dims = {2, 3, 4};
  std::vector<int32_t> input = {-242, -42, 126, -86,
                                -67, -9, 149, -63,
                                -44, -43, 220, 218,

                                100, 38, 73, 149,
                                -93, 117, -125, -63,
                                90, -142, -14, -86};

  std::vector<float> max_output{5.5f, 4.6666665f, 0.57272726f, -0.39449543f,
                                1.5227273f, 1.f, 0.67727274f, -0.28899083f,
                                1.f, 4.7777777f, 1.f, 1.f,

                                1.f, 0.32478634f, 1.f, 1.f,
                                -0.93f, 1.f, -1.7123288f, -0.42281878f,
                                0.9f, -1.2136753f, -0.19178082f, -0.5771812f};

  std::vector<float> l1_output{-0.6855524f, -0.44680852f, 0.25454545f, -0.23433243f,
                               -0.1898017f, -0.09574468f, 0.3010101f, -0.17166212f,
                               -0.12464589f, -0.4574468f, 0.44444445f, 0.59400547f,

                               0.3533569f, 0.12794612f, 0.3443396f, 0.5f,
                               -0.3286219f, 0.3939394f, -0.5896226f, -0.21140939f,
                               0.3180212f, -0.4781145f, -0.06603774f, -0.2885906f};

  std::vector<float> l2_output{-0.94928247f, -0.6910363f, 0.4284698f, -0.3543899f,
                               -0.26281786f, -0.1480792f, 0.5066825f, -0.25961122f,
                               -0.17259681f, -0.7074895f, 0.74812186f, 0.89833724f,

                               0.6114293f, 0.2022622f, 0.5019584f, 0.8132732f,
                               -0.56862926f, 0.62275463f, -0.85951775f, -0.34386718f,
                               0.55028635f, -0.7558219f, -0.09626599f, -0.469406f};

  RunTests(input, dims, max_output, l1_output, l2_output);
}

TEST(Normalizer, InvalidNorm) {
  std::vector<int64_t> dims = {3};
  std::vector<float> input = {-1.f, 0.f, 1.f};
  std::vector<float> output{-1.0f, 0.f, 1.0f};

  RunTest(input, dims, output, "InvalidNormValue", OpTester::ExpectResult::kExpectFailure);
}

}  // namespace test
}  // namespace onnxruntime
