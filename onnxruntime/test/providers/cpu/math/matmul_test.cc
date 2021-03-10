// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct MatMulTestData {
  std::string name;
  std::vector<int64_t> input0_dims;
  std::vector<int64_t> input1_dims;
  std::vector<int64_t> expected_dims;
  std::vector<T> expected_vals;
};

template <typename T>
std::vector<MatMulTestData<T>> GenerateTestCases() {
  std::vector<MatMulTestData<T>> test_cases;

  test_cases.push_back(
      {"test padding and broadcast",
       {3, 1, 1, 2},
       {2, 2, 2},
       {3, 2, 1, 2},
       {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}});

  test_cases.push_back(
      {"test padding and broadcast",
       {2, 3, 2},
       {3, 2, 2, 1},
       {3, 2, 3, 1},
       {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}});

  test_cases.push_back(
      {"test left 1D",
       {2},
       {3, 2, 1},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test right 1D",
       {3, 1, 2},
       {2},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test scalar output",
       {3},
       {3},
       {},
       {5}});

  test_cases.push_back(
      {"test 2D",
       {3, 4},
       {4, 3},
       {3, 3},
       {42, 48, 54, 114, 136, 158, 186, 224, 262}});

  test_cases.push_back(
      {"test 2D special",
       {2, 2, 3},
       {3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 2",
       {2, 2, 3},
       {1, 3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 3",
       {2, 6},
       {1, 1, 6, 1},
       {1, 1, 2, 1},
       {55, 145}});

  test_cases.push_back(
      {"test 2D empty input",
       {3, 4},
       {4, 0},
       {3, 0},
       {}});

  return test_cases;
}

template <typename T>
void RunMatMulTest(int32_t opset_version, bool is_b_constant = false) {
  std::vector<T> common_input_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (auto t : GenerateTestCases<T>()) {
    OpTester test("MatMul", opset_version);

    int64_t size0 = TensorShape::ReinterpretBaseType(t.input0_dims).SizeHelper(0, t.input0_dims.size());
    std::vector<T> input0_vals(common_input_vals.cbegin(), common_input_vals.cbegin() + size0);
    test.AddInput<T>("A", t.input0_dims, input0_vals);

    int64_t size1 = TensorShape::ReinterpretBaseType(t.input1_dims).SizeHelper(0, t.input1_dims.size());
    std::vector<T> input1_vals(common_input_vals.cbegin(), common_input_vals.cbegin() + size1);
    test.AddInput<T>("B", t.input1_dims, input1_vals, is_b_constant);

    test.AddOutput<T>("Y", t.expected_dims, t.expected_vals);

    // OpenVINO EP: Disabled temporarily matmul broadcasting not fully supported
    // Disable TensorRT because of unsupported data type
    std::unordered_set<std::string> excluded_providers{kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
    if (is_b_constant) {
      // NNAPI: currently fails for the "test 2D empty input" case
      excluded_providers.insert(kNnapiExecutionProvider);
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }
}

TEST(MathOpTest, MatMulFloatType) {
  RunMatMulTest<float>(7, false);
  RunMatMulTest<float>(7, true);
}

TEST(MathOpTest, MatMulDoubleType) {
  RunMatMulTest<double>(7);
}

TEST(MathOpTest, MatMulInt32Type) {
  RunMatMulTest<int32_t>(9);
}

TEST(MathOpTest, MatMulUint32Type) {
  RunMatMulTest<uint32_t>(9);
}

TEST(MathOpTest, MatMulInt64Type) {
  RunMatMulTest<int64_t>(9);
}

TEST(MathOpTest, MatMulUint64Type) {
  RunMatMulTest<uint64_t>(9);
}

struct MatMulSparsityData {
  std::string name;
  OrtSparseFlags sparse_flags = OrtSparseFlags::NOTHING;
  bool transA = false;
  bool transB = false;
  bool is_b_constant = false;
  std::vector<int64_t> output_dims;
  std::vector<float> output_data;
};

//static std::vector<MatMulSparsityData> GenerateSparseTestData() {
//  const std::vector<int64_t> non_t_dims = {9, 9};
//  const std::vector<float> non_t_output = {546, 561, 576, 552, 564, 576, 39, 42, 45,
//                                           1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
//                                           2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
//                                           2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
//                                           3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
//                                           4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
//                                           678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
//                                           786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
//                                           894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};
//
//  // A transposed. Problem trans_A_ and trans_B_ flags are non-standard
//  // so testing is mannual
//  const std::vector<float> t_a_output = {
//      5544, 5688, 5832, 5742, 5868, 5994, 234, 252, 270,
//      5688, 5838, 5988, 5877, 6006, 6135, 261, 282, 303,
//      5832, 5988, 6144, 6012, 6144, 6276, 288, 312, 336,
//      5742, 5877, 6012, 7947, 8154, 8361, 2016, 2088, 2160,
//      5868, 6006, 6144, 8154, 8367, 8580, 2097, 2172, 2247,
//      5994, 6135, 6276, 8361, 8580, 8799, 2178, 2256, 2334,
//      234,  261,  288,  2016, 2097, 2178, 2574, 2682, 2790,
//      252,  282,  312,  2088, 2172, 2256, 2682, 2796, 2910,
//      270,  303,  336,  2160, 2247, 2334, 2790, 2910, 3030};
//
//  //const std::vector<float> t_b_output = {
//  //    55,  145, 235,  266,  338,  410,  113,  131,  149,
//  //    145, 451, 757,  662,  842,  1022, 779,  905,  1031,
//  //    235, 757, 1279, 1058, 1346, 1634, 1445, 1679, 1913,
//  //    266, 662, 1058, 2539, 3277, 4015, 2282, 2624, 2966,
//  //    338, 842, 1346, 3277, 4231, 5185, 3002, 3452, 3902,
//  //    410, 1022, 1634, 4015, 5185, 6355, 3722, 4280, 4838,
//  //    113, 779, 1445, 2282, 3002, 3722, 8911, 10297, 11683,
//  //    131, 905, 1679, 2624, 3452, 4280, 10297, 11899, 13501,
//  //    149, 1031, 1913, 2966, 3902, 4838, 11683, 13501, 15319};
//
//  //const std::vector<float> t_a_b_output = {
//  //    546, 1410, 2274, 2784, 3540, 4296, 678, 786, 894,
//  //    561, 1461, 2361, 2850, 3624, 4398, 789, 915, 1041,
//  //    576, 1512, 2448, 2916, 3708, 4500, 900, 1044, 1188,
//  //    552, 1362, 2172, 4362, 5604, 6846, 2892, 3324, 3756,
//  //    564, 1392, 2220, 4485, 5763, 7041, 3012, 3462, 3912,
//  //    576, 1422, 2268, 4608, 5922, 7236, 3132, 3600, 4068,
//  //    39,  201,  363,  1551, 2037, 2523, 4263, 4911, 5559,
//  //    42,  222,  402,  1608, 2112, 2616, 4494, 5178, 5862,
//  //    45,  243,  441,  1665, 2187, 2709, 4725, 5445, 6165
//  //};
//
//  std::vector<MatMulSparsityData> test_data;
//  test_data.push_back(
//      {"Dense cuBlas", OrtSparseFlags::NOTHING, false, false, false,
//       non_t_dims,
//       non_t_output});
//
//    test_data.push_back(
//      {"Dense cuBlas A transpose", OrtSparseFlags::NOTHING, true, false, false,
//       non_t_dims,
//       t_a_output});
//
//  test_data.push_back(
//      {"COO cuSparse", OrtSparseFlags::USE_COO_FORMAT, false, false, true,
//       non_t_dims,
//       non_t_output});
//
//  test_data.push_back(
//      {"CSR cuSparse", OrtSparseFlags::USE_CSR_FORMAT, false, false, true,
//       non_t_dims,
//       non_t_output});
//
//  return test_data;
//}

TEST(MathOpTest, SparseInitializerTests) {
  OpTester test("MatMul", 13);

  const std::vector<int64_t> input_shape = {9, 9};
  const std::vector<float> input_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  test.AddInput<float>("A", input_shape, input_data);
  // B should the initializer so we designate it as a constant. We can
  // also add sparse_initializer method later one but with current implementation
  // all going to Dense bucket is not really necessary.
  const std::vector<int64_t> initializer_shape = {9, 9};
  const std::vector<float> initializer_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  const bool is_b_constant = true;
  test.AddInput<float>("B", initializer_shape, initializer_data, is_b_constant);
  const std::vector<int64_t> output_shape = {9, 9};
  const std::vector<float> non_t_output = {
      546, 561, 576, 552, 564, 576, 39, 42, 45,
      1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
      2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
      2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
      3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
      4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
      678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
      786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
      894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};

 //const std::vector<float> t_a_output = {
 //     5544, 5688, 5832, 5742, 5868, 5994, 234, 252, 270,
 //     5688, 5838, 5988, 5877, 6006, 6135, 261, 282, 303,
 //     5832, 5988, 6144, 6012, 6144, 6276, 288, 312, 336,
 //     5742, 5877, 6012, 7947, 8154, 8361, 2016, 2088, 2160,
 //     5868, 6006, 6144, 8154, 8367, 8580, 2097, 2172, 2247,
 //     5994, 6135, 6276, 8361, 8580, 8799, 2178, 2256, 2334,
 //     234, 261, 288, 2016, 2097, 2178, 2574, 2682, 2790,
 //     252, 282, 312, 2088, 2172, 2256, 2682, 2796, 2910,
 //     270, 303, 336, 2160, 2247, 2334, 2790, 2910, 3030};

  //const std::vector<float> t_b_output = {
  //    55,  145, 235,  266,  338,  410,  113,  131,  149,
  //    145, 451, 757,  662,  842,  1022, 779,  905,  1031,
  //    235, 757, 1279, 1058, 1346, 1634, 1445, 1679, 1913,
  //    266, 662, 1058, 2539, 3277, 4015, 2282, 2624, 2966,
  //    338, 842, 1346, 3277, 4231, 5185, 3002, 3452, 3902,
  //    410, 1022, 1634, 4015, 5185, 6355, 3722, 4280, 4838,
  //    113, 779, 1445, 2282, 3002, 3722, 8911, 10297, 11683,
  //    131, 905, 1679, 2624, 3452, 4280, 10297, 11899, 13501,
  //    149, 1031, 1913, 2966, 3902, 4838, 11683, 13501, 15319};

  //const std::vector<float> t_a_b_output = {
  //    546, 1410, 2274, 2784, 3540, 4296, 678, 786, 894,
  //    561, 1461, 2361, 2850, 3624, 4398, 789, 915, 1041,
  //    576, 1512, 2448, 2916, 3708, 4500, 900, 1044, 1188,
  //    552, 1362, 2172, 4362, 5604, 6846, 2892, 3324, 3756,
  //    564, 1392, 2220, 4485, 5763, 7041, 3012, 3462, 3912,
  //    576, 1422, 2268, 4608, 5922, 7236, 3132, 3600, 4068,
  //    39,  201,  363,  1551, 2037, 2523, 4263, 4911, 5559,
  //    42,  222,  402,  1608, 2112, 2616, 4494, 5178, 5862,
  //    45,  243,  441,  1665, 2187, 2709, 4725, 5445, 6165
  //};

   test.AddOutput<float>("Y", output_shape, non_t_output);

  // OpenVINO EP: Disabled temporarily matmul broadcasting not fully supported
  // Disable TensorRT because of unsupported data type
  // std::unordered_set<std::string> excluded_providers{kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
  std::unordered_set<std::string> excluded_providers{kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
  if (is_b_constant) {
    // NNAPI: currently fails for the "test 2D empty input" case
    excluded_providers.insert(kNnapiExecutionProvider);
  }
  SessionOptions opts;
  opts.constant_initializers_sparse_flags = OrtSparseFlags::USE_CSR_FORMAT;
  opts.constant_initializers_ell_block_size = 3; // Valid only for ELL
  test.Run(opts, OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  // test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

}  // namespace test
}  // namespace onnxruntime
