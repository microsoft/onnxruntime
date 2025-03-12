// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <type_traits>
#include <memory>
#include <utility>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunDepthToSpace(const std::vector<T>& input,
                     const std::vector<int64_t>& input_shape,
                     const int64_t blocksize,
                     const int64_t channels_last,
                     const std::string mode,
                     const std::vector<T>& output,
                     const std::vector<int64_t>& output_shape,
                     OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  auto run_test = [&]() {
    OpTester test("DepthToSpace", 1, kMSDomain);

    test.AddAttribute<int64_t>("blocksize", blocksize);
    test.AddAttribute<int64_t>("channels_last", channels_last);
    test.AddAttribute<std::string>("mode", mode);

    test.AddInput<T>("input", input_shape, input);
    test.AddOutput<T>("output", output_shape, output);

    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());
    test.Run(expect_result, "", {}, nullptr, &eps);
  };

  run_test();
}

TEST(DepthToSpaceOpTest, ContribDCR) {

    constexpr int64_t N = 2, H = 3, W = 2, C = 12;
    constexpr int64_t blocksize = 2;
    std::vector<uint8_t> input = {
          0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
         12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,

         24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
         36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,

         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
         60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,


         72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,

         96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,

        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
        132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
    };
    std::vector<int64_t> input_shape = {N, H, W, C};
    std::vector<uint8_t> output = {
          0,   1,   2,
          3,   4,   5,
         12,  13,  14,
         15,  16,  17,

          6,   7,   8,
          9,  10,  11,
         18,  19,  20,
         21,  22,  23,

         24,  25,  26,
         27,  28,  29,
         36,  37,  38,
         39,  40,  41,

         30,  31,  32,
         33,  34,  35,
         42,  43,  44,
         45,  46,  47,

         48,  49,  50,
         51,  52,  53,
         60,  61,  62,
         63,  64,  65,

         54,  55,  56,
         57,  58,  59,
         66,  67,  68,
         69,  70,  71,


         72,  73,  74,
         75,  76,  77,
         84,  85,  86,
         87,  88,  89,

         78,  79,  80,
         81,  82,  83,
         90,  91,  92,
         93,  94,  95,

         96,  97,  98,
         99, 100, 101,
        108, 109, 110,
        111, 112, 113,

        102, 103, 104,
        105, 106, 107,
        114, 115, 116,
        117, 118, 119,

        120, 121, 122,
        123, 124, 125,
        132, 133, 134,
        135, 136, 137,

        126, 127, 128,
        129, 130, 131,
        138, 139, 140,
        141, 142, 143
    };
    std::vector<int64_t> output_shape = {N, H * blocksize, W * blocksize, C / (blocksize * blocksize)};

    RunDepthToSpace<uint8_t>(input, input_shape, blocksize, 1, "DCR", output, output_shape);
}

TEST(DepthToSpaceOpTest, ContribCRD) {

  constexpr int64_t N = 2, H = 3, W = 2, C = 12;
  constexpr int64_t blocksize = 2;
  std::vector<uint8_t> input = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
       12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,

       24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
       36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,

       48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
       60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,


       72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
       84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,

       96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
      108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,

      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
      132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
  };
  std::vector<int64_t> input_shape = {N, H, W, C};
  std::vector<uint8_t> output = {
        0,   4,   8,
        1,   5,   9,
       12,  16,  20,
       13,  17,  21,

        2,   6,  10,
        3,   7,  11,
       14,  18,  22,
       15,  19,  23,

       24,  28,  32,
       25,  29,  33,
       36,  40,  44,
       37,  41,  45,

       26,  30,  34,
       27,  31,  35,
       38,  42,  46,
       39,  43,  47,

       48,  52,  56,
       49,  53,  57,
       60,  64,  68,
       61,  65,  69,

       50,  54,  58,
       51,  55,  59,
       62,  66,  70,
       63,  67,  71,


       72,  76,  80,
       73,  77,  81,
       84,  88,  92,
       85,  89,  93,

       74,  78,  82,
       75,  79,  83,
       86,  90,  94,
       87,  91,  95,

       96, 100, 104,
       97, 101, 105,
      108, 112, 116,
      109, 113, 117,

       98, 102, 106,
       99, 103, 107,
      110, 114, 118,
      111, 115, 119,

      120, 124, 128,
      121, 125, 129,
      132, 136, 140,
      133, 137, 141,

      122, 126, 130,
      123, 127, 131,
      134, 138, 142,
      135, 139, 143};
  std::vector<int64_t> output_shape = {N, H * blocksize, W * blocksize, C / (blocksize * blocksize)};

  RunDepthToSpace<uint8_t>(input, input_shape, blocksize, 1, "CRD", output, output_shape);
}

}  // namespace test
}  // namespace onnxruntime
