// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunSliceTest(const std::vector<int64_t>& input_dims,
                  const std::vector<T>& input_vals,
                  const std::vector<int64_t>& starts,
                  const std::vector<int64_t>& ends,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& steps,
                  const std::vector<int64_t>& output_dims,
                  const std::vector<T>& output_vals) {
  // V1-9 (soes not support steps)
  if (steps.size() == 0)
  {
	  OpTester testv9("Slice", 9);
	  testv9.AddAttribute("starts", starts);
	  testv9.AddAttribute("ends", ends);
	  if (axes.size() != 0)
		testv9.AddAttribute("axes", axes);
	  testv9.AddInput<T>("data", input_dims, input_vals);
	  testv9.AddOutput<T>("output", output_dims, output_vals);
	  testv9.Run();
  }

  /*
  //V10
  OpTester testv10("Slice", 10);
  testv10.AddInput<T>("data", input_dims, input_vals);
  testv10.AddInput<int64_t>({starts.size()}, starts);
  testv10.AddInput<int64_t>({ends.size()}, ends);
  if (axes.size() != 0)
    testv10.AddInput<int64_t>({axes.size()}, axes);
  if (steps.size() != 0)
    testv10.AddInput<int64_t>({steps.size()}, steps);
  testv10.AddOutput<T>("output", output_dims, output_vals);
  testv10.Run();
  */
}

TEST(SliceTest, Slice1D) {
  RunSliceTest<float>
	          ({6},
               {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
               {2},
               {4},
               {0},
               {},
               {2},
               {2.0f, 3.0f});
}


TEST(SliceTest, Slice1D_Perf) {
  std::vector<float> input(1000, 2.0f);
  std::vector<float> output(500, 2.0f);
  RunSliceTest<float>({1000},
               input,
               {2},
               {502},
               {0},
               {},
               {500},
               output);
}

TEST(SliceTest, Slice2D_OutOfBounds) {
  RunSliceTest<float>({2, 3},
               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
               {0, 1000},
               {10, 1000},
               {0, 1},
               {},
               {2, 0},
               {});
}

TEST(SliceTest, Slice2D_OneAxis) {
  RunSliceTest<float>({6, 4},
               {00.0f, 01.0f, 02.0f, 03.0f,
                10.0f, 11.0f, 12.0f, 13.0f,
                20.0f, 21.0f, 22.0f, 23.0f,
                30.0f, 31.0f, 32.0f, 33.0f,
                40.0f, 41.0f, 42.0f, 43.0f,
                50.0f, 51.0f, 52.0f, 53.0f},
               {1},
               {3},
               {0},
               {},
               {2, 4},
               {10.0f, 11.0f, 12.0f, 13.0f,
                20.0f, 21.0f, 22.0f, 23.0f});
}

TEST(SliceTest, Slice2D_TwoAxes) {
  RunSliceTest<float>({6, 4},
               {00.0f, 01.0f, 02.0f, 03.0f,
                10.0f, 11.0f, 12.0f, 13.0f,
                20.0f, 21.0f, 22.0f, 23.0f,
                30.0f, 31.0f, 32.0f, 33.0f,
                40.0f, 41.0f, 42.0f, 43.0f,
                50.0f, 51.0f, 52.0f, 53.0f},
               {2, 3},
               {1000 ,-1},
               {1, 0},
               {},
               {2, 2},
               {32.0f, 33.0f,
                42.0f, 43.0f});
}

TEST(SliceTest, Slice2D_TwoAxesEque) {
  RunSliceTest<float>({6, 4},
               {00.0f, 01.0f, 02.0f, 03.0f,
                10.0f, 11.0f, 12.0f, 13.0f,
                20.0f, 21.0f, 22.0f, 23.0f,
                30.0f, 31.0f, 32.0f, 33.0f,
                40.0f, 41.0f, 42.0f, 43.0f,
                50.0f, 51.0f, 52.0f, 53.0f},
               {2, 3},
               {1000, 3},
               {1, 0},
               {},
               {0, 2},
               {});
}

TEST(SliceTest, Slice3D) {
  RunSliceTest<float>({3, 3, 3},
               {111.0f, 112.0f, 113.0f,
                121.0f, 122.0f, 123.0f,
                131.0f, 132.0f, 133.0f,

                211.0f, 212.0f, 213.0f,
                221.0f, 222.0f, 223.0f,
                231.0f, 232.0f, 233.0f,

                311.0f, 312.0f, 313.0f,
                321.0f, 322.0f, 323.0f,
                331.0f, 332.0f, 333.0f},
               {0, 1, 1},
               {1000, 1000, 1000},
               {},
               {},
               {3, 2, 2},
               {122.0f, 123.0f,
                132.0f, 133.0f,

                222.0f, 223.0f,
                232.0f, 233.0f,

                322.0f, 323.0f,
                332.0f, 333.0f});
}

TEST(SliceTest, Slice1D_Int) {
  RunSliceTest<int32_t>({6},
               {0L, 1L, 2L, 3L, 4L, 5L},
               {2},
               {4},
               {0},
               {},
               {2},
               {2L, 3L});
}

TEST(SliceTest, Slice1D_String) {
  RunSliceTest<std::string>({6},
               {"0", "1", "2", "3", "4", "5"},
               {2},
               {4},
               {0},
               {},
               {2},
               {"2", "3"});
}

}  // namespace test
}  // namespace onnxruntime
