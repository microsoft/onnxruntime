// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "mlas.h"

TEST(MlasPool1D, LargeOverlappingWindows) {
  struct Shape {
    int64_t width, kernel, stride, left, right, output;
  };
  const Shape cases[] = {
      {97, 32, 1, 15, 16, 97},                                 // Multiple blocks and a partial final block.
      {67, 33, 8, 3, 14, 8},                                   // Unequal pads and a final ceil-mode window.
      {7, 64, 1, 31, 32, 7},                                   // The kernel is larger than the input.
      {97, 31, 1, 15, 15, 97},                                 // Below the kernel threshold.
      {97, 32, 9, 15, 16, 11},                                 // Above the stride threshold.
      {34, 32, 1, 0, 0, 3},                                    // Below the output-count threshold.
      {35, 32, 1, 0, 0, 4},                                    // At the output-count threshold.
      {7, int64_t(1) << 32, 1, 0, (int64_t(1) << 32) - 1, 7},  // The kernel must fit size_t on 32-bit targets.
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const auto& s : cases) {
    SCOPED_TRACE(::testing::Message() << "width=" << s.width << " kernel=" << s.kernel
                                      << " stride=" << s.stride);
    const int64_t input_shape[] = {2, 3, s.width};
    const int64_t output_shape[] = {2, 3, s.output};
    const int64_t pads[] = {s.left, s.right};
    const size_t input_width = static_cast<size_t>(s.width);
    const size_t output_width = static_cast<size_t>(s.output);
    std::vector<float> input(6 * input_width);
    std::vector<float> output(6 * output_width, nan);
    for (int64_t i = 0; i < s.width; i++) {
      const size_t input_index = static_cast<size_t>(i);
      input[input_index] = float(i * 17 % 53 - 26);
      input[input_width + input_index] = nan;
      input[2 * input_width + input_index] = -inf;
      input[3 * input_width + input_index] = i % 2 ? 0.0f : -0.0f;
      input[4 * input_width + input_index] = i % 2 ? -0.0f : 0.0f;
      input[5 * input_width + input_index] = i % 3 ? float(i - s.width) : nan;
    }
    input[5 * input_width + input_width / 2] = inf;

    MlasPool(MlasMaximumPooling, 1, input_shape, &s.kernel, pads, &s.stride,
             output_shape, input.data(), output.data(), nullptr);

    for (int64_t c = 0; c < 6; c++) {
      for (int64_t p = 0; p < s.output; p++) {
        float expected = std::numeric_limits<float>::lowest();
        const int64_t start = p * s.stride - s.left;
        for (int64_t i = std::max(int64_t(0), start); i < std::min(s.width, start + s.kernel); i++) {
          expected = std::max(expected, input[static_cast<size_t>(c * s.width + i)]);
        }
        // Include the sign of zero in the comparison.
        EXPECT_EQ(0, std::memcmp(&expected, &output[static_cast<size_t>(c * s.output + p)], sizeof(float)))
            << "channel=" << c << " output=" << p;
      }
    }
  }
}
