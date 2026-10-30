// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/nn/padding.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(CannPaddingTest, ConvModesPreserveAttributeStrings) {
  EXPECT_STREQ(cann::GetConvAutoPadMode(AutoPadType::NOTSET), "NOTSET");
  EXPECT_STREQ(cann::GetConvAutoPadMode(AutoPadType::VALID), "VALID");
  EXPECT_STREQ(cann::GetConvAutoPadMode(AutoPadType::SAME_UPPER), "SAME_UPPER");
  EXPECT_STREQ(cann::GetConvAutoPadMode(AutoPadType::SAME_LOWER), "SAME_LOWER");
}

TEST(CannPaddingTest, PoolModesPreserveAttributeStrings) {
  EXPECT_STREQ(cann::GetPoolAutoPadMode(AutoPadType::NOTSET), "CALCULATED");
  EXPECT_STREQ(cann::GetPoolAutoPadMode(AutoPadType::VALID), "VALID");
  EXPECT_STREQ(cann::GetPoolAutoPadMode(AutoPadType::SAME_UPPER), "SAME");
  EXPECT_STREQ(cann::GetPoolAutoPadMode(AutoPadType::SAME_LOWER), "SAME");
}

}  // namespace test
}  // namespace onnxruntime
