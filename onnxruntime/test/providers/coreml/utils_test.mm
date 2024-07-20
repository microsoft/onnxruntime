// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__APPLE__)
#import <CoreML/CoreML.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/providers/coreml/model/model.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
auto ValidateGetInfo(MLMultiArray* array,
                     int64_t expected_num_blocks, int64_t expected_block_size, int64_t expected_stride,
                     bool expect_valid) {
  int64_t num_blocks = 0;
  int64_t block_size = 0;
  int64_t stride = 0;
  auto status = coreml::GetMLMultiArrayCopyInfo(array, num_blocks, block_size, stride);

  if (!expect_valid) {
    ASSERT_STATUS_NOT_OK(status);
    return;
  }

  ASSERT_STATUS_OK(status);
  ASSERT_EQ(num_blocks, expected_num_blocks);
  ASSERT_EQ(block_size, expected_block_size);
  ASSERT_EQ(stride, expected_stride);
}
}  // namespace

TEST(CoreMLUtils, GetMLMultiArrayReadInfo) {
  // fake pointer. we don't read any data but initWithDataPointer requires a non-null address
  void* data = reinterpret_cast<void*>(0xfeedf00d);

  // a dim is non-contiguous if the stride is > the total number of elements in its inner dimensions

  // dim -1 with non-contiguous data. 1 element (as it's the inner-most dimension) but the stride is 2.
  {
    NSArray<NSNumber*>* shape = @[ @1, @1, @8, @8 ];
    NSArray<NSNumber*>* strides = @[ @128, @128, @16, @2 ];

    auto* array = [[MLMultiArray alloc] initWithDataPointer:data
                                                      shape:shape
                                                   dataType:MLMultiArrayDataTypeInt32
                                                    strides:strides
                                                deallocator:^(void* /* bytes */) {
                                                }
                                                      error:nil];
    ValidateGetInfo(array, 64, 1, 2, true);
  }

  // dim -2 with non-contiguous data. 8 elements in the inner dimension but the stride is 16.
  {
    NSArray<NSNumber*>* shape = @[ @1, @1, @8, @8 ];
    NSArray<NSNumber*>* strides = @[ @128, @128, @16, @1 ];

    auto* array = [[MLMultiArray alloc] initWithDataPointer:data
                                                      shape:shape
                                                   dataType:MLMultiArrayDataTypeInt32
                                                    strides:strides
                                                deallocator:^(void* /* bytes */) {
                                                }
                                                      error:nil];
    ValidateGetInfo(array, 8, 8, 16, true);
  }

  // dim -3 with non-contiguous data. 16 elements in the innder dimensions but stride is 24.
  {
    NSArray<NSNumber*>* shape = @[ @1, @2, @4, @4 ];
    NSArray<NSNumber*>* strides = @[ @48, @24, @4, @1 ];

    auto* array = [[MLMultiArray alloc] initWithDataPointer:data
                                                      shape:shape
                                                   dataType:MLMultiArrayDataTypeInt32
                                                    strides:strides
                                                deallocator:^(void* /* bytes */) {
                                                }
                                                      error:nil];

    ValidateGetInfo(array, 2, 16, 24, true);
  }

  // two non-contiguous dims (dim -2 and dim -3)
  // dim -2 has 4 elements in the inner dimension and stride of 8
  // dim -3 has 32 elements in the inner dimensions (we need to include the empty elements from the non-contiguous data
  // in dim -2) and stride of 48
  {
    // dim
    NSArray<NSNumber*>* shape = @[ @1, @2, @4, @4 ];
    NSArray<NSNumber*>* strides = @[ @96, @48, @8, @1 ];

    auto* array = [[MLMultiArray alloc] initWithDataPointer:data
                                                      shape:shape
                                                   dataType:MLMultiArrayDataTypeInt32
                                                    strides:strides
                                                deallocator:^(void* /* bytes */) {
                                                }
                                                      error:nil];

    ValidateGetInfo(array, 0, 0, 0, false);
  }
}
}  // namespace test
}  // namespace onnxruntime
#endif
