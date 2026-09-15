// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef _GAMING_XBOX_SCARLETT
#include <d3d12_xs.h>
#elif defined(_GAMING_XBOX_XBOXONE)
#include <d3d12_x.h>
#else
#include "directx/d3d12.h"
#endif
#include <gsl/gsl>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

#include "core/providers/dml/DmlExecutionProvider/src/ReadbackHeap.h"

namespace onnxruntime {
namespace test {

TEST(DmlReadbackHeapTest, ComputeTotalReadbackSizeDoesNotWrapAtUint32Max) {
  if (sizeof(size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "This regression needs a size_t wider than uint32_t.";
  }

  constexpr size_t kUint32Max = std::numeric_limits<uint32_t>::max();
  const std::array<size_t, 3> sizes = {kUint32Max, 1, 7};

  EXPECT_EQ(Dml::detail::ComputeTotalReadbackSize(gsl::make_span(sizes.data(), sizes.size())), kUint32Max + 8);
}

TEST(DmlReadbackHeapTest, ComputeTotalReadbackSizeIncludesSizesAfterZero) {
  const std::array<size_t, 3> sizes = {5, 0, 7};

  EXPECT_EQ(Dml::detail::ComputeTotalReadbackSize(gsl::make_span(sizes.data(), sizes.size())), 12u);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(DmlReadbackHeapTest, ComputeTotalReadbackSizeRejectsSizeTOverflow) {
  const std::array<size_t, 2> sizes = {std::numeric_limits<size_t>::max(), 1};

  EXPECT_ANY_THROW((void)Dml::detail::ComputeTotalReadbackSize(gsl::make_span(sizes.data(), sizes.size())));
}

TEST(DmlReadbackHeapTest, ComputeTotalReadbackSizeRejectsMidBatchOverflow) {
  const size_t half_max = std::numeric_limits<size_t>::max() / 2;
  const std::array<size_t, 2> sizes = {half_max + 1, half_max + 1};

  EXPECT_ANY_THROW((void)Dml::detail::ComputeTotalReadbackSize(gsl::make_span(sizes.data(), sizes.size())));
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_DML
