// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <limits>
#include <optional>
#include <vector>

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

#include "core/providers/dml/DmlExecutionProvider/src/PooledUploadHeap.h"

namespace onnxruntime {
namespace test {

#ifndef ORT_NO_EXCEPTIONS
TEST(DmlUploadHeapTest, RejectsAllocationLargerThanMaximumChunkBeforeResourceCreation) {
  EXPECT_NO_THROW(Dml::detail::ValidateUploadHeapAllocationSize(Dml::detail::c_maxUploadHeapChunkSize));
  EXPECT_ANY_THROW(Dml::detail::ValidateUploadHeapAllocationSize(Dml::detail::c_maxUploadHeapChunkSize + 1));
}
#endif

TEST(DmlUploadHeapTest, ConvertsRepresentableUploadSize) {
  const auto converted = Dml::detail::TryConvertToUploadSize(std::numeric_limits<size_t>::max());

  ASSERT_TRUE(converted.has_value());
  EXPECT_EQ(*converted, std::numeric_limits<size_t>::max());
}

TEST(DmlUploadHeapTest, RejectsUploadSizeThatOverflowsSizeTOnX86) {
  if (sizeof(size_t) > sizeof(uint32_t)) {
    GTEST_SKIP() << "This regression applies to 32-bit size_t.";
  }

  constexpr uint64_t kUint32Overflow = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(Dml::detail::TryConvertToUploadSize(kUint32Overflow).has_value());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_DML
