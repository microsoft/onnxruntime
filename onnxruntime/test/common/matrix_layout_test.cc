// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/util/matrix_layout.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace {

constexpr bool PositionOperationsAreConstexpr() {
  auto position = make_Position(2, 3);
  position += make_Position(4, 5);
  position *= make_Position(2, 3);
  position -= make_Position(1, 2);
  position /= make_Position(11, 11);
  position.clamp(make_Position(2, 3));

  return position == make_Position(1, 2) && position.sum() == 3 && position.product() == 2;
}

static_assert(PositionOperationsAreConstexpr());

constexpr auto kPosition3D = make_Position(2, 3, 4);
static_assert(decltype(kPosition3D)::kRank == 3);
static_assert(kPosition3D[0] == 2 && kPosition3D[1] == 3 && kPosition3D[2] == 4);
static_assert(!std::is_constructible_v<Position<3>, Position<2>>);

constexpr Position<2, int64_t> kWidePosition = make_Position<int64_t>(5, 7);
constexpr Position<2> kConvertedPosition = kWidePosition;
static_assert(kConvertedPosition == make_Position(5, 7));

using TestMatrixShape = MatrixShape<3, 4>;
static_assert(TestMatrixShape::kCount == 12);
static_assert(TestMatrixShape::toCoord() == make_Position(3, 4));

constexpr auto kRowMajorLayout = RowMajorLayout::packed(TestMatrixShape::toCoord());
static_assert(kRowMajorLayout.stride() == 4);
static_assert(kRowMajorLayout(make_Position(2, 1)) == 9);
static_assert(kRowMajorLayout.inverse(9) == make_Position(2, 1));

constexpr auto kColumnMajorLayout = ColumnMajorLayout::packed(TestMatrixShape::toCoord());
static_assert(kColumnMajorLayout.stride() == 3);
static_assert(kColumnMajorLayout(make_Position(2, 1)) == 5);
static_assert(kColumnMajorLayout.inverse(5) == make_Position(2, 1));

TEST(MatrixLayoutTest, MapsRuntimeCoordinates) {
  const int row = 2;
  const int column = 1;
  const auto coordinate = make_Position(row, column);

  EXPECT_EQ(kRowMajorLayout(coordinate), 9);
  EXPECT_EQ(kColumnMajorLayout(coordinate), 5);
}

}  // namespace
}  // namespace onnxruntime
