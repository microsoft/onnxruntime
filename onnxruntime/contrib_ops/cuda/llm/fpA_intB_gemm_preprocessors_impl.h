/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "core/common/common.h"
#include "core/common/span_utils.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {

constexpr int get_weight_quant_bits(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::W8_A16:
      return 8;
    case QuantType::W4_A16:
    case QuantType::W4_AFP8:
      return 4;
  }

  return -1;
}

struct LayoutDetails {
  enum class Layout {
    UNKNOWN,
    ROW_MAJOR,
    COLUMN_MAJOR
  };

  Layout layoutB = Layout::UNKNOWN;
  int rows_per_column_tile = 1;
  int columns_interleaved = 1;

  bool uses_imma_ldsm = false;
};

template <typename Layout>
struct getLayoutDetails {
};

template <>
struct getLayoutDetails<cutlass::layout::RowMajor> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::ROW_MAJOR;
    return layout_details;
  }
};

template <>
struct getLayoutDetails<cutlass::layout::ColumnMajor> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
    return layout_details;
  }
};

template <int RowsPerTile, int ColumnsInterleaved>
struct getLayoutDetails<cutlass::layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
    layout_details.rows_per_column_tile = RowsPerTile;
    layout_details.columns_interleaved = ColumnsInterleaved;
    return layout_details;
  }
};

template <typename cutlassArch, typename TypeA, typename TypeB>
LayoutDetails getLayoutDetailsForArchAndQuantType() {
  using CompileTraits = cutlass::gemm::kernel::LayoutDetailsB<TypeA, TypeB, cutlassArch>;
  using LayoutB = typename CompileTraits::Layout;
  using MmaOperator = typename CompileTraits::Operator;
  LayoutDetails details = getLayoutDetails<LayoutB>()();
  details.uses_imma_ldsm = std::is_same<MmaOperator, cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA>::value;
  return details;
}

template <typename cutlassArch>
LayoutDetails getLayoutDetailsForArch(QuantType quant_type) {
  LayoutDetails details;
  switch (quant_type) {
    case QuantType::W8_A16:
      details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, uint8_t>();
      break;
    case QuantType::W4_A16:
      details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, cutlass::uint4b_t>();
      break;
    case QuantType::W4_AFP8:
      details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::float_e4m3_t, cutlass::uint4b_t>();
      break;
  }
  return details;
}

LayoutDetails getLayoutDetailsForTransform(QuantType quant_type, int arch) {
  ORT_ENFORCE(arch >= 75, "Unsupported CUDA architecture: ", arch);
  if (arch < 80) {
    return getLayoutDetailsForArch<cutlass::arch::Sm75>(quant_type);
#ifndef EXCLUDE_SM_90
  } else if (arch >= 90 && arch < 100) {
    return getLayoutDetailsForArch<cutlass::arch::Sm90>(quant_type);
#endif
  } else {
    return getLayoutDetailsForArch<cutlass::arch::Sm80>(quant_type);
  }
}

constexpr std::array<int, 16> kPerm_W8_A16 = {
    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};

constexpr std::array<int, 32> kPerm_W4_A16 = {
    0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
    4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31};

constexpr std::array<int, 32> kPerm_W4_AFP8 = {
    0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
    8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};

// Permutes the rows of B in a way that is compatible with Turing+ architectures.
//
// Throws an error for other architectures.
// The data is permuted such that:
// For W8_A16, each group of 16 rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
// For W4_A16, each group of 32 rows is permuted using the map below:
//  0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
// For W4_A8, see the map in the code. The idea is similar to above.
// The goal of this permutation is to ensure data ends up in the correct threads after
// we execute LDSM. It counteracts the effect of the data being of different widths.
// For more information about the expected layouts, see the MMA section in the PTX docs.
gsl::span<const int> get_permutation_map(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::W8_A16:
      return AsSpan(kPerm_W8_A16);
    case QuantType::W4_A16:
      return AsSpan(kPerm_W4_A16);
    case QuantType::W4_AFP8:
      return AsSpan(kPerm_W4_AFP8);
    default:
      ORT_THROW("Invalid quantization type for LDSM permutation");
  }
}

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
