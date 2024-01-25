/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80_testcu.cu
 *
 * Abstract:
 *   Test code for invoking block-wise quantized 4b GEMM kernels.
 *   This part requires CUTLASS header files, which do not play
 *   well with gtest headers.
 */

#include "core/mickey/blk_q4/f16_gemm_sm80.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "core/common/common.h"

#include "blkq4_fp16_gemm_sm80.h"

namespace onnxruntime {
namespace cuda{
namespace test{

Status sm80_supported(){
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::ostringstream ss;
    ss << "Unable to obtain GPU device properties: " << cudaGetErrorString(error);
    return Status(common::ONNXRUNTIME, common::ENGINE_ERROR, ss.str());
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::ostringstream ss;
    ss << "Device compute capability mismatch, desired 8.0, actual " << props.major << "." << props.minor;
    return Status(common::ONNXRUNTIME, common::ENGINE_ERROR, ss.str());
  }
  return Status::OK();
}

/**
 * @brief Reference implementation of GEMM
 *        Copied directly from cutlass util/reference/device/gemm.h
 *        for the strange reason that compiler insists on asking
 *        for explicit stream argument in kernel launch.
*/
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename AccumulatorType
>
void compute_gemm_ref(
  cutlass::gemm::GemmCoord problem_size,
  ScalarType alpha,
  cutlass::TensorRef<ElementA, LayoutA> tensor_a,
  cutlass::TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  cutlass::TensorRef<ElementC, LayoutC> tensor_c,
  cutlass::TensorRef<ElementC, LayoutC> tensor_d,
  AccumulatorType initial_accum = AccumulatorType(0)) {

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  using OutputTile = cutlass::MatrixShape<4, 4>;

  dim3 block(16, 8);

  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kRow - 1) / (block.x * OutputTile::kRow),
    (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn)
  );

  // Launch a GEMM kernel
  cutlass::reference::device::kernel::Gemm<
    cutlass::TensorRef<ElementA, LayoutA>,
    cutlass::TensorRef<ElementB, LayoutB>,
    cutlass::TensorRef<ElementC, LayoutC>,
    ScalarType,
    AccumulatorType,
    OutputTile,
    cutlass::multiply_add<AccumulatorType>,
    cutlass::NumericConverter<ElementC, ScalarType>
  ><<<grid, block, 0, 0>>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    tensor_d,
    initial_accum
  );
}
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Converting cutlass tensor to MatrixRef
//

template <
  typename Element,
  typename LayoutCutlass,
  typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
  >
__forceinline__
MatrixRef<Element, Layout, true> make_MatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
                || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
  return MatrixRef<Element, Layout, true>(ptr, tensor.capacity(), shape);
}

template <
  typename Element,
  typename LayoutCutlass,
  typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
  >
__forceinline__
MatrixRef<Element const, Layout, true> make_ConstMatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
                || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  return MatrixRef<Element const, Layout, true>(tensor.host_data(), tensor.capacity(), shape);
}

//
// Invoking the kernel
//

template<
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k) {

  using ElementDequant = cutlass::half_t;
  using QuantBlocking =
    typename std::conditional<column_wise_blocking,
                     cutlass::MatrixShape<block_size, 1>,
                     cutlass::MatrixShape<1, block_size>>::type;

  using GemmRunner = BlkQ4F16GemmImpl<ElementDequant, QuantBlocking, small_m, has_offsets>;

  using ElementAccumulator = typename GemmRunner::ElementAccumulator;
  using ElementComputeEpilogue = typename GemmRunner::ElementComputeEpilogue;
  using ElementInputA = typename GemmRunner::ElementInputA;
  using ElementOutput = typename GemmRunner::ElementOutput;
  using ElementW = typename GemmRunner::ElementW;
  using ElementWPack = typename GemmRunner::ElementWPack;
  using ElementQScale = typename GemmRunner::ElementQScale;
  using ElementQOffset = typename GemmRunner::ElementQOffset;

  using LayoutInputA = typename GemmRunner::LayoutInputA;
  using LayoutOutput = typename GemmRunner::LayoutOutput;
  using LayoutInputWPack = typename GemmRunner::LayoutInputWPack;
  using LayoutInputQScale = typename GemmRunner::LayoutInputQScale;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K

  // Create weight matrix with dimensions K x N.
  // Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
  // troubles. Since the layout is column major, we are packing 2 weights in a column
  // into one int8
  cutlass::HostTensor<ElementW, LayoutInputWPack> tensor_weight(
      {problem_size.k()/2, problem_size.n()});
  // Create weight quantization scale and offset with dimensions K x N
  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      2);  // <- Fill matrix A on host with uniform-distribution random data
  if constexpr (has_offsets) {
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_offset.host_view(),
        1,
        ElementQOffset(0),
        ElementQOffset(15),
        0);  // <- Fill weight offsets on host with uniform-distribution random data
  }
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

  //
  // For testing quantization and dequantization, it is not straight
  // forward to avoid flaky tests due to rounding errors. The way we
  // try to achieve this is to:
  // 1. Generate a set of quantized weights, scales and offsets
  // 2. Dequantize the weights
  // 3. Quantize the dequantized weights
  // 4. Compare the dequantied-and-then-quantized weights with
  //    the original quantized weights
  //
  // Random filling of the initial values are key to get this right.
  // For weights, we must ensure each block gets a full range of
  // values, i.e. must contain 0 and 15. And for scales, they must
  // all be positive.
  //

  int v = 7;
  for (int c = 0; c < tensor_weight.extent()[1]; c++) {
    for (int r = 0; r < tensor_weight.extent()[0]; ++r) {
      uint8_t v0 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }
      uint8_t v1 = 0;
      v1 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }

      tensor_weight.at({r, c}) = ElementW((v1 << 4) | v0);
    }
  }

  for (int c = 0; c < tensor_scale.extent()[1]; c++) {
    for (int r = 0; r < tensor_scale.extent()[0]; ++r) {
      int f = (((c * v + r + v / 3 ) % 63) + 1);
      v += 41;
      int m = (c * v + r + v / 8 ) % 4;
      tensor_scale.at({r, c}) = ElementQScale(static_cast<float>(f) / static_cast<float>(1 << (2 + m)));
    }
  }

//   // Fill tensor_weight with the patterned data, so that we can use
//   // print to make sure the layout matches after loaded to registers
//   int loop_val = 0;
//   int offset = 3;
//   for (int col_tile = 0; col_tile < tensor_weight.extent().column()/8; ++col_tile) {
//     for (int row_tile = 0; row_tile < tensor_weight.extent().row()/4; ++row_tile) {
//       for (int col = 0; col < 8; ++col) {
//         for (int row = 0; row < 4; ++row) {
//           auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
//           auto val = (loop_val + offset) % 256;
//           tensor_weight.at(weight_cord) = ElementW(val);
//           loop_val++;
//           if (loop_val == 256) {
//             loop_val = 0;
//             offset += 11;
//           }
//         }
//       }
//     }
//   }
//   for (int col = 0; col < tensor_scale.extent().column(); ++col){
//     int c =  col * QuantBlocking::kColumn;
//     for (int row = 0; row < tensor_scale.extent().row(); ++row){
//       int r = row * QuantBlocking::kRow;
//       auto weight_cord = cutlass::make_Coord(r/2, c);
//       int w = 0;
//       if (r % 2 == 0) {
//         w = int(tensor_weight.at(weight_cord) & 0x0f);
//       } else {
//         w = int(tensor_weight.at(weight_cord) >> 4);
//       }
//       tensor_scale.at({row, col}) = w;
// #ifdef USE_QUANT_OFFSET
//       tensor_offset.at({row, col}) = ElementQOffset(w);
// #endif
//     }
//   }

  // int fill_val = -512;
  // int factor = 1;
  // for (int col = 0; col < tensor_scale.extent().column(); ++col){
  //   for (int row = 0; row < tensor_scale.extent().row(); ++row){
  //     tensor_scale.at({row, col}) = ElementQScale((float)fill_val * float(factor));
  //     fill_val++;
  //     if (fill_val == 512) {
  //       fill_val = -512;
  //       factor += 1;
  //     }
  //   }
  // }

  // std::cout << "Matrix Weight:\n" << tensor_weight.host_view() << "\n";

  // Prepacking weight matrix and quantization meta data ...

  cutlass::HostTensor<ElementW, LayoutInputWPack> tensor_weight_prepacked(
    cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  onnxruntime::test::sm80_prepack_weights_ref(
    problem_size.k(), problem_size.n(),
    make_ConstMatrixRef(tensor_weight),
    make_MatrixRef(tensor_weight_prepacked));

  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});

  auto scale_ref = make_ConstMatrixRef(tensor_scale);
  onnxruntime::test::sm80_prepack_quant_scales_ref<ElementQScale, typename decltype(scale_ref)::Layout, QuantBlocking>(
      problem_size.k(), problem_size.n(), scale_ref,
      make_MatrixRef(tensor_scale_prepacked));
  if constexpr (has_offsets) {
    auto offset_ref = make_ConstMatrixRef(tensor_offset);
    onnxruntime::test::sm80_prepack_quant_offsets_ref<typename decltype(offset_ref)::Layout, QuantBlocking>(
        problem_size.k(), problem_size.n(), offset_ref,
        make_MatrixRef(tensor_offset_prepacked));
  }

  // Copy data from host to GPU...
  tensor_a.sync_device();
  tensor_weight_prepacked.sync_device();
  tensor_scale_prepacked.sync_device();
  if constexpr (has_offsets) {
    tensor_offset_prepacked.sync_device();
  }
  tensor_c.sync_device();
  tensor_d.sync_device();
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(tensor_weight_prepacked.device_data()),
    LayoutInputWPack::packed({problem_size.k()/2, problem_size.n()/2}));

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets){
    status = GemmRunner::run(
      nullptr, problem_size, tensor_a.device_ref(), ref_W,
      tensor_scale_prepacked.device_ref(), tensor_offset_prepacked.device_ref(),
      tensor_c.device_ref(), tensor_d.device_ref());
  } else {
    status = GemmRunner::run(
      nullptr, problem_size, tensor_a.device_ref(), ref_W,
      tensor_scale_prepacked.device_ref(),
      tensor_c.device_ref(), tensor_d.device_ref());
  }
  ORT_ENFORCE(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));

  // Preparing reference kernel arguments
  // Dequantizing weights and running reference kernel

  using ElementInputB = ElementInputA;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create dequantized matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Dequantize weights and save into matrix B for reference
  for (int col = 0; col < tensor_b.extent().column(); ++col){
    for (int row = 0; row < tensor_b.extent().row(); ++row) {
      auto weight_cord = cutlass::make_Coord(row/2, col);
      auto scale_cord = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      const uint8_t offset = has_offsets ? tensor_offset.at(scale_cord) : 8;
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_weight.at(weight_cord) & 0x0f) - offset;
      } else {
        w = int(tensor_weight.at(weight_cord) >> 4) - offset;
      }
      auto scale = tensor_scale.at(scale_cord);
      tensor_b.at({row, col}) = scale * float(w);
    }
  }
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  tensor_b.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  compute_gemm_ref<ElementInputA, LayoutInputA,
               ElementInputB, LayoutInputB,
               ElementOutput, LayoutOutput,
               ElementComputeEpilogue, ElementAccumulator>(
      problem_size,
      alpha,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      beta,
      tensor_c.device_ref(),
      tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());
  ORT_ENFORCE(passed, "Gemm kernel result wrong!");
}

template void run_blkq4_gemm<16, true, false, true>(int m, int n, int k);
template void run_blkq4_gemm<16, true, false, false>(int m, int n, int k);
template void run_blkq4_gemm<32, true, false, true>(int m, int n, int k);
template void run_blkq4_gemm<32, true, false, false>(int m, int n, int k);
template void run_blkq4_gemm<64, true, false, true>(int m, int n, int k);
template void run_blkq4_gemm<64, true, false, false>(int m, int n, int k);
template void run_blkq4_gemm<16, false, false, true>(int m, int n, int k);
template void run_blkq4_gemm<16, false, false, false>(int m, int n, int k);
template void run_blkq4_gemm<32, false, false, true>(int m, int n, int k);
template void run_blkq4_gemm<32, false, false, false>(int m, int n, int k);
template void run_blkq4_gemm<64, false, false, true>(int m, int n, int k);
template void run_blkq4_gemm<64, false, false, false>(int m, int n, int k);
template void run_blkq4_gemm<16, true, true, true>(int m, int n, int k);
template void run_blkq4_gemm<16, true, true, false>(int m, int n, int k);
template void run_blkq4_gemm<32, true, true, true>(int m, int n, int k);
template void run_blkq4_gemm<32, true, true, false>(int m, int n, int k);
template void run_blkq4_gemm<64, true, true, true>(int m, int n, int k);
template void run_blkq4_gemm<64, true, true, false>(int m, int n, int k);
template void run_blkq4_gemm<16, false, true, true>(int m, int n, int k);
template void run_blkq4_gemm<16, false, true, false>(int m, int n, int k);
template void run_blkq4_gemm<32, false, true, true>(int m, int n, int k);
template void run_blkq4_gemm<32, false, true, false>(int m, int n, int k);
template void run_blkq4_gemm<64, false, true, true>(int m, int n, int k);
template void run_blkq4_gemm<64, false, true, false>(int m, int n, int k);

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
