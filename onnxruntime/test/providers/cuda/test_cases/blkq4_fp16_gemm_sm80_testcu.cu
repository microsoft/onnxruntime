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

// This test has build error with cuda 12.5
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if defined(CUDA_VERSION) && CUDA_VERSION <= 12030

#include "blkq4_fp16_gemm_sm80.h"

#include "core/mickey/blk_q4/f16_gemm_sm80.h"
#include "core/mickey/gemm/device/quant_b4_gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {
namespace test {

Status sm80_supported() {
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
    typename AccumulatorType>
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
      (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn));

  // Launch a GEMM kernel
  cutlass::reference::device::kernel::Gemm<
      cutlass::TensorRef<ElementA, LayoutA>,
      cutlass::TensorRef<ElementB, LayoutB>,
      cutlass::TensorRef<ElementC, LayoutC>,
      ScalarType,
      AccumulatorType,
      OutputTile,
      cutlass::multiply_add<AccumulatorType>,
      cutlass::NumericConverter<ElementC, ScalarType>><<<grid, block, 0, 0>>>(
      problem_size,
      alpha,
      tensor_a,
      tensor_b,
      beta,
      tensor_c,
      tensor_d,
      initial_accum);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Converting cutlass tensor to MatrixRef
//

template <
    typename Element,
    typename LayoutCutlass,
    typename Layout = std::conditional_t<std::is_same<LayoutCutlass,
                                                      cutlass::layout::ColumnMajor>::value,
                                         ColumnMajorLayout, RowMajorLayout>>
__forceinline__
    MatrixRef<Element, Layout, true>
    make_MatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value ||
                std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  auto* ptr = const_cast<typename std::remove_const<Element>::type*>(tensor.host_data());
  return MatrixRef<Element, Layout, true>(ptr, tensor.capacity(), shape);
}

template <
    typename Element,
    typename LayoutCutlass,
    typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value,
                                         ColumnMajorLayout, RowMajorLayout>>
__forceinline__
    MatrixRef<Element const, Layout, true>
    make_ConstMatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value ||
                std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  return MatrixRef<Element const, Layout, true>(tensor.host_data(), tensor.capacity(), shape);
}

//
// Invoking the kernel
//

template <
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k) {
  unsigned int seed = 28571;  // Replace with desired seed value
  std::seed_seq seq{seed};
  std::mt19937 gen(seq);
  std::uniform_int_distribution<> dis(0, 8192);

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
  const auto q_weight_shape = cutlass::make_Coord(problem_size.k() / 2, problem_size.n());
  const auto meta_shape = cutlass::make_Coord(problem_size.k() / QuantBlocking::kRow, problem_size.n() /
                                                                                          QuantBlocking::kColumn);

  //
  // Generate quantized and dequantizeed input matrix B [K, N]
  //
  static_assert(std::is_same<LayoutInputWPack, cutlass::layout::ColumnMajor>::value);
  thrust::host_vector<ElementW> q_weights;
  thrust::host_vector<ElementQScale> q_scales;
  thrust::host_vector<ElementQOffset> q_zp;
  thrust::host_vector<ElementDequant> dequants;
  onnxruntime::cuda::test::blkq4_weights_gen<ElementDequant, block_size, column_wise_blocking, has_offsets>(
      problem_size.k(), problem_size.n(), dequants, q_weights, q_scales, q_zp);

  using PrepackT = onnxruntime::cuda::BlockwiseQuantization<
      ElementDequant,
      block_size,
      4,
      column_wise_blocking>;

  thrust::host_vector<ElementW> packed_w(q_weight_shape.product());
  PrepackT::prepack_weights(problem_size.k(), problem_size.n(), q_weights, packed_w);
  thrust::host_vector<ElementQScale> packed_scales(meta_shape.product());
  PrepackT::prepack_quant_scales(problem_size.k(), problem_size.n(), q_scales, packed_scales);
  thrust::host_vector<ElementQOffset> packed_zp;
  if constexpr (has_offsets) {
    packed_zp.resize(meta_shape.product());
    PrepackT::prepack_quant_offsets(problem_size.k(), problem_size.n(), q_zp, packed_zp);
  }

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
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
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

  //
  // Copy data from host to GPU...
  //
  thrust::device_vector<ElementW> d_packed_w(packed_w);
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
      reinterpret_cast<ElementWPack const*>(d_packed_w.data().get()),
      LayoutInputWPack::packed({problem_size.k() / 2, problem_size.n() / 2}));

  thrust::device_vector<ElementQScale> d_packed_scales(packed_scales);
  cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_scales(
      d_packed_scales.data().get(), LayoutInputQScale::packed(meta_shape));

  thrust::device_vector<ElementQOffset> d_packed_zp(packed_zp);
  cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_zp(
      d_packed_zp.data().get(), LayoutInputQScale::packed(meta_shape));

  tensor_a.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets) {
    status = GemmRunner::run(
        nullptr, problem_size, tensor_a.device_ref(), ref_W,
        ref_scales, ref_zp,
        tensor_c.device_ref(), tensor_d.device_ref());
  } else {
    status = GemmRunner::run(
        nullptr, problem_size, tensor_a.device_ref(), ref_W,
        ref_scales,
        tensor_c.device_ref(), tensor_d.device_ref());
  }
  ORT_ENFORCE(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));

  // Running reference kernel
  using ElementInputB = ElementInputA;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  thrust::device_vector<ElementInputB> d_dequants(dequants);
  cutlass::TensorRef<ElementInputB, LayoutInputB> ref_B(
      d_dequants.data().get(), LayoutInputB::packed(problem_size.kn()));
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
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
      ref_B,
      beta,
      tensor_c.device_ref(),
      tensor_ref_d.device_ref());

  //// Wait for kernels to finish
  cudaDeviceSynchronize();

  //// Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  //// Check if output from CUTLASS kernel and reference kernel are equal or not
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



/// @brief Testing small tile GEMM impl
template <
    int block_size,
    bool column_wise_blocking,
    bool has_offsets>
void run_blkq4_small_gemm(int m, int n, int k) {
  unsigned int seed = 28571;  // Replace with desired seed value
  std::seed_seq seq{seed};
  std::mt19937 gen(seq);
  std::uniform_int_distribution<> dis(0, 8192);

  using PrepackT = onnxruntime::cuda::BlockwiseQuantization<
      cutlass::half_t,
      block_size,
      4,
      column_wise_blocking,
      true>;
  using QuantBlocking = cutlass::MatrixShape<PrepackT::QuantBlocking::kRow, PrepackT::QuantBlocking::kColumn>;
  using LayoutQmeta = typename std::conditional<std::is_same<typename PrepackT::LayoutQmeta, RowMajorLayout>::value,
                                            cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

  using WarpShape = cutlass::gemm::GemmShape<16, 16, 64>;
  // change split k to 1 to help debug in case of test failure
  using GemmRunner = mickey::gemm::device::QuantB4Gemm<QuantBlocking, has_offsets, WarpShape, 4, 3>;

  using ElementW = uint8_t;
  using ElementWPack = cutlass::half_t;
  using ElementQScale = cutlass::half_t;
  using ElementQOffset = uint8_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};
  const auto q_weight_shape = cutlass::make_Coord(problem_size.k() / 2, problem_size.n());
  const auto meta_shape = cutlass::make_Coord(problem_size.k() / QuantBlocking::kRow,
                                              problem_size.n() / QuantBlocking::kColumn);
  if ((problem_size.k() % QuantBlocking::kRow != 0) ||
    (problem_size.n() % QuantBlocking::kColumn) != 0){
    ORT_THROW("Test case setup fail: partial quantization block not supported!");
  }

  //
  // Generate quantized and dequantizeed input matrix B [K, N]
  //
  thrust::host_vector<ElementW> q_weights;
  thrust::host_vector<ElementQScale> q_scales;
  thrust::host_vector<ElementQOffset> q_zp;
  thrust::host_vector<cutlass::half_t> dequants;
  onnxruntime::cuda::test::blkq4_weights_gen<cutlass::half_t, block_size, column_wise_blocking, has_offsets>(
      problem_size.k(), problem_size.n(), dequants, q_weights, q_scales, q_zp);

  thrust::host_vector<ElementW> packed_w(q_weight_shape.product());
  PrepackT::prepack_weights(problem_size.k(), problem_size.n(), q_weights, packed_w);
  thrust::host_vector<ElementQScale> packed_scales(meta_shape.product());
  PrepackT::prepack_quant_scales(problem_size.k(), problem_size.n(), q_scales, packed_scales);
  thrust::host_vector<ElementQOffset> packed_zp;
  if constexpr (has_offsets) {
    packed_zp.resize(meta_shape.product());
    PrepackT::prepack_quant_offsets(problem_size.k(), problem_size.n(), q_zp, packed_zp);
  }

  cutlass::HostTensor<cutlass::half_t, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<cutlass::half_t, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<cutlass::half_t, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      cutlass::half_t(1.25),
      cutlass::half_t(-1.0),
      5);  // <- Fill matrix A on host with uniform-distribution random data
//   std::cout << "==========  A:  ============ \n" << tensor_a.host_view() << std::endl;
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      cutlass::half_t(1.25),
      cutlass::half_t(-1.0),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

  //
  // Copy data from host to GPU...
  //
  thrust::device_vector<ElementW> d_packed_w(packed_w);
  cutlass::TensorRef<ElementW const, cutlass::layout::ColumnMajor> ref_W(
      d_packed_w.data().get(),
      cutlass::layout::ColumnMajor::packed({problem_size.k(), problem_size.n() / 2}));

  thrust::device_vector<ElementQScale> d_packed_scales(packed_scales);
  cutlass::TensorRef<ElementQScale const, LayoutQmeta> ref_scales(
      d_packed_scales.data().get(), LayoutQmeta::packed(meta_shape));

  thrust::device_vector<ElementQOffset> d_packed_zp(packed_zp);
  cutlass::TensorRef<ElementQOffset const, LayoutQmeta> ref_zp(
      d_packed_zp.data().get(), LayoutQmeta::packed(meta_shape));

  tensor_a.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  // run GEMM
  const void* ptr_zp = has_offsets ? thrust::raw_pointer_cast(d_packed_zp.data()) : nullptr;
  size_t zp_byte_stride = has_offsets ? ref_zp.stride(0) * sizeof(ElementQOffset) : size_t(0);
  cutlass::Status status = GemmRunner::run(
      nullptr, problem_size,
      tensor_d.device_data(), tensor_d.stride(0) * sizeof(cutlass::half_t),
      tensor_a.device_data(), tensor_a.stride(0) * sizeof(cutlass::half_t),
      thrust::raw_pointer_cast(d_packed_w.data()), problem_size.k() * sizeof(ElementW),
      thrust::raw_pointer_cast(d_packed_scales.data()), ref_scales.stride(0) * sizeof(ElementQScale),
      ptr_zp, zp_byte_stride);
  ORT_ENFORCE(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));

  // Running reference kernel
  thrust::device_vector<cutlass::half_t> d_dequants(dequants);
  cutlass::TensorRef<cutlass::half_t, LayoutInputB> ref_B(
      d_dequants.data().get(), LayoutInputB::packed(problem_size.kn()));
  cutlass::HostTensor<cutlass::half_t, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  float alpha = 1.0f;
  float beta = 0.0f;

  compute_gemm_ref<cutlass::half_t, LayoutInputA,
                   cutlass::half_t, LayoutInputB,
                   cutlass::half_t, LayoutOutput,
                   float, float>(
      problem_size,
      alpha,
      tensor_a.device_ref(),
      ref_B,
      beta,
      tensor_c.device_ref(),
      tensor_ref_d.device_ref());

  //// Wait for kernels to finish
  cudaDeviceSynchronize();

  //// Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  //// Check if output from CUTLASS kernel and reference kernel are equal or not
  for (int row = 0; row < problem_size.m(); ++row) {
    for (int col = 0; col < problem_size.n(); ++col) {
      if (tensor_d.at({row, col}) != tensor_ref_d.at({row, col})) {
        std::cout << "Mismatch at (" << row << ", " << col << "): "
                  << tensor_d.at({row, col}) << " vs " << tensor_ref_d.at({row, col}) << std::endl;
      }
    }
  }
  bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());
  ORT_ENFORCE(passed, "Gemm kernel result wrong!");
}

template void run_blkq4_small_gemm<16, true, true>(int m, int n, int k);
template void run_blkq4_small_gemm<16, true, false>(int m, int n, int k);
template void run_blkq4_small_gemm<32, true, true>(int m, int n, int k);
template void run_blkq4_small_gemm<32, true, false>(int m, int n, int k);
template void run_blkq4_small_gemm<64, true, true>(int m, int n, int k);
template void run_blkq4_small_gemm<64, true, false>(int m, int n, int k);
template void run_blkq4_small_gemm<128, true, true>(int m, int n, int k);
template void run_blkq4_small_gemm<128, true, false>(int m, int n, int k);
template void run_blkq4_small_gemm<16, false, true>(int m, int n, int k);
template void run_blkq4_small_gemm<16, false, false>(int m, int n, int k);
template void run_blkq4_small_gemm<32, false, true>(int m, int n, int k);
template void run_blkq4_small_gemm<32, false, false>(int m, int n, int k);
template void run_blkq4_small_gemm<64, false, true>(int m, int n, int k);
template void run_blkq4_small_gemm<64, false, false>(int m, int n, int k);
template void run_blkq4_small_gemm<128, false, true>(int m, int n, int k);
template void run_blkq4_small_gemm<128, false, false>(int m, int n, int k);


}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime

#endif
