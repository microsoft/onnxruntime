// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/math/group_gemm_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/common/gsl.h"
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define DEFINE_CUTLASS_GROUP_GEMM_TYPE(GROUP_GEMM_NAME,                                  \
                                       ElementA, ElementB, ElementC, ElementAccumulator, \
                                       LayoutA, LayoutB, LayoutC,                        \
                                       kTransformA, kTransformB,                         \
                                       kAlignmentA, kAlignmentB, OperatorClass,          \
                                       ThreadblockSwizzle, kStages, ArchTag,             \
                                       GroupScheduleMode)                                \
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;                       \
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;                                \
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;                            \
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<                 \
      ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,                             \
      ElementAccumulator, ElementAccumulator>;                                           \
  using GROUP_GEMM_NAME##_KERNEL = typename cutlass::gemm::kernel::DefaultGemmGrouped<   \
      ElementA, LayoutA, kTransformA, kAlignmentA, ElementB, LayoutB,                    \
      kTransformB, kAlignmentB, ElementC, LayoutC,                                       \
      ElementAccumulator, OperatorClass, ArchTag,                                        \
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,                   \
      ThreadblockSwizzle, kStages, GroupScheduleMode>::GemmKernel;                       \
                                                                                         \
  using GROUP_GEMM_NAME = cutlass::gemm::device::GemmGrouped<GROUP_GEMM_NAME##_KERNEL>;

//
// Define the Grouped GEMM types
//
// NOTE: Threadblock swizzling is currently not supported by CUTLASS's
// grouped kernels. This parameter is passed in at present to match the
// APIs of other kernels. The parameter is unused within the kernel.
constexpr static int kAlignmentA = 8;
constexpr static int kAlignmentB = 8;

DEFINE_CUTLASS_GROUP_GEMM_TYPE(
    GemmGrouped_Half_DeviceOnly,
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,
    kAlignmentA, kAlignmentB, cutlass::arch::OpClassTensorOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, cutlass::arch::Sm70,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly);

DEFINE_CUTLASS_GROUP_GEMM_TYPE(
    GemmGrouped_Half_HostPrecompute,
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,
    kAlignmentA, kAlignmentB, cutlass::arch::OpClassTensorOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, cutlass::arch::Sm70,
    cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute);

// using ElementA = cutlass::half_t;
// using ElementB = cutlass::half_t;
// using ElementC = cutlass::half_t;
// using ElementAccumulator = float;

// using LayoutA = cutlass::layout::ColumnMajor;
// using LayoutB = cutlass::layout::ColumnMajor;
// using LayoutC = cutlass::layout::ColumnMajor;

// // using kTransformA = cutlass::ComplexTransform::kNone;
// // using kTransformB = cutlass::ComplexTransform::kNone;

// // constexpr static int kAlignmentA = 8;
// // constexpr static int kAlignmentB = 8;

// using OperatorClass = cutlass::arch::OpClassTensorOp;

// // NOTE: Threadblock swizzling is currently not supported by CUTLASS's
// // grouped kernels. This parameter is passed in at present to match the
// // APIs of other kernels. The parameter is unused within the kernel.
// using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// constexpr static int kStages = 2;

// using ArchTag = cutlass::arch::Sm70;
// using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
// using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
// using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
// using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
//     ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
//     ElementAccumulator, ElementAccumulator>;

// // constexpr static int GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly;

// // Define a grouped GEMM kernel with all template parameters set except
// // for scheduling mode. This will be used as the template for all scheduling
// // modes executed.
// using GroupGemmKernel_Half = typename cutlass::gemm::kernel::DefaultGemmGrouped<
//     ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
//     ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
//     ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
//     ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
//     ThreadblockSwizzle, kStages>::GemmKernel;
// //, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly

// using GemmGrouped_Half_DeviceOnly = cutlass::gemm::device::GemmGrouped<GroupGemmKernel_Half>;

// // // Redefine GEMM with different GroupScheduleMode_
// // using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
// //     typename Gemm_::ElementA, typename Gemm_::LayoutA, Gemm_::kTransformA,
// //     Gemm_::kAlignmentA, typename Gemm_::ElementB, typename Gemm_::LayoutB,
// //     Gemm_::kTransformB, Gemm_::kAlignmentB, typename Gemm_::ElementC,
// //     typename Gemm_::LayoutC, typename Gemm_::ElementAccumulator,
// //     typename Gemm_::OperatorClass, typename Gemm_::ArchTag,
// //     typename Gemm_::ThreadblockShape, typename Gemm_::WarpShape,
// //     cutlass::gemm::GemmShape<8, 8, 4>, typename Gemm_::EpilogueOutputOp,
// //     typename Gemm_::ThreadblockSwizzle, Gemm_::kStages,
// //     GroupScheduleMode_>::GemmKernel;

// // using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

template <bool UseGroupGemm>
void GenerateLdaLdbLdcLdd(const std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                          gsl::span<int64_t> lda_span,
                          gsl::span<int64_t> ldb_span,
                          gsl::span<int64_t> ldc_span,
                          gsl::span<int64_t> ldd_span) {
  using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
  if (UseGroupGemm) {
    using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
  } else {
    using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
  }

  using LayoutA = typename Gemm_::LayoutA;
  using LayoutB = typename Gemm_::LayoutB;
  using LayoutC = typename Gemm_::LayoutC;
  for (size_t i = 0; i < problem_sizes.size(); ++i) {
    auto& problem = problem_sizes[i];
    // problem_sizes_span[i] = cutlass::gemm::GemmCoord problem(problem.m, problem.n, problem.k);
    lda_span[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_span[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    ldd_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
  }
}

// Type mapping for MLFloat16 to half
template <typename T>
class ToCutlassType {
 public:
  typedef T MappedType;
};

template <>
class ToCutlassType<half**> {
 public:
  typedef cutlass::half_t** MappedType;
};

template <typename T, bool UseGroupGemm, bool DeviceOnlyMode>
// template <typename T, typename Gemm_, cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_>
Status GroupGemm_Impl(
    const CudaKernel* kernel,
    Stream* stream,
    std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
    cutlass::gemm::GemmCoord* problem_sizes_gpu_ptr,
    int64_t problem_count,
    int64_t* lda_gpu_ptr,
    int64_t* ldb_gpu_ptr,
    int64_t* ldc_gpu_ptr,
    int64_t* ldd_gpu_ptr,
    T** data_ptr_a_gpu_ptr,
    T** data_ptr_b_gpu_ptr,
    T** data_ptr_c_gpu_ptr,
    T** data_ptr_d_gpu_ptr
    // ,
    // const T* left_x,
    // const T* right_x,
    // T* y
) {
  typedef typename ToCutlassType<T**>::MappedType CutlassType;

  using Gemm_ = GemmGrouped_Half_DeviceOnly;
  if (UseGroupGemm) {
    if (DeviceOnlyMode)
      using Gemm_ = GemmGrouped_Half_DeviceOnly;
    else
      using Gemm_ = GemmGrouped_Half_DeviceOnly;
  } else {
    using Gemm_ = GemmGrouped_Half_DeviceOnly;  // todo: change to GemmUniversal
  }

  // using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  int threadblock_count = Gemm_::sufficient(problem_sizes.data(), problem_sizes.size());

  // Early exit
  ORT_RETURN_IF_NOT(!threadblock_count, "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");

  float alpha = 1.0f;
  float beta = 0.0f;

  //
  // Initialize the CUTLASS GEMM operator
  //

  // Configure the GEMM arguments
  typename Gemm_::EpilogueOutputOp::Params epilogue_op(alpha, beta);

  // Configure GEMM arguments
  typename Gemm_::Arguments args(
      problem_sizes_gpu_ptr, static_cast<int>(problem_count),
      threadblock_count, epilogue_op,
      reinterpret_cast<CutlassType>(data_ptr_a_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_b_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_c_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_d_gpu_ptr),
      lda_gpu_ptr, ldb_gpu_ptr, ldc_gpu_ptr, ldd_gpu_ptr,
      problem_sizes.data());

  // Initialize the GEMM object
  Gemm_ gemm;
  size_t workspace_size = gemm.get_workspace_size(args);

  auto workspace = kernel->GetScratchBuffer<uint8_t>(workspace_size, stream);
  // cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

  auto status = gemm.initialize(args, workspace.get());
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS Grouped GEMM kernel.");

  // Run the grouped GEMM object
  status = gemm.run();
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess, "Failed to run CUTLASS Grouped GEMM kernel.");
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return Status::OK();

  // // Wait for completion
  // result.error = cudaDeviceSynchronize();

  // if (result.error != cudaSuccess) {
  //   std::cerr << "Kernel execution error: "
  //             << cudaGetErrorString(result.error);
  //   return result;
  // }

  //////////////////////////////////////
};

#define SPECIALIZE_GROUP_GEMM_IMPL(T, G, M)                 \
  template Status GroupGemm_Impl<T, G, M>(                  \
      const CudaKernel* kernel,                             \
      Stream* stream,                                       \
      std::vector<cutlass::gemm::GemmCoord>& problem_sizes, \
      cutlass::gemm::GemmCoord* problem_sizes_gpu_ptr,      \
      int64_t problem_count,                                \
      int64_t* lda_gpu_ptr,                                 \
      int64_t* ldb_gpu_ptr,                                 \
      int64_t* ldc_gpu_ptr,                                 \
      int64_t* ldd_gpu_ptr,                                 \
      T** data_ptr_a_span_gpu_ptr,                          \
      T** data_ptr_b_span_gpu_ptr,                          \
      T** data_ptr_c_span_gpu_ptr,                          \
      T** data_ptr_d_span_gpu_ptr);

SPECIALIZE_GROUP_GEMM_IMPL(half, true, true);

// SPECIALIZE_GROUP_GEMM_IMPL(float, GemmGrouped, GroupScheduleMode::kDeviceOnly)
template void GenerateLdaLdbLdcLdd<true>(const std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                                         gsl::span<int64_t> lda_span,
                                         gsl::span<int64_t> ldb_span,
                                         gsl::span<int64_t> ldc_span,
                                         gsl::span<int64_t> ldd_span);

#undef SPECIALIZE_GROUP_GEMM_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
