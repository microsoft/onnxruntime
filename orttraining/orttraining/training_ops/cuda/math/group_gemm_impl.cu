// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "orttraining/training_ops/cuda/math/group_gemm_impl.h"
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

// template <bool UseGroupGemm, typename CudaT>
// void GenerateLdaLdbLdcLdd(const std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
//                           gsl::span<int64_t> lda_span,
//                           gsl::span<int64_t> ldb_span,
//                           gsl::span<int64_t> ldc_span,
//                           gsl::span<int64_t> ldd_span,
//                           CudaAsyncBufferForCutlass<cutlass::gemm::GemmCoord>& problem_sizes_device,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<int64_t>& lda,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<int64_t>& lda,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<int64_t>& ldc,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<int64_t>& ldd,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<CudaT*>& data_ptr_a,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<CudaT*>& data_ptr_b,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<CudaT*>& data_ptr_c,
//                           onnxruntime::cuda ::CudaKernel::CudaAsyncBuffer<CudaT*>& data_ptr_d,
//                           onnxruntime::Stream* stream) {
//   using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
//   if (UseGroupGemm) {
//     using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
//   } else {
//     using Gemm_ = typename onnxruntime::contrib::cuda::GemmGrouped_Half_DeviceOnly;
//   }

//   using LayoutA = typename Gemm_::LayoutA;
//   using LayoutB = typename Gemm_::LayoutB;
//   using LayoutC = typename Gemm_::LayoutC;
//   for (size_t i = 0; i < problem_sizes.size(); ++i) {
//     auto& problem = problem_sizes[i];
//     // problem_sizes_span[i] = cutlass::gemm::GemmCoord problem(problem.m, problem.n, problem.k);
//     lda_span[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
//     ldb_span[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
//     ldc_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
//     ldd_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
//   }

//   ORT_ENFORCE(problem_sizes_device.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(lda.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(ldb.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(ldc.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(ldd.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(data_ptr_a.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(data_ptr_b.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(data_ptr_c.CopyToGpu(stream).IsOK());
//   ORT_ENFORCE(data_ptr_d.CopyToGpu(stream).IsOK());
// }

// To support cudaMemcpyAsync, the cpu memory should be allocated in pinned memory
// and it can only be released after the copy has finished
template <typename T>
class CudaAsyncBufferForCutlass {
 public:
  CudaAsyncBufferForCutlass(const CudaKernel* op_kernel) : count_(0), op_kernel_(op_kernel) {}

  CudaAsyncBufferForCutlass(const CudaKernel* op_kernel, size_t count) : CudaAsyncBufferForCutlass(op_kernel) {
    AllocCpuPtr(count);
  }

  // CudaAsyncBufferForCutlass(const CudaKernel* op_kernel, const T& value, size_t count)
  //     : CudaAsyncBufferForCutlass(op_kernel, count) {
  //   T* p = CpuPtr();
  //   for (size_t i = 0; i != count; ++i) {
  //     *p++ = value;
  //   }
  // }

  CudaAsyncBufferForCutlass(const CudaKernel* op_kernel, gsl::span<T const> vec) : CudaAsyncBufferForCutlass(op_kernel, vec.size()) {
    memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
  }

  void AllocCpuPtr(size_t count) {
    cpu_pinned_copy_ = *(op_kernel_->AllocateBufferOnCPUPinned<T>(count).release());
    // if (cpu_pinned_copy_ == nullptr)
    //   throw std::runtime_error("alloc failed");
    count_ = count;
  }

  Status CopyToGpu(onnxruntime::Stream* stream) {
    // if (cpu_pinned_copy_) {
    gpu_copy_ = *(op_kernel_->GetScratchBuffer<T>(count_, stream).release());
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&gpu_copy_, &cpu_pinned_copy_, count_ * sizeof(T), cudaMemcpyHostToDevice,
                                         cuda_stream));
    op_kernel_->AddDeferredReleaseCPUPtr(&cpu_pinned_copy_, stream);
    // }
    return Status::OK();
  }

  T* CpuPtr() {
    return &cpu_pinned_copy_;
  }

  gsl::span<T> CpuSpan() {
    return gsl::span<T>(CpuPtr(), count_);
  }

  T* GpuPtr() {
    return &gpu_copy_;
  }

  size_t count() const {
    return count_;
  }

 protected:
  T gpu_copy_;
  T cpu_pinned_copy_;
  size_t count_;
  const CudaKernel* op_kernel_;
};

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
    std::vector<std::tuple<int64_t, int64_t, int64_t>>& problem_sizes,
    int64_t problem_count,
    std::vector<const T*> data_ptr_a_vec,
    std::vector<const T*> data_ptr_b_vec,
    std::vector<T*> data_ptr_c_vec,
    std::vector<T*> data_ptr_d_vec) {
  // Prepare Gemm problem sizes
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_vec(problem_count);
  CudaAsyncBufferForCutlass<cutlass::gemm::GemmCoord> problem_sizes_device(kernel, problem_count);
  CudaAsyncBufferForCutlass<int64_t> lda(kernel, problem_count);
  CudaAsyncBufferForCutlass<int64_t> ldb(kernel, problem_count);
  CudaAsyncBufferForCutlass<int64_t> ldc(kernel, problem_count);
  CudaAsyncBufferForCutlass<int64_t> ldd(kernel, problem_count);

  CudaAsyncBufferForCutlass<const T*> data_ptr_a(kernel, problem_count);
  CudaAsyncBufferForCutlass<const T*> data_ptr_b(kernel, problem_count);
  CudaAsyncBufferForCutlass<T*> data_ptr_c(kernel, problem_count);
  CudaAsyncBufferForCutlass<T*> data_ptr_d(kernel, problem_count);

  gsl::span<cutlass::gemm::GemmCoord> problem_sizes_span = problem_sizes_device.CpuSpan();
  gsl::span<int64_t> lda_span = lda.CpuSpan();
  gsl::span<int64_t> ldb_span = ldb.CpuSpan();
  gsl::span<int64_t> ldc_span = ldc.CpuSpan();
  gsl::span<int64_t> ldd_span = ldd.CpuSpan();

  gsl::span<const T*> data_ptr_a_span = data_ptr_a.CpuSpan();
  gsl::span<const T*> data_ptr_b_span = data_ptr_b.CpuSpan();
  gsl::span<T*> data_ptr_c_span = data_ptr_c.CpuSpan();
  gsl::span<T*> data_ptr_d_span = data_ptr_d.CpuSpan();

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
    int m, n, k;
    std::tie(m, n, k) = problem_sizes[i];
    problem_sizes_vec[i] = cutlass::gemm::GemmCoord(m, n, k);
    problem_sizes_span[i] = problem_sizes_vec[i];
    lda_span[i] = LayoutA::packed({m, k}).stride(0);
    ldb_span[i] = LayoutB::packed({k, n}).stride(0);
    ldc_span[i] = LayoutC::packed({m, n}).stride(0);
    ldd_span[i] = LayoutC::packed({m, n}).stride(0);
  }

  for (size_t i = 0; i < problem_count; ++i) {
    data_ptr_a_span[i] = data_ptr_a_vec[i];
    data_ptr_b_span[i] = data_ptr_b_vec[i];
    data_ptr_c_span[i] = data_ptr_c_vec[i];
    data_ptr_d_span[i] = data_ptr_d_vec[i];
  }

  ORT_ENFORCE(problem_sizes_device.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(lda.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(ldb.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(ldc.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(ldd.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(data_ptr_a.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(data_ptr_b.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(data_ptr_c.CopyToGpu(stream).IsOK());
  ORT_ENFORCE(data_ptr_d.CopyToGpu(stream).IsOK());

  cutlass::gemm::GemmCoord* problem_sizes_gpu_ptr = problem_sizes_device.GpuPtr();
  int64_t* lda_gpu_ptr = lda.GpuPtr();
  int64_t* ldb_gpu_ptr = ldb.GpuPtr();
  int64_t* ldc_gpu_ptr = ldc.GpuPtr();
  int64_t* ldd_gpu_ptr = ldd.GpuPtr();
  T** data_ptr_a_gpu_ptr = const_cast<T**>(data_ptr_a.GpuPtr());
  T** data_ptr_b_gpu_ptr = const_cast<T**>(data_ptr_b.GpuPtr());
  T** data_ptr_c_gpu_ptr = data_ptr_c.GpuPtr();
  T** data_ptr_d_gpu_ptr = data_ptr_d.GpuPtr();

  // Launch Group Gemm
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

  int threadblock_count = Gemm_::sufficient(problem_sizes_vec.data(), problem_sizes_vec.size());

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
      problem_sizes_gpu_ptr,
      static_cast<int>(problem_count),
      threadblock_count, epilogue_op,
      reinterpret_cast<CutlassType>(data_ptr_a_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_b_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_c_gpu_ptr),
      reinterpret_cast<CutlassType>(data_ptr_d_gpu_ptr),
      lda_gpu_ptr, ldb_gpu_ptr, ldc_gpu_ptr, ldd_gpu_ptr,
      problem_sizes_vec.data());

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

#define SPECIALIZE_GROUP_GEMM_IMPL(T, G, M)                              \
  template Status GroupGemm_Impl<T, G, M>(                               \
      const CudaKernel* kernel,                                          \
      Stream* stream,                                                    \
      std::vector<std::tuple<int64_t, int64_t, int64_t>>& problem_sizes, \
      int64_t problem_count,                                             \
      std::vector<const T*> data_ptr_a_vec,                              \
      std::vector<const T*> data_ptr_b_vec,                              \
      std::vector<T*> data_ptr_c_vec,                                    \
      std::vector<T*> data_ptr_d_vec);

SPECIALIZE_GROUP_GEMM_IMPL(half, true, true);

// SPECIALIZE_GROUP_GEMM_IMPL(float, GemmGrouped, GroupScheduleMode::kDeviceOnly)

#undef SPECIALIZE_GROUP_GEMM_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
