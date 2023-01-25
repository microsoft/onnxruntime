// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "gemm_gelu.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/activation.h"

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator followed by the GELU activation to an array of elements.
///
/// D = gelu(alpha * accumulator + beta * source + uniform)
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationGELUTaylor = LinearCombinationGeneric<GELU_taylor, ElementOutput_, Count, ElementAccumulator_,
                                                       ElementCompute_, Scale, Round, true>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;     // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

// Note that if the output is column major, the bias has to be per row. i.e. every row has different bias.
// If the output is row major, the bias has to be per column, i.e. every column has different bias.
// Below list some other notices:
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the
//      row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELUTaylor<
    ElementOutput,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;


#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GemmGelu<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)

using namespace ONNX_NAMESPACE;

template <typename T>
Status GemmGelu<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = context->Input<Tensor>(0);
  const Tensor* right_X = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);

  MatMulComputeHelper helper;
  // TODO: Handle transpose attributes
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(),
                                     right_X->Shape(),
                                     false, false, false, false, false));

  // TODO: Fix me
  if (helper.OutputOffsets().size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported");
  }

  Tensor* Y = context->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  const int lda = helper.Lda(false);
  const int ldb = helper.Ldb(false);
  const int ldc = helper.Ldc();

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulHelper(
      CublasLtHandle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      reinterpret_cast<const CudaT*>(right_X->Data<T>()),
      ldb,
      reinterpret_cast<const CudaT*>(left_X->Data<T>()),
      lda,
      &zero,
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      ldc,
      bias != nullptr
          ? reinterpret_cast<const CudaT*>(bias->Data<T>())
          : nullptr,
      true,
      NULL, 0,
      Stream(context)));

  return Status::OK();
}

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template <>
Status GemmGelu<MLFloat16>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;

  const Tensor* left_X = context->Input<Tensor>(0);
  const Tensor* right_X = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);

  MatMulComputeHelper helper;
  // TODO: Handle transpose attributes
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(),
                                     right_X->Shape(),
                                     false, false, false, false, false));

  // TODO: Fix me
  if (helper.OutputOffsets().size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported");
  }

  Tensor* Y = context->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  cutlass::gemm::GemmCoord problem_size(helper.N(), helper.M(), helper.K());
  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    // Split K dimension into 1 partitions
  int split_k_slices = 1;

  const int lda = helper.Lda(false);
  const int ldb = helper.Ldb(false);
  const int ldc = helper.Ldc();

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
    problem_size,                       // <- problem size of matrix multiplication
    {reinterpret_cast<const cutlass::half_t*>(right_X->Data<MLFloat16>()), ldb},              // <- reference to matrix A on device
    {reinterpret_cast<const cutlass::half_t*>(left_X->Data<MLFloat16>()), lda},              // <- reference to matrix B on device
    {bias != nullptr
          ? reinterpret_cast<const cutlass::half_t*>(bias->Data<MLFloat16>())
          : nullptr, 1},  // <- the C matrix is treated as the bias vector. We can enable the GEMM
                                        //    to project away the N dimension by setting the stride to zero.

    {reinterpret_cast<cutlass::half_t*>(Y->MutableData<MLFloat16>()), ldc},              // <- reference to matrix D on device
    {alpha},                              // <- alpha
    split_k_slices};                    // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  // Allocate workspace memory

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  void* workspace = allocator->Alloc(workspace_size);
  BufferUniquePtr temp_buffer(workspace, BufferDeleter(allocator));

   // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op(Stream(context));
  CUTLASS_CHECK(status);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
