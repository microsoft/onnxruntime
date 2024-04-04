// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_config.h>
#include "core/providers/cpu/math/gemm.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
#include "core/util/math_cpuonly.h"
#include "gemm_helper.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    7,
    8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    7,
    8,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Gemm<double>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    7,
    8,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Gemm<MLFloat16>);

// opset 9 added support for additional types (int32, uint32, int64, uint64), however we haven't enabled those yet.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    9,
    10,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    9,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Gemm<double>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    9,
    10,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Gemm<MLFloat16>);

// opset 11 made bias input 'C' optional
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    11,
    12,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Gemm<double>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Gemm,
    11,
    12,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Gemm<MLFloat16>);

// opset 13 Adds BFloat16 support but we are not supporting it yet
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Gemm,
    13,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Gemm,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Gemm<double>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Gemm,
    13,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Gemm<MLFloat16>);

bool GemmPackBFp32(AllocatorPtr& alloc,
                   const Tensor& tensor_b,
                   bool trans_b,
                   IAllocatorUniquePtr<void>& packed_b,
                   size_t& packed_b_size,
                   TensorShape& b_shape) {
  // Only handle the common case of a 2D weight matrix. Additional matrices
  // could be handled by stacking the packed buffers.
  if (tensor_b.Shape().NumDimensions() != 2) {
    return false;
  }
  b_shape = tensor_b.Shape();

  const size_t K = trans_b ? static_cast<size_t>(b_shape[1]) : static_cast<size_t>(b_shape[0]);
  const size_t N = trans_b ? static_cast<size_t>(b_shape[0]) : static_cast<size_t>(b_shape[1]);

  packed_b_size = MlasGemmPackBSize(N, K);
  if (packed_b_size == 0) {
    return false;
  }

  packed_b = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
  auto* packed_b_data = packed_b.get();

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_b_data, 0, packed_b_size);

  MlasGemmPackB(trans_b ? CblasTrans : CblasNoTrans,
                N,
                K,
                tensor_b.Data<float>(),
                trans_b ? K : N,
                packed_b_data);
  return true;
}

template <typename T>
void Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
                          T alpha,
                          const T* a_data, const T* b_data,
                          T beta,
                          const T* c_data, const TensorShape* c_shape,
                          T* y_data,
                          concurrency::ThreadPool* thread_pool) {
  // if input is empty tensor, return directly as nothing need to be calculated.
  if (M == 0 || N == 0)
    return;

  // Broadcast the bias as needed if bias is given
  GemmBroadcastBias(M, N, beta, c_data, c_shape, y_data);

  math::Gemm<T>(trans_a, trans_b,
                M, N, K,
                alpha,
                a_data,
                b_data,
                // ideally we need to set the output buffer contents to 0 if bias is missing,
                // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
                c_data != nullptr ? beta : 0,
                y_data,
                thread_pool);
}

template <>
void Gemm<MLFloat16>::ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                                  ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
                                  MLFloat16 alpha,
                                  const MLFloat16* a_data, const MLFloat16* b_data,
                                  MLFloat16 beta,
                                  const MLFloat16* c_data, const TensorShape* c_shape,
                                  MLFloat16* y_data,
                                  concurrency::ThreadPool* thread_pool) {
  // if input is empty tensor, return directly as nothing need to be calculated.
  if (M == 0 || N == 0)
    return;

#if defined(__GNUC__) && defined(HAS_CLASS_MEMACCESS)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
  // MLFloat16's constructor is explicit, so here we need to use memset
  if (c_data == nullptr)
    memset(&beta, 0, sizeof(MLFloat16));
#if defined(__GNUC__) && defined(HAS_CLASS_MEMACCESS)
#pragma GCC diagnostic pop
#endif
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
  bool support_mlas = false;
  if (c_shape == nullptr) {
    support_mlas = true;
  } else if (c_shape->NumDimensions() == 1 && (*c_shape)[0] == N) {
    support_mlas = true;
  } else if (c_shape->NumDimensions() == 2 && (((*c_shape)[0] == 1 && (*c_shape)[1] == N) || ((*c_shape)[0] == N && (*c_shape)[1] == 1))) {
    support_mlas = true;
  }
  if (trans_a == CblasNoTrans && trans_b == CblasNoTrans && support_mlas && alpha.ToFloat() == 1.0 && beta.ToFloat() == 1.0) {
    MLAS_HALF_GEMM_DATA_PARAMS data;
    data.A = a_data;
    data.lda = K;
    data.B = b_data;
    data.ldb = N;
    data.C = y_data;
    data.ldc = N;
    if (c_shape != nullptr) {
      data.Bias = c_data;
    }
    MlasHalfGemmBatch(M, N, K, 1, &data, thread_pool);
    return;
  }
#endif
  // Fallback to Eigen
  // Broadcast the bias as needed if bias is given
  GemmBroadcastBias(M, N, beta, c_data, c_shape, y_data);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
  math::Gemm<Eigen::half>(trans_a, trans_b, M, N, K, *reinterpret_cast<Eigen::half*>(&alpha),
                          reinterpret_cast<const Eigen::half*>(a_data), reinterpret_cast<const Eigen::half*>(b_data), *reinterpret_cast<Eigen::half*>(&beta), reinterpret_cast<Eigen::half*>(y_data), thread_pool);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template void Gemm<float>::ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                                       ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
                                       float alpha,
                                       const float* a_data, const float* b_data,
                                       float beta,
                                       const float* c_data, const TensorShape* c_shape,
                                       float* y_data,
                                       concurrency::ThreadPool* thread_pool);

template <typename T>
Status Gemm<T>::PrePack(const Tensor& /* tensor */, int /* input_idx */, AllocatorPtr /*alloc_for_caching*/,
                        /*out*/ bool& is_packed,
                        /*out*/ PrePackedWeights* /*prepacked_weight_for_caching*/) {
  is_packed = false;
  return Status::OK();
}

template <>
Status Gemm<float>::PrePack(const Tensor& tensor, int input_idx,
                            AllocatorPtr alloc, /*out*/ bool& is_packed,
                            /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // only pack Matrix B
  if (input_idx == 1) {
    size_t packed_b_size;
    is_packed = GemmPackBFp32(alloc, tensor, trans_B_ != CblasNoTrans, packed_b_, packed_b_size, b_shape_);
    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (is_packed && share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size);
    }
  }
  return Status::OK();
}

template <typename T>
Status Gemm<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& /*prepacked_buffers*/,
                                          int /*input_idx*/,
                                          /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;
  return Status::OK();
}

template <>
Status Gemm<float>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                              int input_idx,
                                              /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  return Status::OK();
}

template <typename T>
void Gemm<T>::ComputeActivation(_Inout_updates_(y_size) T* y_data, ptrdiff_t y_size, _Inout_opt_ concurrency::ThreadPool* thread_pool) const {
  if (activation_) {
    std::unique_ptr<functors::ElementWiseRangedTransform<T>> f(activation_->Copy());
    f->input = y_data;
    f->output = y_data;
    double cost = f->Cost();
    functors::ElementWiseRangedTransform<T>* c(f.get());
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, y_size,
        {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), cost},
        [c](std::ptrdiff_t first, std::ptrdiff_t last) { (*c)(first, last); });
  }
}

template <typename T>
Status Gemm<T>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const auto* A = context->Input<Tensor>(0);
  const auto* B = context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_ != CblasNoTrans, B->Shape(), trans_B_ != CblasNoTrans,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  ptrdiff_t M = helper.M();
  ptrdiff_t N = helper.N();
  ptrdiff_t K = helper.K();

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  T* y_data = Y->MutableData<T>();
  const T* c_data = C != nullptr ? C->Data<T>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  ComputeGemm(trans_A_, trans_B_, M, N, K, alpha_, A->Data<T>(), B->Data<T>(), beta_,
              c_data, c_shape, y_data, thread_pool);

  ComputeActivation(y_data, SafeInt<ptrdiff_t>(M) * N, thread_pool);

  return Status::OK();
}

template <>
Status Gemm<MLFloat16>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const auto* A = context->Input<Tensor>(0);
  const auto* B = packed_b_ ? nullptr : context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_ != CblasNoTrans, B ? B->Shape() : b_shape_, trans_B_ != CblasNoTrans,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  ptrdiff_t M = helper.M();
  ptrdiff_t N = helper.N();
  ptrdiff_t K = helper.K();

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  MLFloat16* y_data = Y->MutableData<MLFloat16>();

  const MLFloat16* c_data = C != nullptr ? C->Data<MLFloat16>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  if (B) {
    ComputeGemm(trans_A_, trans_B_, M, N, K, static_cast<MLFloat16>(alpha_), A->Data<MLFloat16>(), B->Data<MLFloat16>(), static_cast<MLFloat16>(beta_),
                c_data, c_shape, y_data, thread_pool);
  } else {
    ORT_NOT_IMPLEMENTED("Prepacking of B is supported by MLAS half gemm API, but not implemented by this kernel yet");
  }

  ComputeActivation(y_data, SafeInt<size_t>(M) * N, thread_pool);

  return Status::OK();
}

template <>
Status Gemm<float>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const auto* A = context->Input<Tensor>(0);
  const auto* B = packed_b_ ? nullptr : context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_ != CblasNoTrans, B ? B->Shape() : b_shape_, trans_B_ != CblasNoTrans,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  ptrdiff_t M = helper.M();
  ptrdiff_t N = helper.N();
  ptrdiff_t K = helper.K();

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  float* y_data = Y->MutableData<float>();

  const float* c_data = C != nullptr ? C->Data<float>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  if (B) {
    ComputeGemm(trans_A_, trans_B_, M, N, K, alpha_, A->Data<float>(), B->Data<float>(), beta_,
                c_data, c_shape, y_data, thread_pool);
  } else {
    GemmBroadcastBias(M, N, beta_, c_data, c_shape, y_data);
    MlasGemm(
        trans_A_,
        static_cast<size_t>(M),
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        alpha_,
        A->Data<float>(),
        static_cast<size_t>(trans_A_ != CblasNoTrans ? M : K),
        packed_b_.get(),
        c_data != nullptr ? beta_ : 0.0f,
        y_data,
        static_cast<size_t>(N),
        thread_pool);
  }

  ComputeActivation(y_data, SafeInt<size_t>(M) * N, thread_pool);

  return Status::OK();
}

}  // namespace onnxruntime
