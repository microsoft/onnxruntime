// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"
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

bool GemmPackBFp32(AllocatorPtr& alloc,
                   const Tensor& tensor_b,
                   bool trans_b,
                   BufferUniquePtr& packed_b,
                   TensorShape& b_shape) {
  // Only handle the common case of a 2D weight matrix. Additional matrices
  // could be handled by stacking the packed buffers.
  if (tensor_b.Shape().NumDimensions() != 2) {
    return false;
  }
  b_shape = tensor_b.Shape();

  const size_t K = trans_b ? static_cast<size_t>(b_shape[1]) : static_cast<size_t>(b_shape[0]);
  const size_t N = trans_b ? static_cast<size_t>(b_shape[0]) : static_cast<size_t>(b_shape[1]);

  const size_t packed_b_size = MlasGemmPackBSize(N, K);
  if (packed_b_size == 0) {
    return false;
  }

  auto* packed_b_data = alloc->Alloc(packed_b_size);
  packed_b = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
  MlasGemmPackB(trans_b ? CblasTrans : CblasNoTrans,
                N,
                K,
                tensor_b.Data<float>(),
                trans_b ? K : N,
                packed_b_data);
  return true;
}

template <typename T>
static void GemmBroadcastBias(int64_t M, int64_t N, float beta,
                              const T* c_data, const TensorShape* c_shape,
                              T* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<T>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<T>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<T>(c_data, M, N);
    }
  }
}

template <typename T>
void Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          int64_t M, int64_t N, int64_t K,
                          float alpha,
                          const T* a_data, const T* b_data,
                          float beta,
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

template void Gemm<float>::ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                                       int64_t M, int64_t N, int64_t K,
                                       float alpha,
                                       const float* a_data, const float* b_data,
                                       float beta,
                                       const float* c_data, const TensorShape* c_shape,
                                       float* y_data,
                                       concurrency::ThreadPool* thread_pool);

template <typename T>
Status Gemm<T>::PrePack(const Tensor& /* tensor */, int /* input_idx */, /*out*/ bool& is_packed,
                        /*out*/ PrepackedWeight* /*prepacked_weight_for_caching*/,
                        AllocatorPtr /*alloc_for_caching*/) {
  is_packed = false;
  return Status::OK();
}

template <>
Status Gemm<float>::PrePack(const Tensor& tensor, int input_idx, /*out*/ bool& is_packed,
                            /*out*/ PrepackedWeight* prepacked_weight_for_caching,
                            AllocatorPtr alloc) {
  is_packed = false;

  // only pack Matrix B
  if (input_idx == 1) {
    is_packed = GemmPackBFp32(alloc, tensor, trans_B_ != CblasNoTrans, packed_b_, b_shape_);
    bool kernel_owns_prepacked_buffer = (prepacked_weight_for_caching == nullptr);
    if (is_packed && !kernel_owns_prepacked_buffer) {
      prepacked_weight_for_caching->buffers_.push_back(std::move(packed_b_));
      prepacked_weight_for_caching->shapes_.push_back(b_shape_);
      prepacked_weight_for_caching->weights_sizes_.push_back(b_shape_.Size());
      prepacked_weight_for_caching->is_filled_ = true;
      packed_b_ = BufferUniquePtr(prepacked_weight_for_caching->buffers_[0].get(), BufferDeleter(nullptr));
    }
  }
  return Status::OK();
}

template <typename T>
Status Gemm<T>::UseCachedPrePackedWeight(const PrepackedWeight& /*cached_prepacked_weight*/,
                                         int /*input_idx*/,
                                         /*out*/ bool& read_from_cache) {
  read_from_cache = false;
  return Status::OK();
}

template <>
Status Gemm<float>::UseCachedPrePackedWeight(const PrepackedWeight& cached_prepacked_weight,
                                             int input_idx,
                                             /*out*/ bool& read_from_cache) {
  read_from_cache = false;

  if (input_idx == 1) {
    // Cached pre-packed weight
    read_from_cache = true;
    // This is a cached pre-packed buffer and this kernel doesn't own it and hence the deleter is null
    packed_b_ = BufferUniquePtr(cached_prepacked_weight.buffers_[0].get(), BufferDeleter(nullptr));
    b_shape_ = cached_prepacked_weight.shapes_[0];
  }
  return Status::OK();
}

template <typename T>
void Gemm<T>::ComputeActivation(T* y_data, size_t y_size, concurrency::ThreadPool* thread_pool) const {
  if (activation_) {
    std::unique_ptr<functors::ElementWiseRangedTransform<T>> f(activation_->Copy());
    f->input = y_data;
    f->output = y_data;
    std::ptrdiff_t total_len = static_cast<std::ptrdiff_t>(y_size);
    double cost = f->Cost();
    functors::ElementWiseRangedTransform<T>* c(f.get());
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, total_len,
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

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  T* y_data = Y->MutableData<T>();
  const T* c_data = C != nullptr ? C->Data<T>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  ComputeGemm(trans_A_, trans_B_, M, N, K, alpha_, A->Data<T>(), B->Data<T>(), beta_,
              c_data, c_shape, y_data, thread_pool);

  ComputeActivation(y_data, M * N, thread_pool);

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

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

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

  ComputeActivation(y_data, M * N, thread_pool);

  return Status::OK();
}

}  // namespace onnxruntime
