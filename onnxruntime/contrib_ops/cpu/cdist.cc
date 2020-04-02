// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cdist.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {
#define DEFINE_KERNEL(data_type)                                                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(CDist, kMSDomain, 1, data_type, kCpuExecutionProvider,                            \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
                                CDist<data_type>);
DEFINE_KERNEL(float);
DEFINE_KERNEL(double);

template <typename T>
static void CalculateSqeuclidean(const Tensor& a, const Tensor& b, Tensor& c, concurrency::ThreadPool* threadpool) {
  // input shapes have already been validated
  const auto& shape_a = a.Shape().GetDims();  // {m, k}
  const auto& shape_b = b.Shape().GetDims();  // {n, k}
  int64_t m = shape_a[0];
  int64_t n = shape_b[0];
  int64_t k = shape_a[1];

  // https://github.com/droyed/eucl_dist/wiki/Main-Article
  // dist(Xi,Yj) = sum_k(Xik**2) + sum_k(Yjk**2) - 2*sum_k(Xik*Yjk)

  const auto* a_data = a.Data<T>();
  const auto* b_data = b.Data<T>();
  auto* c_data = c.MutableData<T>();

  // ReduceSumSquare for A
  std::vector<T> a_ss;
  a_ss.reserve(m);
  const auto* cur_a = a_data;
  for (int64_t i = 0; i < m; ++i) {
    a_ss.push_back(ConstEigenVectorMap<T>(cur_a, k).squaredNorm());
    cur_a += k;
  }

  // ReduceSumSquare for B
  std::vector<T> b_ss;
  b_ss.reserve(n);
  const auto* cur_b = b_data;
  for (int64_t i = 0; i < n; ++i) {
    b_ss.push_back(ConstEigenVectorMap<T>(cur_b, k).squaredNorm());
    cur_b += k;
  }

  // add a_ss and b_ss with broadcast
  // output shape is {m, n}
  auto* cur_out = c_data;
  for (int64_t i = 0; i < m; ++i) {
    T a_val = a_ss[i];
    for (int64_t j = 0; j < n; ++j) {
      *cur_out++ = a_val + b_ss[j];
    }
  }

  // Now use GEMM of A and B^T with -2 as alpha to calculate -2*sum_k(Xik*Yjk)
  // This is added to the data already in c (as beta is 1.f), achieving our end result
  math::Gemm<T>(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                m, n, k,
                -2.f, a_data, b_data, 1.f,
                c_data,
                threadpool);
}

template <typename T>
common::Status CDist<T>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  assert(context->InputCount() == 2);
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = context->Input<Tensor>(1);
  const TensorShape& shape_a = A->Shape();
  const TensorShape& shape_b = B->Shape();
  if (shape_a.NumDimensions() != 2 || shape_a[1] <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The first input of CDist kernel has wrong shape: ", shape_a);
  }

  if (shape_b.NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The second input of CDist kernel has wrong shape: ", shape_b);
  }
  if (shape_a[1] != shape_b[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input shape dimensions mismatch:", shape_a, " and ", shape_b);
  }

  TensorShape output_shape = {shape_a[0], shape_b[0]};
  Tensor* C = context->Output(0, output_shape);
  T* output = C->MutableData<T>();

  CalculateSqeuclidean<T>(*A, *B, *C, tp);

  if (mode_ == Mode::EUCLIDEAN) {
    auto map_out = EigenVectorArrayMap<T>(output, output_shape.Size());
    // because we use GEMM in CalculateSqeuclidean there's a slight chance a number extremely close to zero
    // could be negative, so we need to run abs() to avoid NaN's in the results.
    map_out = map_out.abs().sqrt();
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
