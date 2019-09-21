// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/util/distance.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/util/math_cpuonly.h"
#include "assert.h"
#ifndef USE_OPENMP
#include "core/util/eigen_common_wrapper.h"
#endif

namespace onnxruntime {
namespace contrib {

// https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
//\param a: matrix with shape of[ma,n]
//\param b: matrix with shape of[mb,n]
//\param dest: matrix with shape of [ma,mb]
template <typename T, typename ElemFunc>
void cdist_single_threaded(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n) {
  ElemFunc f;
  for (size_t i = 0; i != ma; ++i) {
    const T* a1 = a + n * i;
    for (size_t j = 0; j != mb; ++j) {
      const T* b1 = b + n * j;
      *dest++ = f(a1, b1, n);
    }
  }
}

template <typename T, typename ElemFunc>
class CDistOneBlock {
 public:
  const T* a;
  const T* b;
  T* dest;
  size_t mb;
  size_t n;

  CDistOneBlock(const T* a1, const T* b1, T* dest1, size_t mb1, size_t n1)
      : a(a1), b(b1), dest(dest1), mb(mb1), n(n1) {}

  void operator()(Eigen::Index start, Eigen::Index end) {
    Eigen::Index mb_local = mb;
    Eigen::Index i = start / mb_local;
    Eigen::Index j = start - i * mb_local;
    assert(i * mb_local + j == start);
    Eigen::Index i_end = end / mb_local;
    Eigen::Index j_end = end - i_end * mb_local;
    assert(i_end * mb_local + j_end == end);

    T* dest_local = dest + start;
    ElemFunc f;
    const T* a1 = a + n * i;
    for (; i != i_end; ++i) {
      a1 = a + n * i;
      for (; j != mb_local; ++j) {
        const T* b1 = b + n * j;
        *dest_local++ = f(a1, b1, n);
      }
      j = 0;
    }
    a1 = a + n * i;
    for (j = 0; j != j_end; ++j) {
      const T* b1 = b + n * j;
      *dest_local++ = f(a1, b1, n);
    }
    assert(dest_local == dest + end);
  }
};

template <typename T, typename ElemFunc>
void cdist(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n, concurrency::ThreadPool* tp) {
#ifndef USE_OPENMP
  if (tp == nullptr) {
#else
  (void)tp;
#endif
    return cdist_single_threaded<T, ElemFunc>(a, b, dest, ma, mb, n);
#ifndef USE_OPENMP
  }
  Eigen::ThreadPoolDevice device(&tp->GetHandler(), tp->NumThreads());
  device.parallelFor(ma * mb, Eigen::TensorOpCost(0, 0, static_cast<double>(3 * n)),
                     CDistOneBlock<T, ElemFunc>(a, b, dest, mb, n));
#endif
}

template <typename T>
class CDist final : public OpKernel {
 private:
  typedef void (*DistFunc)(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n,
                           concurrency::ThreadPool* tp);
  enum { EUCLIDEAN, SQEUCLIDEAN } mode_;

 public:
  CDist(const OpKernelInfo& info) : OpKernel(info) {
    std::string metric;
    ORT_ENFORCE(info.GetAttr<std::string>("metric", &metric).IsOK());
    if (metric.compare("sqeuclidean") == 0)
      mode_ = SQEUCLIDEAN;
    else if (metric.compare("euclidean") == 0) {
      mode_ = EUCLIDEAN;
    } else
      ORT_NOT_IMPLEMENTED();
  }

  common::Status Compute(OpKernelContext* context) const override {
    auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
    concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

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
    switch (mode_) {
      case EUCLIDEAN:
        if (shape_a[1] >= 8)
          cdist<T, EuclideanWithEigen<T> >(A->Data<T>(), B->Data<T>(), output, shape_a[0], shape_b[0], shape_a[1], tp);
        else  // for smaller vector size, a raw loop is better
          cdist<T, Euclidean<T> >(A->Data<T>(), B->Data<T>(), output, shape_a[0], shape_b[0], shape_a[1], tp);
        break;
      case SQEUCLIDEAN:
        if (shape_a[1] >= 8)
          cdist<T, SqeuclideanWithEigen<T> >(A->Data<T>(), B->Data<T>(), output, shape_a[0], shape_b[0], shape_a[1],
                                             tp);
        else  // for smaller vector size, a raw loop is better
          cdist<T, Sqeuclidean<T> >(A->Data<T>(), B->Data<T>(), output, shape_a[0], shape_b[0], shape_a[1], tp);
        break;
      default:
        return Status(ONNXRUNTIME, NOT_IMPLEMENTED);
    }
    return Status::OK();
  }
};
}  // namespace contrib
}  // namespace onnxruntime