/**
 * Copyright (c) 2016-present, Facebook, Inc.
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
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/nn/lrn.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace functors {
template <typename T>
struct Powx {
  const T* input1 = nullptr;
  const T* input2 = nullptr;
  float b;
  T* output = nullptr;
  float Cost() const {
    // std::pow function is super costly
    return 320.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> a(this->input1 + first, len);
    ConstEigenVectorArrayMap<T> a2(this->input2 + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = a2 * a.pow(b);
  }
};
}  // namespace functors

template <>
Status LRN<float>::Compute(OpKernelContext* context) const {
  const auto& X = context->RequiredInput<Tensor>(0);
  const auto& X_shape = X.Shape();

  Tensor* Y = context->Output(0, X_shape);

  // Supports NCHW image format.

  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  const ptrdiff_t N = narrow<ptrdiff_t>(X_shape[0]);
  const ptrdiff_t C = narrow<ptrdiff_t>(X_shape[1]);
  const ptrdiff_t H = narrow<ptrdiff_t>(X_shape[2]);
  const ptrdiff_t W = narrow<ptrdiff_t>(X_shape[3]);

  const ptrdiff_t X_size = narrow<ptrdiff_t>(X_shape.Size());

  if (X_size == 0) {
    // Nothing to compute.
    return Status::OK();
  }

  // Note: `ptrdiff_t X_size` being set successfully implies that N*C*H*W will not overflow ptrdiff_t.

  const ptrdiff_t image_size = C * H * W;
  const ptrdiff_t pre_pad = (size_ - 1) / 2;
  const int H_times_W = SafeInt<int>(H) * W;  // H_times_W is passed to math::Axpy() which takes an int.

  const auto* Xdata = X.Data<float>();
  auto* Ydata = Y->MutableData<float>();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  void* sdata = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * X_size);
  BufferUniquePtr scale_buffer(sdata, BufferDeleter(alloc));
  auto* scale_data = static_cast<float*>(scale_buffer.get());
  math::Set<float, CPUMathUtil>(X_size, bias_, scale_data, &CPUMathUtil::Instance());

  const ptrdiff_t padded_square_size = (SafeInt<ptrdiff_t>(C) + size_ - 1) * H * W;
  auto psdata = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * padded_square_size);
  BufferUniquePtr padded_square_buffer(psdata, BufferDeleter(std::move(alloc)));
  auto* padded_square_data = static_cast<float*>(padded_square_buffer.get());
  math::Set<float, CPUMathUtil>(padded_square_size, 0.0f, padded_square_data, &CPUMathUtil::Instance());

  const float alpha_over_size = alpha_ / size_;
  // go through the images
  for (ptrdiff_t n = 0; n < N; ++n) {
    const ptrdiff_t n_times_image_size = n * image_size;

    // compute the padded square
    {
      const ptrdiff_t padded_square_data_offset = SafeInt<ptrdiff_t>(pre_pad) * H_times_W;
      math::Sqr<float, CPUMathUtil>(image_size, Xdata + n_times_image_size, padded_square_data + padded_square_data_offset,
                                    &CPUMathUtil::Instance());
    }
    // Create the first channel scale
    for (ptrdiff_t c = 0; c < size_; ++c) {
      const ptrdiff_t padded_square_data_offset = c * H_times_W;
      math::Axpy<float, CPUMathUtil>(H_times_W, alpha_over_size, padded_square_data + padded_square_data_offset,
                                     scale_data + n_times_image_size, &CPUMathUtil::Instance());
    }

    for (ptrdiff_t c = 1; c < C; ++c) {
      const ptrdiff_t this_scale_offset = n * image_size + c * H_times_W;

      float* this_scale_slice = scale_data + this_scale_offset;
      // copy previous scale
      memcpy(this_scale_slice, this_scale_slice - H_times_W, SafeInt<size_t>(H_times_W) * sizeof(float));
      // add head
      const ptrdiff_t padded_square_data_head_offset = (SafeInt<ptrdiff_t>(c) + size_ - 1) * H_times_W;
      math::Axpy<float, CPUMathUtil>(H_times_W, alpha_over_size, padded_square_data + padded_square_data_head_offset,
                                     this_scale_slice, &CPUMathUtil::Instance());
      // subtract tail
      const ptrdiff_t padded_square_data_tail_offset = (c - 1) * H_times_W;
      math::Axpy<float, CPUMathUtil>(H_times_W, -alpha_over_size, padded_square_data + padded_square_data_tail_offset,
                                     this_scale_slice, &CPUMathUtil::Instance());
    }
  }

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  using T = float;
  functors::Powx<T> f;
  f.input1 = scale_data;
  f.input2 = Xdata;
  f.b = -beta_;
  f.output = Ydata;
  concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(X_size),
                                          {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), f.Cost()}, f);
  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(LRN, 1, 12, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   LRN<float>);
ONNX_CPU_OPERATOR_KERNEL(LRN, 13, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         LRN<float>);

}  // namespace onnxruntime
