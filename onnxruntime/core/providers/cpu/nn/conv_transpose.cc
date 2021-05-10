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

#include "core/providers/cpu/nn/conv_transpose.h"

#include "core/mlas/inc/mlas.h"
#include "core/common/safeint.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConvTranspose,
    1, 10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTranspose<float>);

ONNX_CPU_OPERATOR_KERNEL(
    ConvTranspose,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTranspose<float>);

template <typename T>
Status ConvTranspose<T>::PrePack(const Tensor& /*tensor*/, int /*input_idx*/, /*out*/ bool& is_packed,
                                 /*out*/ PrePackedWeights* /*prepacked_weight_for_caching*/,
                                 AllocatorPtr /*alloc*/) {
  is_packed = false;
  return Status::OK();
}

template <>
Status ConvTranspose<float>::PrePack(const Tensor& tensor, int input_idx, /*out*/ bool& is_packed,
                                     /*out*/ PrePackedWeights* prepacked_weight_for_caching,
                                     AllocatorPtr alloc) {
  is_packed = false;

  // only pack filter tensor
  if (input_idx == 1) {
    if (tensor.Shape().NumDimensions() <= 2) {
      return Status::OK();
    }
    filter_shape_ = tensor.Shape();

    const size_t K = static_cast<size_t>(filter_shape_[0]) / conv_transpose_attrs_.group;
    const size_t N = filter_shape_.SizeFromDimension(1);
    auto packed_elements_per_group = N * K;
    if (packed_elements_per_group == 0 || N == 1 || K == 1) {  // No need for single row or single col case
      return Status::OK();
    }

    size_t packed_filter_data_size = packed_elements_per_group * sizeof(float) * conv_transpose_attrs_.group;
    auto* packed_filter_data = alloc->Alloc(packed_filter_data_size);
    transposed_filter_ = BufferUniquePtr(packed_filter_data, BufferDeleter(alloc));

    for (int64_t group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      MlasTranspose(tensor.Data<float>() + (N * K * group_id),
                    ((float*)packed_filter_data) + (group_id * packed_elements_per_group),
                    K, N);
    }

    bool share_prepacked_weights = (prepacked_weight_for_caching != nullptr);
    if (share_prepacked_weights) {
      prepacked_weight_for_caching->buffers_.push_back(std::move(transposed_filter_));
      prepacked_weight_for_caching->buffer_sizes_.push_back(packed_filter_data_size);
      prepacked_weight_for_caching->shapes_.push_back(filter_shape_);
      prepacked_weight_for_caching->is_filled_ = true;
    }

    is_packed = true;
  }
  return Status::OK();
}

template <typename T>
Status ConvTranspose<T>::StorePrePackedWeight(const PrePackedWeights& /*prepacked_weight*/,
                                              int /*input_idx*/,
                                              /*out*/ bool& stored_weight) {
  stored_weight = false;
  return Status::OK();
}

template <>
Status ConvTranspose<float>::StorePrePackedWeight(const PrePackedWeights& prepacked_weight,
                                                  int input_idx,
                                                  /*out*/ bool& stored_weight) {
  stored_weight = false;

  if (input_idx == 1) {
    stored_weight = true;
    filter_shape_ = prepacked_weight.shapes_[0];
    transposed_filter_ = BufferUniquePtr(prepacked_weight.buffers_[0].get(), BufferDeleter(nullptr));
  }

  return Status::OK();
}

template <typename T>
Status ConvTranspose<T>::Compute(OpKernelContext* context) const {
  return ConvTranspose<T>::DoConvTranspose(context, false);
}

template <typename T>
Status ConvTranspose<T>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  ConvTransposeAttributes::Prepare p;
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, dynamic_padding));

  // Bail out early if one of the dimensions is zero.
  if (p.Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = p.input_shape.Size();
  const int64_t X_offset = p.num_input_channels / conv_transpose_attrs_.group * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / conv_transpose_attrs_.group;
  const int64_t W_offset = p.F->Shape().Size() / conv_transpose_attrs_.group;
  const int64_t kernel_size = TensorShape(p.kernel_shape).Size();
  const int64_t kernel_dim = p.num_output_channels / conv_transpose_attrs_.group * kernel_size;
  const int64_t output_size = (p.Y->Shape().Slice(2)).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const int64_t col_buffer_size = kernel_dim * p.input_shape.Size();
  auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(T)) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = p.X->template Data<T>();
  const T* filter_data = p.F->template Data<T>();
  T* Ydata = p.Y->template MutableData<T>();
  TensorShape output_shape = p.Y->Shape().Slice(2);

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      // Weight term
      math::Gemm<T>(
          CblasTrans,
          CblasNoTrans,
          kernel_dim,
          input_image_size,
          p.num_input_channels / conv_transpose_attrs_.group,
          1,
          filter_data + group_id * W_offset,
          Xdata + group_id * X_offset,
          0,
          col_buffer_data,
          thread_pool);

      if (p.X->Shape().NumDimensions() == 4) {
        math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            p.num_output_channels / conv_transpose_attrs_.group,
            p.Y->Shape()[2],
            p.Y->Shape()[3],
            p.kernel_shape[0],
            p.kernel_shape[1],
            p.dilations[0],
            p.dilations[1],
            p.pads[0],
            p.pads[1],
            p.pads[2],
            p.pads[3],
            p.strides[0],
            p.strides[1],
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      } else {
        math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            output_shape.GetDims().data(),
            p.input_shape.GetDims().data(),
            kernel_dim,
            Y_offset,
            p.kernel_shape.data(),
            p.strides.data(),
            p.dilations.data(),
            p.pads.data(),
            static_cast<int>(p.kernel_shape.size()),
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }
    }

    if (p.B != nullptr) {
      auto Ymatrix = EigenMatrixMap<T>(Ydata, output_size, p.num_output_channels);
      auto Bvec = ConstEigenVectorMap<T>(p.B->template Data<T>(), p.num_output_channels);
      Ymatrix.rowwise() += Bvec.transpose();
    }

    Xdata += X_offset * conv_transpose_attrs_.group;
    Ydata += Y_offset * conv_transpose_attrs_.group;
  }

  return Status::OK();
}

template <>
Status ConvTranspose<float>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  ConvTransposeAttributes::Prepare p;
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(
      context, has_bias, p, dynamic_padding, transposed_filter_ ? &filter_shape_ : nullptr));

  // Bail out early if one of the dimensions is zero.
  if (p.Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = p.input_shape.Size();
  const int64_t X_offset = p.num_input_channels / conv_transpose_attrs_.group * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / conv_transpose_attrs_.group;
  const int64_t W_offset = (p.F ? p.F->Shape().Size() : filter_shape_.Size()) / conv_transpose_attrs_.group;
  const int64_t kernel_size = TensorShape(p.kernel_shape).Size();
  const int64_t kernel_dim = p.num_output_channels / conv_transpose_attrs_.group * kernel_size;
  const int64_t output_size = (p.Y->Shape().Slice(2)).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const int64_t col_buffer_size = kernel_dim * p.input_shape.Size();
  auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  float* col_buffer_data = static_cast<float*>(col_buffer.get());

  const float* Xdata = p.X->template Data<float>();
  const float* filter_data = p.F ? p.F->template Data<float>() : static_cast<float*>(transposed_filter_.get());
  float* Ydata = p.Y->template MutableData<float>();
  TensorShape output_shape = p.Y->Shape().Slice(2);

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      // Weight term
      math::Gemm<float>(
          p.F ? CblasTrans : CblasNoTrans,
          CblasNoTrans,
          kernel_dim,
          input_image_size,
          p.num_input_channels / conv_transpose_attrs_.group,
          1,
          filter_data + group_id * W_offset,
          Xdata + group_id * X_offset,
          0,
          col_buffer_data,
          thread_pool);

      if (p.X->Shape().NumDimensions() == 4) {
        math::Col2im<float, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            p.num_output_channels / conv_transpose_attrs_.group,
            p.Y->Shape()[2],
            p.Y->Shape()[3],
            p.kernel_shape[0],
            p.kernel_shape[1],
            p.dilations[0],
            p.dilations[1],
            p.pads[0],
            p.pads[1],
            p.pads[2],
            p.pads[3],
            p.strides[0],
            p.strides[1],
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      } else {
        math::Col2imNd<float, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            output_shape.GetDims().data(),
            p.input_shape.GetDims().data(),
            kernel_dim,
            Y_offset,
            p.kernel_shape.data(),
            p.strides.data(),
            p.dilations.data(),
            p.pads.data(),
            static_cast<int>(p.kernel_shape.size()),
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }
    }

    if (p.B != nullptr) {
      auto Ymatrix = EigenMatrixMap<float>(Ydata, output_size, p.num_output_channels);
      auto Bvec = ConstEigenVectorMap<float>(p.B->template Data<float>(), p.num_output_channels);
      Ymatrix.rowwise() += Bvec.transpose();
    }

    Xdata += X_offset * conv_transpose_attrs_.group;
    Ydata += Y_offset * conv_transpose_attrs_.group;
  }

  return Status::OK();
}
}  // namespace onnxruntime
