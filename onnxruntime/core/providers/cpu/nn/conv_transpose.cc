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

static void Transpose(const float* src, float* dst, size_t M, size_t N) {
  EigenMatrixMapRowMajor<float>(dst, N, M) = ConstEigenMatrixMapRowMajor<float>(src, M, N).transpose();
}

template <typename T>
Status ConvTranspose<T>::PrePack(const Tensor& /* tensor */, int /* input_idx */, bool& is_packed) {
  is_packed = false;
  return Status::OK();
}

template <>
Status ConvTranspose<float>::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  // only pack filter tensor
  if (input_idx == 1) {
    if (tensor.Shape().NumDimensions() <= 2) {
      return Status::OK();
    }
    filter_shape_ = tensor.Shape();

    const size_t K = static_cast<size_t>(filter_shape_[0]) / conv_transpose_attrs_.group;
    const size_t N = filter_shape_.SizeFromDimension(1);
    packed_bytes_per_group_ = MlasGemmPackBSize(N, K);
    if (packed_bytes_per_group_ == 0) {
      return Status::OK();
    }

    auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
    auto* packed_filter_data = alloc->Alloc(packed_bytes_per_group_ * conv_transpose_attrs_.group);
    packed_filter_ = BufferUniquePtr(packed_filter_data, BufferDeleter(alloc));

    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      MlasGemmPackB(
          CblasNoTrans,
          N,
          K,
          tensor.Data<float>() + (N * K * group_id),
          N,
          ((char*)packed_filter_data) + (packed_bytes_per_group_ * group_id));
    }

    // Do not set
    //    is_packed = true;
    // This kind of packing improve well when input image size (h x w) is small.
    // Need keep original tensor for other case.
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
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, dynamic_padding));
  const TensorShape& F_Shape = p.F->Shape();

  // Bail out early if one of the dimensions is zero.
  if (p.Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = p.input_shape.Size();
  const int64_t X_offset = p.num_input_channels / conv_transpose_attrs_.group * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / conv_transpose_attrs_.group;
  const int64_t W_offset = F_Shape.Size() / conv_transpose_attrs_.group;
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
  bool use_prepacked_filter = !dynamic_padding && packed_filter_ && input_image_size <= 16 && kernel_dim > input_image_size;
  const float* filter_data = use_prepacked_filter ? static_cast<float*>(packed_filter_.get()) : p.F->template Data<float>();
  float* Ydata = p.Y->template MutableData<float>();
  TensorShape output_shape = p.Y->Shape().Slice(2);

  BufferUniquePtr extra_buffer;
  if (use_prepacked_filter) {
    auto extra_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * col_buffer_size);
    extra_buffer = BufferUniquePtr(extra_data, BufferDeleter(alloc));
  }
  float* extra_buffer_data = static_cast<float*>(extra_buffer.get());

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      // Weight term
      if (!use_prepacked_filter) {
        math::Gemm<float>(
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
      } else {
        MlasGemm(
            CblasTrans,
            static_cast<size_t>(input_image_size),
            static_cast<size_t>(kernel_dim),
            static_cast<size_t>(p.num_input_channels / conv_transpose_attrs_.group),
            1.0f,
            Xdata + group_id * X_offset,
            static_cast<size_t>(input_image_size),
            (float*)(((char*)filter_data) + group_id * packed_bytes_per_group_),
            0.0f,
            extra_buffer_data,
            static_cast<size_t>(kernel_dim),
            thread_pool);

        Transpose(extra_buffer_data, col_buffer_data, input_image_size, kernel_dim);
      }

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
