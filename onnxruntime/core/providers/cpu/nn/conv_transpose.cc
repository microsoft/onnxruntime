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

  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), p.input_shape.GetDims().begin(), p.input_shape.GetDims().end());

  if (p.X->Shape().NumDimensions() == 4) {
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

        // Col2im
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
      }

      if (p.B != nullptr) {
        auto Ymatrix = EigenMatrixMap<T>(Ydata, output_size, p.num_output_channels);
        auto Bvec = ConstEigenVectorMap<T>(p.B->template Data<T>(), p.num_output_channels);
        Ymatrix.rowwise() += Bvec.transpose();
      }

      Xdata += X_offset * conv_transpose_attrs_.group;
      Ydata += Y_offset * conv_transpose_attrs_.group;
    }
  } else {
    TensorShape output_shape = p.Y->Shape().Slice(1);
    output_shape[0] = output_shape[0] / conv_transpose_attrs_.group;

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

        // Col2im
        math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            output_shape.GetDims().data(),
            col_buffer_shape.data(),
            output_shape.Size(),
            col_buffer_size,
            p.kernel_shape.data(),
            p.strides.data(),
            p.dilations.data(),
            p.pads.data(),
            static_cast<int>(p.kernel_shape.size()),
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }

      if (p.B != nullptr) {
        auto Ymatrix = EigenMatrixMap<T>(Ydata, output_size, p.num_output_channels);
        auto Bvec = ConstEigenVectorMap<T>(p.B->template Data<T>(), p.num_output_channels);
        Ymatrix.rowwise() += Bvec.transpose();
      }

      Xdata += X_offset * conv_transpose_attrs_.group;
      Ydata += Y_offset * conv_transpose_attrs_.group;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
