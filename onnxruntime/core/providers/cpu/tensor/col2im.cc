// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/providers/cpu/tensor/col2im.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/common.h"
#include "core/framework/copy.h"
#include "core/common/safeint.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

#define REGISTER_KERNEL_TYPED(T)                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                           \
      Col2Im,                                                                               \
      1,                                                                                    \
      T,                                                                                    \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Col2Im<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status Col2Im<T>::Compute(OpKernelContext* context) const {
  const auto* col_input = context->Input<Tensor>(0);
  const auto* image_shape = context->Input<Tensor>(1);
  const auto* kernel_shape = context->Input<Tensor>(2);
  std::cout << "Status Col2Im<T>::Compute(OpKernelContext* context)" << std::endl;

  const T* col_input_data = col_input->template Data<T>();
  TensorShape col_input_shape = col_input->Shape();
  const auto col_input_N = col_input_shape[0];

  int64_t image_shape_size = 1;
  int64_t kernel_shape_size = 1;
  int64_t kernel_shape_rank = 0;
  for (auto i=0; i < image_shape->Shape().Size(); ++i) {
    ++kernel_shape_rank;
    image_shape_size *=  image_shape->Data<int64_t>()[i];
    kernel_shape_size *=  kernel_shape->Data<int64_t>()[i];
  }
  const int64_t C = col_input_shape[1] / kernel_shape_size;
  const int64_t col_output_stride = col_input_shape.SizeFromDimension(1);
  const int64_t col_input_stride = C * image_shape_size;

  TensorShapeVector Y_dims;
  Y_dims.insert(Y_dims.begin(), {col_input_N, C});
  for (auto i=0; i < image_shape->Shape()[0]; ++i) {
    Y_dims.push_back(image_shape->Data<int64_t>()[i]);
  }
  TensorShape Yshape(Y_dims);
  Tensor* Y = context->Output(0, Yshape);
  T* Ydata = Y->template MutableData<T>();
  for (auto i=0; i < Yshape.Size(); ++i)
    Ydata[i] = -1;  // just for debug (to know what has been written to Ydata in the end)
  // const int64_t Y_offset = Yshape.Size() / Yshape[0];

  std::cout << "\n\tInput 0: col_input = ("; for (auto i=0; i < Yshape.Size(); ++i) std::cout <<
    col_input_data[i] << ", "; std::cout << ") with shape "<< col_input_shape << std::endl;
  std::cout << "\tInput 1: image_shape = ("; for (auto i=0; i < image_shape->Shape().Size(); ++i) std::cout <<
    image_shape->Data<int64_t>()[i] << ", "; std::cout << ")" << std::endl;
  std::cout << "\tInput 2: kernel_shape = ("; for (auto i=0; i < kernel_shape->Shape().Size(); ++i) std::cout <<
    kernel_shape->Data<int64_t>()[i] << ", "; std::cout << ")" << std::endl;
  std::cout << "\tAttribute strides = ("; for (size_t i=0; i < col2im_attrs_.strides.size(); ++i) std::cout <<
    col2im_attrs_.strides[i] << ", "; std::cout << ")"<< std::endl;
  std::cout << "\tAttribute dilations = ("; for (size_t i=0; i < col2im_attrs_.dilations.size(); ++i) std::cout <<
    col2im_attrs_.dilations[i] << ", "; std::cout << ")"<< std::endl;
  std::cout << "\tAttribute pads = ("; for (size_t i=0; i < col2im_attrs_.pads.size(); ++i) std::cout <<
    col2im_attrs_.pads[i] << ", "; std::cout << ")"<< std::endl;

  std::cout << "\tVariable C: " << C << std::endl;
  std::cout << "\tVariable col_input_N = " << col_input_N << std::endl;
  std::cout <<  "\tVariable image_shape_size: " << image_shape_size << std::endl;
  std::cout <<  "\tVariable kernel_shape_size: " << kernel_shape_size << std::endl;

  std::cout << "\n\tStatus Col2Im<T>::Compute() --> math::Col2imNd<>()" << std::endl;

  assert(image_shape_size == Y_offset);  // just for temp debug

  for (auto image_id = 0; image_id < col_input_N; ++image_id) {
    std::cout << "Image " << image_id+1 << " out of "<< col_input_N << std::endl;
    if (image_shape->Shape()[0] == 2) {
      std::cout << "image_shape->Shape()[0] == 2 --> Col2Im" << std::endl;
      math::Col2im<float, CPUMathUtil, StorageOrder::NCHW>(
        col_input_data + image_id * col_output_stride,
        C,
        image_shape->Data<int64_t>()[0],
        image_shape->Data<int64_t>()[1],
        kernel_shape->Data<int64_t>()[0],
        kernel_shape->Data<int64_t>()[1],
        col2im_attrs_.dilations[0],
        col2im_attrs_.dilations[1],
        col2im_attrs_.pads[0],
        col2im_attrs_.pads[1],
        col2im_attrs_.pads[2],
        col2im_attrs_.pads[3],
        col2im_attrs_.strides[0],
        col2im_attrs_.strides[1],
        Ydata + image_id * col_input_stride,
        &CPUMathUtil::Instance());
    } else {
      std::cout << "image_shape->Shape()[0] != 2 --> Col2ImNd (nd=" << image_shape->Shape()[0] << ") " << std::endl;
      math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
        col_input_data + image_id * col_output_stride,    // const T* data_col,
        image_shape->Data<int64_t>(),                     // const int64_t* img_shape,
        Yshape.Slice(2).GetDims().data(),                 // const int64_t* output_shape,
        // col_input_shape[1],                               // int64_t channels_col,
        //    leads to output
        //          {1, -nan, 11, 16, 2.58141e+34, 2, 8.80295e+34, 12, 17, 22, 3, 4.59718e+24, 13, 18, 2.85144e+34, 4,
        //           -443.863, 14, -nan, 24, 5, 10, 15, 20, 25}
        //    that is similar to input with some spots with random values

        C,                                   // int64_t channels_col,
        //    leads to output {1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24, 5, 10, 15, 20, 25, }
        //    that is identical to input

        // col_input_shape[2],                               // int64_t channels_col,
        //    leads to output
        //          {1, 6, 1.92869e+31, 4.84145e+30, 1.88774e+31, 2, 7, 12, 17, 22, 3, 8, 1.86549e+31, 3.40686e+25,
        //           2.20182e+24, 4, -2.56655e+29, 5.08551e+31, -1.05888e+29, 1.51107e+29, 5, 10, 15, 20, 7.2793e+31}
        //    that is very similar to input, but with some rounded numbers and corrupted "25" value
        image_shape_size,                                 // int64_t img_size,
        kernel_shape->Data<int64_t>(),                    // const int64_t* kernel_shape,
        col2im_attrs_.strides.data(),                     // const int64_t* stride,
        col2im_attrs_.dilations.data(),                   // const int64_t* dilation,
        col2im_attrs_.pads.data(),                        // const int64_t* pad,
        kernel_shape->Shape().Size(),                     // ptrdiff_t N, --> #spatial_dims?
        Ydata + image_id * col_input_stride,              // T* data_img,
        &CPUMathUtil::Instance());                        // Provider* provider
    }
  }
  std::cout << "\n\n Return Col2Im<T>::Compute() --> "; for (auto i=0; i < Yshape.Size(); ++i) std::cout <<
    Ydata[i] << ", "; std::cout << ") with shape " << Yshape << std::endl << std::endl;

  return Status::OK();
}

}  // namespace onnxruntime
