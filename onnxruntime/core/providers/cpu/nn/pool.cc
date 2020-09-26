// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/pool.h"
#include "core/framework/data_types_internal.h"
#include "core/platform/threadpool.h"
#include "pool_functors.h"
#include <iostream>

using namespace ::onnxruntime::common;

namespace onnxruntime {

template <typename T>
inline static void RunLoop(concurrency::ThreadPool* tp, std::ptrdiff_t total_channels, T&& task) {
  concurrency::ThreadPool::TryParallelFor(tp, total_channels, task.Cost(), task);
}

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  if (pool_attrs_.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, output_dims);

  const auto* X_data = X->template Data<T>();
  auto* Y_data = Y->template MutableData<T>();

  // The main loop
  const int64_t channels = x_shape[1];
  const int64_t height = x_shape[2];
  const int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  const int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  const int64_t pooled_height = output_dims[2];
  const int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  const int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;
  const int64_t total_channels = x_shape[0] * channels;
  const int64_t x_step = height * width * depth;
  const int64_t y_step = pooled_height * pooled_width * pooled_depth;

  switch (kernel_shape.size()) {
    case 1: {
      RunLoop<Pool1DTask<T, PoolType>>(tp, total_channels,
                                       {X_data, Y_data, x_step, y_step, pooled_height, stride_h(), height, kernel_shape,
                                        pads, pool_context_, pool_attrs_});

      break;
    }

    case 2: {
      RunLoop<Pool2DTask<T, PoolType>>(tp, total_channels,
                                       {X_data, Y_data, x_step, y_step, pooled_height, pooled_width, stride_h(),
                                        stride_w(), height, width, kernel_shape, pads, pool_context_, pool_attrs_});

      break;
    }
    case 3: {
      RunLoop<Pool3DTask<T, PoolType>>(
          tp, total_channels,
          {X_data, Y_data, x_step, y_step, pooled_height, pooled_width, pooled_depth, stride_h(), stride_w(),
           stride_d(), height, width, depth, kernel_shape, pads, pool_context_, pool_attrs_});

      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

Status PoolBase::Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!pool_attrs_.global_pooling) {
    ORT_RETURN_IF_NOT(pooling_dims == pool_attrs_.kernel_shape.size(),
                      "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape output_shape(output_dims);
  Tensor* Y = context->Output(0, output_shape);

  // edge case: one or more dims with value of 0
  if (output_shape.Size() == 0)
    return Status::OK();

  // Get access to the internal threadpool
  // Temporarily derive concurrency parameters without access to session state
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  MlasPool(kind, pooling_dims, X->Shape().GetDims().data(),
           pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
           pool_attrs_.global_pooling ? nullptr : pads.data(),
           pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(), output_dims.data(),
           X->template Data<float>(), Y->template MutableData<float>(), thread_pool);

  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1 /*VERSION*/>>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context, MlasMaximumPooling);
}

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context,
                           pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad : MlasAveragePoolingExcludePad);
}


Status MaxPoolV8::Compute(OpKernelContext* context) const {
  utils::MLTypeCallDispatcherRet<Status, ComputeHelper, float, double, int8_t, uint8_t>
      t_disp(context->Input<Tensor>(0)->GetElementType());
  return t_disp.Invoke(this, context);
}

template <typename T>
Status MaxPoolV8::ComputeImpl(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  // Use MLAS pooling if the index output tensor is not used
  // and also if dilation is not required

  bool need_dilation = false;
  for (auto n : pool_attrs_.dilations) {
    need_dilation |= n > 1;
  }

  // MLAS implementation currently supports only floats
  if (std::is_same<T, float>::value) {
    if (OpKernel::Node().OutputDefs().size() == 1 && pool_attrs_.storage_order == 0 && !need_dilation) {
      return PoolBase::Compute(context, MlasMaximumPooling);
    }
  }

  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, output_dims);
  Tensor* I = context->Output(1, output_dims);

  const auto* X_data = X->template Data<T>();
  auto* Y_data = Y->template MutableData<T>();
  int64_t* I_data = I != nullptr ? I->template MutableData<int64_t>() : nullptr;

  // The main loop
  int64_t channels = x_shape[1];
  int64_t height = x_shape[2];
  int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  int64_t pooled_height = output_dims[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;
  const int64_t total_channels = x_shape[0] * channels;

  switch (kernel_shape.size()) {
    case 1: {
      int64_t x_step = height;
      int64_t y_step = pooled_height;
      const int64_t dilation_h = pool_attrs_.dilations[0];

      RunLoop<MaxPool1DTask<T>>(tp, total_channels,
                                {X_data, Y_data, I_data, x_step, y_step, dilation_h, pooled_height, stride_h(),
                                 height, kernel_shape, pads});
      break;
    }

    case 2: {
      int64_t x_step = height * width;
      int64_t y_step = pooled_height * pooled_width;
      const int64_t dilation_h = pool_attrs_.dilations[0];
      const int64_t dilation_w = pool_attrs_.dilations[1];
      RunLoop<MaxPool2DTask<T>>(
          tp, total_channels,
          {X_data, Y_data, I_data, x_step, y_step, dilation_h, dilation_w, pooled_height, pooled_width, stride_h(),
           stride_w(), height, width, kernel_shape, pads, pool_attrs_.storage_order});
      break;
    }
    case 3: {
      int64_t x_step = height * width * depth;
      int64_t y_step = pooled_height * pooled_width * pooled_depth;
      const int64_t dilation_h = pool_attrs_.dilations[0];
      const int64_t dilation_w = pool_attrs_.dilations[1];
      const int64_t dilation_d = pool_attrs_.dilations[2];
      RunLoop<MaxPool3DTask<T>>(tp, total_channels,
                                {X_data, Y_data, I_data, x_step, y_step,
                                 dilation_h, dilation_w, dilation_d, pooled_height, pooled_width,
                                 pooled_depth, stride_h(), stride_w(), stride_d(), height,
                                 width, depth, kernel_shape, pads, pool_attrs_.storage_order});
      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

bool MaxPoolV8::OptimizeWorthy1D(int64_t total_height, int64_t pooled_height, int64_t pool_size) const {
  float layer_1_weight [8][3] =
    {{-0.4044543 ,  0.5032941 ,  0.60934484},
     {-0.20548528,  0.593789  ,  0.4940056 },
     { 0.11417829, -0.46909815,  0.1906032 },
     { 0.40023336, -0.4313046 ,  0.13845278},
     { 0.0216579 ,  0.25878215, -4.625198  },
     { 0.45919985,  0.03440802, -0.0537789 },
     {-0.29694313, -0.39878082, -0.56551826},
     { 0.2505781 , -0.505424  , -0.09284496}};
  float layer_1_bias[8] = {-1.9949601, -15.046263, -11.331387, -5.585094 ,  
                           16.095648, -6.071184 , -0.5011854,  -4.4836655};
  float layer_1_output[8];
  for (int64_t i = 0; i < 8; i++) {
    layer_1_output[i] = layer_1_weight[i][0] * total_height +
                        layer_1_weight[i][1] * pooled_height + 
                        layer_1_weight[i][2] * pool_size +
                        layer_1_bias[i];
    if (layer_1_output[i] < 0) {
      layer_1_output[i] = 0;
    }
  }//for
  float layer_2_weight[6][8] = 
    {{-0.1791764 ,  0.12306077, -0.2359769 ,  0.272206  , -0.0967816 , -0.23264278, -0.22643457, -0.27852008},
     { 0.20220038,  0.08625653, -0.07199997,  0.00175294, -0.37816948,  0.10361968,  0.04760075, -0.1257332  },
     { 0.2817652 , -0.16080262, -0.34706196,  0.10345992, -0.08791664, -0.32266164,  0.35026613, -0.2816986  },
     {-0.18292245,  0.3425173 , -0.28006834,  0.12972653, -0.46211034,  0.03205561, -0.0626438 , -0.10230546},
     {-0.31267866, -0.18372081,  0.06138921, -0.24455047,  0.10364124, -0.30473444, -0.11504889,  0.3488299 },
     {-0.03049979, -0.3484844 , -0.09377559,  0.2814676 ,  0.14773709, -0.26475713,  0.30883536, -0.08175805}};
  float layer_2_bias[6] = {0.0548716 , -5.820129  , -0.14218597, -6.267937  , -0.21381079, -0.14634752};
  float layer_2_output[6];
  for (int64_t i = 0; i < 6; i++) {
    layer_2_output[i] = 0;
    for (int64_t j = 0; j < 8; j++) {
      layer_2_output[i] += layer_2_weight[i][j] * layer_1_output[j];
    }
    layer_2_output[i] += layer_2_bias[i];
    if (layer_2_output[i] < 0) {
      layer_2_output[i] = 0;
    }
  }
  float layer_3_weight[6] = {-0.09621256,  0.03613849, -0.0679929 , 0.03177442,  0.04737011, -0.07048248};
  float layer_3_bias = -2.2467241;
  float layer_3_output = 0;
  for (int64_t i = 0; i < 6; i++) {
    layer_3_output += layer_2_output[i] * layer_3_weight[i];
  }
  layer_3_output += layer_3_bias;
  float sigmoid = 1.0 / (1 + std::pow(2.718, -layer_3_output));
  return std::round(sigmoid) > 0.9;
}

template <typename T>
Status MaxPoolV8::ComputeImplOptimized(OpKernelContext* context) const {

  const auto* X = context->Input<Tensor>(0);
  const auto* X_data = X->template Data<T>();
  const TensorShape& x_shape = X->Shape();
  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  const auto& dilations = pool_attrs_.dilations;
  const auto& pool_size = pool_attrs_.kernel_shape;

  if (dilations[0] != stride_h() ||
      pool_size.size() > 1 && dilations[1] != stride_w() || 
      pool_size.size() > 2 && dilations[2] != stride_d() ||
      context->Output(1, output_dims)) {

    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Arguments are not optimizable.");
  }

  if (pool_size.size() == 1 && !OptimizeWorthy1D(x_shape[2] + pads[0] + pads[1], output_dims[2], pool_size[0])) {
    std::cout << "case should go with naive." << std::endl;
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Arguments should not be optimized.");
  }

  Tensor* Y = context->Output(0, output_dims);
  auto* Y_data = Y->template MutableData<T>();
  auto tp = context->GetOperatorThreadPool();
  auto channels = x_shape[0] * x_shape[1];

  switch (pool_size.size()) {
    case 1: {
      RunLoop<MaxPool1DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], output_dims[2],
                                    pads[0], pads[1], stride_h(), pool_size[0]});
      break;
    }
    case 2: {
      RunLoop<MaxPool2DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], x_shape[3],
                                    output_dims[2], output_dims[3],
                                    pads[0], pads[2], pads[1], pads[3],
                                    stride_h(), stride_w(),
                                    pool_size[0], pool_size[1]});
      break;
    }
    case 3: {
      RunLoop<MaxPool3DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], x_shape[3], x_shape[4],
                                    output_dims[2], output_dims[3], output_dims[4],
                                    pads[0], pads[3], pads[1], pads[4], pads[2], pads[5],
                                    stride_h(), stride_w(), stride_d(),
                                    pool_size[0], pool_size[1], pool_size[2]});
      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  return Status::OK();
}
 
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(AveragePool, 7, 9,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(AveragePool, 10, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(AveragePool, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(MaxPool, 1, 7,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, MaxPool<1 /*VERSION*/>>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(MaxPool, 8, 11, 
                                         KernelDefBuilder()
                                             .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                   DataTypeImpl::GetTensorType<double>()})
                                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                         MaxPoolV8);

ONNX_CPU_OPERATOR_KERNEL(MaxPool, 12,
                         KernelDefBuilder()
                             .TypeConstraint("T", {DataTypeImpl::GetTensorType<double>(),
                                                   DataTypeImpl::GetTensorType<float>(),
                                                   DataTypeImpl::GetTensorType<int8_t>(),
                                                   DataTypeImpl::GetTensorType<uint8_t>()})
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                         MaxPoolV8);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(LpPool, 2, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(LpPool, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalLpPool, 2, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalAveragePool, 1,
                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalMaxPool, 1, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, MaxPool<1 /*VERSION*/>>);

}  // namespace onnxruntime
