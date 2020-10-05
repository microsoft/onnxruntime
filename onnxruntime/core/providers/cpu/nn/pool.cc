// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/pool.h"
#include "core/framework/data_types_internal.h"
#include "core/platform/threadpool.h"
#include "pool_functors.h"

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

bool MaxPoolV8::Optimizable1D(int64_t total_height, int64_t pooled_height, int64_t pool_size) {

  float layer_1_weight [8][3] =
    {{-0.4044543f ,  0.5032941f ,  0.60934484f},
     {-0.20548528f,  0.593789f  ,  0.4940056f },
     { 0.11417829f, -0.46909815f,  0.1906032f },
     { 0.40023336f, -0.4313046f ,  0.13845278f},
     { 0.0216579f ,  0.25878215f, -4.625198f  },
     { 0.45919985f,  0.03440802f, -0.0537789f },
     {-0.29694313f, -0.39878082f, -0.56551826f},
     { 0.2505781f , -0.505424f  , -0.09284496f}};
  float layer_1_bias[8] = {-1.9949601f, -15.046263f, -11.331387f,  -5.585094f,
                           16.095648f,   -6.071184f,  -0.5011854f, -4.4836655f};
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
    {{-0.1791764f ,  0.12306077f, -0.2359769f ,  0.272206f  , -0.0967816f , -0.23264278f, -0.22643457f, -0.27852008f},
     { 0.20220038f,  0.08625653f, -0.07199997f,  0.00175294f, -0.37816948f,  0.10361968f,  0.04760075f, -0.1257332f  },
     { 0.2817652f , -0.16080262f, -0.34706196f,  0.10345992f, -0.08791664f, -0.32266164f,  0.35026613f, -0.2816986f  },
     {-0.18292245f,  0.3425173f , -0.28006834f,  0.12972653f, -0.46211034f,  0.03205561f, -0.0626438f , -0.10230546f},
     {-0.31267866f, -0.18372081f,  0.06138921f, -0.24455047f,  0.10364124f, -0.30473444f, -0.11504889f,  0.3488299f },
     {-0.03049979f, -0.3484844f , -0.09377559f,  0.2814676f ,  0.14773709f, -0.26475713f,  0.30883536f, -0.08175805f}};
  float layer_2_bias[6] = {0.0548716f , -5.820129f  , -0.14218597f, -6.267937f  , -0.21381079f, -0.14634752f};
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
  float layer_3_weight[6] = {-0.09621256f,  0.03613849f, -0.0679929f , 0.03177442f,  0.04737011f, -0.07048248f};
  float layer_3_bias = -2.2467241f;
  float layer_3_output = 0;
  for (int64_t i = 0; i < 6; i++) {
    layer_3_output += layer_2_output[i] * layer_3_weight[i];
  }
  layer_3_output += layer_3_bias;
  float sigmoid = 1.0f / (1 + std::pow(2.718f, -layer_3_output));
  return std::round(sigmoid) > 0.9f;
}

bool MaxPoolV8::Optimizable2D(int64_t total_height, int64_t total_width,
                              int64_t pooled_height, int64_t pooled_width,
                              int64_t pool_height, int64_t pool_width) {

  float layer_1_weight [10][6] =
    {{-3.6831051e-01f, -3.6247692e-01f,  2.9945856e-01f, -2.7339315e-01f, -3.2982734e-01f,  1.2428343e-01f},
     { 4.2411339e-02f,  7.3652379e-02f, -1.1140941e-01f,  1.9908203e-01f,  6.4203119e-01f,  7.2492361e-01f},
     {-4.3322185e-01f,  1.3085116e-02f,  3.9197430e-01f,  6.1275326e-02f,  4.0028703e-01f,  1.2761176e+00f},
     { 1.3293332e-01f,  1.4291838e-01f,  3.2274619e-01f, -1.9370590e-01f,  9.8541480e-01f,  1.1948491e+00f},
     { 2.5940394e-02f,  3.8113617e-03f, -3.4423352e-03f,  1.4519133e-01f, -7.7429314e+00f, -5.7754173e+00f},
     {-2.0399491e-01f, -3.1316891e-01f,  3.2469466e-01f,  3.0748990e-01f,  4.2247924e-01f, -1.4207372e-01f},
     { 2.3843075e-01f,  2.5791006e-02f,  3.8117608e-01f,  8.0572687e-02f,  1.2876539e+00f,  7.6808077e-01f},
     { 1.9901858e-01f,  4.5600232e-02f, -9.8639183e-02f, -5.6079019e-02f, -2.5981524e+00f,  7.9628939e-05f},
     { 1.5695563e-01f,  2.5528669e-03f, -1.2300680e+00f,  4.4656624e-03f,  6.9656110e-01f,  1.7935342e-01f},
     { 1.7079201e-01f, -2.7161598e-02f, -1.3937990e-01f,  8.6947553e-02f,  2.2510707e+00f,  8.4009208e-02f}};
  float layer_1_bias[10] = {-0.3770961f, -2.3918433f,  1.8521361f, -4.5703444f, -2.7904446f,
                             6.6001234f, -2.1826804f,  3.2673945f,  9.796883f , -1.8809853f};
  float layer_1_output[10];
  for (int64_t i = 0; i < 10; i++) {
    layer_1_output[i] = layer_1_weight[i][0] * total_height +
                        layer_1_weight[i][1] * total_width + 
                        layer_1_weight[i][2] * pooled_height +
                        layer_1_weight[i][3] * pooled_width +
                        layer_1_weight[i][4] * pool_height +
                        layer_1_weight[i][5] * pool_width +
                        layer_1_bias[i];
    if (layer_1_output[i] < 0) {
      layer_1_output[i] = 0;
    }
  }//for

  float layer_2_weight[8][10] = 
     {{-0.06003293f,  0.21225819f, -0.27200642f, -0.02082756f, -0.0701707f,  -0.20068413f, -0.50153553f,  0.00336754f,  0.6702372f ,  0.05447913f},
      { 0.23352525f, -0.08489721f,  0.19231986f, -0.27247515f, -0.15134875f,  0.49599656f,  0.11655813f, -0.02076937f,  0.17092028f, -0.07972863f},
      {-0.06445351f,  0.1792246f ,  0.16155557f, -0.07104914f, -0.50501835f, -1.741571f  ,  0.11375787f, -0.10069937f, -0.09629883f,  0.0153533f },
      {-0.28012472f,  0.19438729f, -0.05561933f,  0.05643161f, -0.48072016f, -0.10830858f,  0.03165498f, -0.29761288f, -0.7303268f ,  0.23473336f},
      { 0.06735539f,  0.10022206f,  0.64842635f, -0.009133f  , -0.6126588f , -0.10844892f,  0.07485867f, -0.10075383f, -0.04458744f,  0.07074562f},
      { 0.1900272f , -0.09800401f, -0.21638612f, -0.18487929f, -0.13792641f, -0.25938094f, -0.15732956f, -0.01412544f,  0.05573884f, -0.09582533f},
      {-0.14016639f, -0.03206995f, -0.1200158f ,  0.07844546f, -0.28183854f, -0.04650053f, -0.19275935f, -0.2222099f ,  0.29764298f, -0.18808417f},
      {-0.30399063f,  0.18053997f, -0.3222996f , -0.01604891f, -0.44561228f, -0.22320613f, -0.09742685f, -0.28637683f, -0.5639017f , -0.05816495f}};
  float layer_2_bias[8] = {4.5093737f ,  7.8021812f , -3.8440096f,  1.0618207f , -3.847487f, 0.2664036f , -0.11398777f,  0.15493515f};
  float layer_2_output[8];
  for (int64_t i = 0; i < 8; i++) {
    layer_2_output[i] = 0;
    for (int64_t j = 0; j < 10; j++) {
      layer_2_output[i] += layer_2_weight[i][j] * layer_1_output[j];
    }
    layer_2_output[i] += layer_2_bias[i];
    if (layer_2_output[i] < 0) {
      layer_2_output[i] = 0;
    }
  }
  float layer_3_weight[8] = {-0.3139295f, -0.5689301f, 0.04450566f, 0.05143051f, 0.03166565f, -0.02240658f, -0.18378934f, 0.8769102f};
  float layer_3_bias = -1.4846476f;
  float layer_3_output = 0.0f;
  for (int64_t i = 0; i < 8; i++) {
    layer_3_output += layer_2_output[i] * layer_3_weight[i];
  }
  layer_3_output += layer_3_bias;
  float sigmoid = 1.0f / (1 + std::pow(2.718f, -layer_3_output));
  return std::round(sigmoid) > 0.9f;
}

bool MaxPoolV8::Optimizable3D(int64_t total_height, int64_t total_width, int64_t total_depth,
                              int64_t pooled_height, int64_t pooled_width, int64_t pooled_depth,
                              int64_t pool_height, int64_t pool_width, int64_t pool_depth) {

  float layer_1_weight [24][9] =
    {{-2.62563792e-03f, -5.30446507e-02f,  2.34065652e-01f,
       5.46038374e-02f, -3.45419914e-01f, -3.37030739e-01f,
      -1.68257964e+00f, -3.75687337e+00f,  6.25928283e-01f},
     {-2.90238768e-01f,  5.11394918e-01f, -2.33412325e-01f,
      -3.06519028e-02f, -9.72875133e-02f, -9.02358592e-01f,
       2.55234428e-02f,  1.42679536e+00f, -1.05575956e-01f},
     { 2.28006095e-01f, -2.12063938e-01f, -8.93138722e-03f,
      -4.38422300e-02f,  3.93723994e-01f,  3.95294964e-01f,
      -4.49067593e+00f, -2.12804770e+00f, -7.67036259e-01f},
     { 2.49757111e-01f,  8.44950229e-02f, -6.02803230e-02f,
       1.49219617e-01f, -2.67901540e-01f,  2.23192289e-01f,
      -9.25262809e-01f, -1.88115227e+00f, -8.86296034e-01f},
     {-1.02339089e-02f, -3.50278914e-02f, -1.75999761e-01f,
      -1.23177618e-01f, -2.83262551e-01f, -2.10444257e-01f,
       1.94676965e-01f, -1.28772065e-01f,  8.25972557e-02f},
     {-5.02367388e-04f,  2.08596200e-01f, -1.11318283e-01f,
       6.20643049e-02f,  1.41828865e-01f,  1.58213690e-01f,
      -5.40728760e+00f, -6.18583727e+00f, -1.31650937e+00f},
     {-1.55238152e-01f, -7.08388863e-03f,  6.67445511e-02f,
       6.80753887e-02f,  2.53242403e-01f,  4.21643853e-01f,
       8.56192291e-01f, -2.95422107e-01f,  4.33228398e-03f},
     { 9.16738510e-02f,  1.65791899e-01f, -1.33574992e-01f,
       1.69505015e-01f, -9.71464992e-01f,  4.19773668e-01f,
      -5.37779510e-01f, -1.56131935e+00f, -6.95862412e-01f},
     { 1.52093828e-01f,  1.18437685e-01f,  1.13814184e-02f,
      -8.06498110e-01f, -3.68173346e-02f,  3.00366312e-01f,
      -8.09030771e-01f, -2.62377357e+00f,  1.04450233e-01f},
     { 1.10504225e-01f,  4.65243123e-02f, -1.38675615e-01f,
       4.75711375e-01f,  6.39054656e-01f,  5.17303906e-02f,
       3.84413958e+00f,  3.48990345e+00f,  9.68467116e-01f},
     { 9.89119187e-02f,  1.69153407e-01f, -4.01038587e-01f,
       3.73209305e-02f, -1.03733778e+00f,  4.28319722e-01f,
      -4.51276016e+00f,  2.95217156e+00f, -1.26944888e+00f},
     { 2.27370858e-01f, -1.21777460e-01f, -4.39528435e-01f,
      -1.71541139e-01f, -1.63905323e-01f,  6.10745013e-01f,
       4.70478326e-01f,  1.07273519e-01f,  7.86691964e-01f},
     {-2.07956076e-01f, -3.28115761e-01f, -1.19947195e-01f,
       2.91641384e-01f,  8.40614140e-02f, -2.07141280e-01f,
      -9.04080570e-02f,  1.46309495e-01f,  1.40942603e-01f},
     { 8.92240107e-02f, -2.87236810e-01f, -1.35209084e-01f,
      -3.98740172e-03f, -9.66299474e-02f, -2.66291559e-01f,
       8.35621357e-02f, -7.33549595e-02f,  2.45053500e-01f},
     {-5.85124344e-02f,  3.15563306e-02f, -7.05983192e-02f,
      -1.51799947e-01f, -5.31043150e-02f,  7.27567732e-01f,
       1.70895469e+00f,  1.67074513e+00f,  3.31129998e-01f},
     { 3.44953872e-02f, -1.61964670e-02f,  1.49024531e-01f,
       1.36987075e-01f, -1.14930034e-01f,  5.58183670e-01f,
      -3.62688303e+00f, -5.29517937e+00f, -1.51534808e+00f},
     { 1.77419022e-01f,  2.91416943e-01f, -2.08404243e-01f,
      -1.43991232e+00f,  7.04851151e-02f,  1.16853431e-01f,
      -2.46599698e+00f, -8.28545392e-01f, -1.95772592e-02f},
     {-5.90571165e-02f, -2.32495189e-01f, -2.58228123e-01f,
       5.89510798e-03f,  2.16038853e-01f, -1.14887565e-01f,
       1.73090965e-01f,  2.27321059e-01f, -1.15775391e-01f},
     {-3.76597904e-02f,  1.94580909e-02f,  1.08192563e-01f,
      -9.76224989e-03f, -8.37564841e-02f,  1.43837899e-01f,
       3.45656204e+00f,  5.42169142e+00f,  6.35727704e-01f},
     { 7.53453299e-02f,  1.42042771e-01f,  2.08832577e-01f,
      -3.80776495e-01f, -1.19377777e-01f, -2.06261396e-01f,
      -2.29968596e+00f, -3.19733167e+00f,  4.12264168e-01f},
     { 2.10351646e-02f, -8.14160109e-02f, -1.38926983e-01f,
      -3.04925084e-01f,  9.73871052e-02f, -3.08652014e-01f,
      -1.19096205e-01f,  2.55605370e-01f, -1.94084927e-01f},
     {-9.20104235e-02f, -4.34091985e-02f, -3.25461477e-01f,
      -2.59938538e-02f, -3.56670618e-02f, -2.33437702e-01f,
       1.84952021e-02f, -2.37167567e-01f, -1.98466271e-01f},
     { 3.77891287e-02f, -1.41012460e-01f,  9.92085487e-02f,
       4.27701473e-01f,  3.30760986e-01f, -5.25441706e-01f,
       8.34708810e-01f,  1.75935049e-02f, -7.89435506e-01f},
     { 5.45598157e-02f, -4.91183847e-01f,  3.82780731e-01f,
      -2.34775975e-01f,  1.23290317e-02f,  5.79094231e-01f,
       2.74773240e+00f,  6.63134813e-01f,  7.92681575e-02f}};
  float layer_1_bias[24] = 
    { 0.07245271f, -0.17032284f,  0.03332921f, -0.51800644f,  0.09215787f,
     -0.3023607f ,  0.8202663f ,  0.56285465f,  0.24442022f,  0.16603865f,
     -0.9463791f ,  1.0399476f , -0.2339435f ,  0.07585597f,  2.997388f  ,
     -0.7557977f , -0.08628041f, -0.26461753f, -0.37879473f,  0.37767094f,
      0.09888637f,  0.0759401f ,  0.49256304f,  0.7478588f}; 
  float layer_1_output[24];
  for (int64_t i = 0; i < 24; i++) {
    layer_1_output[i] = layer_1_weight[i][0] * total_height +
                        layer_1_weight[i][1] * total_width + 
                        layer_1_weight[i][2] * total_depth + 
                        layer_1_weight[i][3] * pooled_height +
                        layer_1_weight[i][4] * pooled_width +
                        layer_1_weight[i][5] * pooled_depth +
                        layer_1_weight[i][6] * pool_height +
                        layer_1_weight[i][7] * pool_width +
                        layer_1_weight[i][8] * pool_depth +
                        layer_1_bias[i];
    if (layer_1_output[i] < 0) {
      layer_1_output[i] = 0;
    }
  }//for
  float layer_2_weight[18][24] = 
    {{ 3.25579375e-01f, -1.43363506e-01f,  2.91126639e-01f,
       8.70179292e-03f,  1.45740807e-01f,  6.25467420e-01f,
       1.26648888e-01f,  2.02801764e-01f,  1.59520030e-01f,
      -4.08159107e-01f, -1.09892118e+00f,  2.06926405e-01f,
       5.47469556e-02f, -1.07619211e-01f, -2.87918061e-01f,
       3.40168387e-01f, -4.36676182e-02f,  1.65376484e-01f,
      -3.65924478e-01f,  2.91361868e-01f,  1.19412690e-01f,
      -8.39510486e-02f,  7.66147003e-02f, -5.00059068e-01f},
     {-3.09232980e-01f, -1.37700543e-01f,  2.35218272e-01f,
      -3.08380984e-02f,  1.70388132e-01f,  7.55291820e-01f,
      -1.38875574e-01f,  1.21041216e-01f,  6.67028725e-01f,
      -2.53519624e-01f,  8.37031245e-01f,  5.50382957e-02f,
       1.10191911e-01f, -3.60269248e-02f, -3.60205799e-01f,
       3.76392454e-01f, -1.04238081e+00f, -1.33523121e-01f,
      -4.34185386e-01f, -1.43906253e-03f,  1.26164556e-02f,
       6.53965473e-02f, -1.51397763e-02f,  2.33571097e-01f},
     {-1.46996416e-02f,  9.32471678e-02f, -8.47202074e-03f,
      -2.41985559e-01f,  1.27554238e-01f, -1.35598451e-01f,
      -1.74137384e-01f, -6.16661645e-03f, -8.13471526e-02f,
       6.75069764e-02f, -1.20747283e-01f, -3.96600217e-02f,
       3.03606689e-03f,  4.92227972e-02f, -1.33338228e-01f,
       5.08449711e-02f,  4.90823314e-02f, -1.58856750e-01f,
      -1.49854690e-01f, -3.47922966e-02f, -4.25842255e-02f,
       1.86898440e-01f, -1.77283853e-01f,  1.50340796e-01f},
     {-1.42046437e-01f, -9.51200798e-02f, -5.27066529e-01f,
      -2.02464670e-01f,  1.76374793e-01f, -6.43707097e-01f,
       2.77709160e-02f, -3.30471754e-01f, -2.77649611e-02f,
       4.37590629e-01f,  2.78118014e-01f,  5.35892546e-01f,
      -1.17371202e-01f, -4.87322807e-02f, -5.60680509e-01f,
      -7.17674270e-02f, -2.05091670e-01f,  1.50647223e-01f,
      -4.61828662e-04f, -9.73732322e-02f,  1.45116389e-01f,
      -1.70713723e-01f, -1.57550368e-02f, -1.01773310e+00f},
     { 3.51592511e-01f, -2.53087163e-01f,  2.50908196e-01f,
       1.65608823e-01f, -8.51340666e-02f,  6.20210707e-01f,
      -3.28132063e-01f,  3.97074461e-01f,  4.73497659e-01f,
      -5.24955630e-01f, -3.72119218e-01f,  2.73004830e-01f,
      -1.03070140e-02f,  1.32160038e-02f,  2.60990888e-01f,
       2.40020394e-01f, -1.14903820e+00f,  1.40207440e-01f,
      -1.30430102e-01f,  4.99394350e-02f, -1.37217239e-01f,
      -1.33530974e-01f,  4.41660956e-02f, -2.82477528e-01f},
     {-2.16649622e-01f, -5.53433061e-01f, -8.93768482e-03f,
      -6.32673860e-01f, -1.75256863e-01f,  5.99353194e-01f,
       1.50826365e-01f,  9.99259531e-01f,  2.78158803e-02f,
       2.13626004e-03f, -5.29637873e-01f,  8.34616840e-01f,
       7.03365505e-02f, -2.22517401e-02f,  1.39930636e-01f,
      -4.07026798e-01f, -7.44533241e-01f,  1.01966083e-01f,
      -6.42009452e-02f, -4.20810640e-01f, -1.36671975e-01f,
      -7.85837024e-02f, -2.15396062e-01f,  1.50120050e-01f},
     {-3.24216425e-01f, -1.16558103e-02f,  2.81447619e-01f,
       6.84683472e-02f, -3.64668816e-02f,  6.94456518e-01f,
      -1.66115344e-01f, -1.98925659e-02f,  3.05160135e-01f,
      -2.26922646e-01f,  1.21507895e+00f, -4.98768032e-01f,
      -2.38120556e-02f, -2.26864219e-03f, -4.47534412e-01f,
       4.15572196e-01f,  1.60578102e-01f,  1.95816606e-01f,
      -3.91342819e-01f, -2.39947647e-01f, -1.93307310e-01f,
      -1.45235926e-01f, -1.85608745e-01f,  5.87138772e-01f},
     { 8.23349580e-02f,  1.07204709e-02f, -1.32537723e-01f,
      -1.68501124e-01f, -1.60431266e-02f, -5.92740029e-02f,
       9.96957533e-04f, -7.05506727e-02f,  7.58752078e-02f,
      -3.73780653e-02f,  5.12917787e-02f,  1.53271407e-02f,
      -8.69580060e-02f,  1.58401787e-01f, -1.62573174e-01f,
      -2.00308084e-01f, -1.89347342e-01f, -8.01983848e-02f,
      -7.83697665e-02f,  1.13473803e-01f,  3.97579670e-02f,
      -9.40723643e-02f, -1.21766657e-01f, -1.27446204e-01f},
     {-1.87878966e-01f, -8.20456371e-02f, -7.74592608e-02f,
      -1.04678825e-01f, -1.42413124e-01f, -1.46764770e-01f,
      -9.11001265e-02f, -3.30407023e-02f,  1.20729446e-01f,
      -1.57779992e-01f, -1.72130764e-03f,  1.77877426e-01f,
       7.09260404e-02f,  1.85215920e-01f, -1.90215886e-01f,
       2.52883732e-02f, -1.17826968e-01f,  7.24883378e-03f,
      -1.17555775e-01f,  6.98681772e-02f, -1.20113902e-01f,
      -1.90116569e-01f, -1.14500359e-01f,  6.80078566e-02f},
     { 5.98913096e-02f, -4.90788698e-01f,  3.54912996e-01f,
      -3.04605141e-02f,  1.47488594e-01f,  5.54586291e-01f,
       7.10668862e-02f,  7.95825779e-01f,  1.24558881e-01f,
      -4.57609177e-01f,  7.77798653e-01f, -6.13706470e-01f,
       1.81260586e-01f, -7.59455413e-02f, -1.27487451e-01f,
       1.34991154e-01f,  5.04741907e-01f, -1.60926044e-01f,
      -1.67974606e-02f,  2.42200255e-01f, -1.17304042e-01f,
       6.95458651e-02f, -9.90122184e-02f, -2.08844900e-01f},
     { 4.43800151e-01f, -7.94112980e-02f,  3.28762740e-01f,
      -3.22961435e-02f,  1.73567444e-01f,  5.98389089e-01f,
       1.91843566e-02f,  3.34660783e-02f, -1.64856128e-02f,
      -3.57722700e-01f, -2.34204575e-01f,  3.48274797e-01f,
      -8.31513703e-02f, -1.30962491e-01f, -9.14340615e-01f,
       3.64279747e-01f,  4.97707129e-02f,  1.27601743e-01f,
      -3.65876138e-01f,  2.13202581e-01f,  9.53516662e-02f,
      -1.40080482e-01f,  4.23965156e-02f, -6.76112592e-01f},
     { 7.91603625e-02f, -1.19425789e-01f, -2.43775845e-02f,
       3.23729659e-03f, -1.09774768e-02f,  1.36542872e-01f,
       1.64479911e-01f, -1.28639296e-01f, -1.47093281e-01f,
      -1.73581883e-01f, -1.00426808e-01f,  3.74512970e-02f,
      -1.27387598e-01f,  1.09096229e-01f, -1.57243669e-01f,
       2.52539702e-02f,  1.50189444e-01f,  6.28880858e-02f,
       1.49470521e-02f,  3.32546420e-02f, -1.45456970e-01f,
      -1.02996826e-03f, -1.17249615e-01f, -2.00911798e-02f},
     { 1.15998000e-01f, -4.46091413e-01f,  5.46111949e-02f,
       7.93148130e-02f,  1.78622395e-01f,  4.35426921e-01f,
      -1.08058363e-01f,  6.42945349e-01f,  2.41041929e-01f,
      -4.15860265e-01f,  7.80571699e-01f, -3.93925518e-01f,
      -7.02788085e-02f,  1.94254220e-02f, -1.26532003e-01f,
       3.32491457e-01f,  5.16936839e-01f, -1.23047821e-01f,
      -1.61967024e-01f,  1.63455203e-01f, -7.74515271e-02f,
      -1.70872435e-01f, -2.24204808e-01f, -1.24687552e-01f},
     { 4.38408077e-01f, -2.12102920e-01f,  2.08436120e-02f,
      -1.30177632e-01f, -1.49440750e-01f, -2.71422565e-02f,
       4.44359392e-01f,  5.36480725e-01f, -4.95767236e-01f,
      -1.18874744e-01f, -1.45481184e-01f,  1.56909585e+00f,
       3.85473520e-02f,  1.63013637e-01f,  1.39604688e-01f,
      -8.05395663e-01f, -9.94463712e-02f, -1.41212881e-01f,
       1.27888069e-01f, -3.82867344e-02f, -1.69497415e-01f,
       8.49550068e-02f,  9.75115299e-02f, -3.06227565e-01f},
     { 3.63943011e-01f, -5.95641553e-01f, -5.18703938e-01f,
      -2.90537924e-01f,  6.35516047e-02f, -9.22656655e-01f,
       2.72293538e-01f,  6.53651059e-01f, -9.13953558e-02f,
       3.19077261e-03f,  2.00095564e-01f,  1.53240156e+00f,
      -1.16094671e-01f, -8.30657184e-02f,  3.76244307e-01f,
      -4.93049264e-01f,  3.09002995e-01f,  7.98221529e-02f,
      -2.95217428e-02f, -1.84656763e+00f, -3.32381427e-02f,
       1.20233893e-01f,  1.73625186e-01f, -2.23886073e-01f},
     {-5.33011705e-02f,  1.56638831e-01f,  2.05632687e-01f,
      -1.52538300e-01f,  7.67476261e-02f,  4.71502721e-01f,
      -1.26955080e+00f,  4.70376521e-01f, -1.71605006e-01f,
      -9.35368389e-02f,  1.12155282e+00f,  2.41585016e-01f,
      -7.58866519e-02f,  1.76631659e-01f,  4.25829552e-02f,
       8.66330639e-02f,  2.05610469e-02f, -1.32992446e-01f,
       1.31201595e-01f, -3.65580350e-01f, -2.80796587e-02f,
       1.54423624e-01f, -3.13763833e+00f,  9.35740247e-02f},
     {-1.30747125e-01f, -1.86169356e-01f,  5.74030519e-01f,
       1.04887962e-01f, -1.51518986e-01f, -4.81547177e-01f,
       1.79648191e-01f,  5.83938301e-01f,  9.82384980e-02f,
      -4.62074876e-01f,  1.05995786e+00f, -8.83224666e-01f,
      -1.05079211e-01f, -7.99378157e-02f, -6.77995265e-01f,
       4.07487720e-01f,  1.13807738e+00f,  1.02636635e-01f,
      -7.78916627e-02f, -3.27958941e-01f, -1.19272061e-01f,
       9.66027975e-02f, -1.68073341e-01f,  3.44389617e-01f},
     { 6.28027469e-02f, -3.89415592e-01f,  5.09455800e-01f,
      -5.91972619e-02f, -1.67251080e-01f,  4.51156437e-01f,
       3.02566644e-02f,  4.52512324e-01f,  2.19412446e-01f,
      -4.16285634e-01f,  2.73217946e-01f, -8.91215324e-01f,
      -1.88148022e-01f,  8.18876028e-02f,  3.29352766e-01f,
       6.46478683e-03f,  3.87815684e-01f,  6.14425242e-02f,
       5.77934086e-02f,  2.02642858e-01f,  9.71483588e-02f,
       1.90722704e-01f, -2.74714738e-01f, -2.56136447e-01f}};
  float layer_2_bias[18] = 
    { 0.05490978f, -0.5855493f,   0.11810035f, -0.28029796f, -0.207425f,
      0.86734784f, -0.89562637f, -0.16912879f,  0.15428719f, -0.37533942f,
     -0.323099f,   -0.23854022f, -0.46981212f,  1.0469478f,   0.8655093f,
     -0.49080566f, -0.82833606f, -0.28620005f};
  float layer_2_output[18];
  for (int64_t i = 0; i < 18; i++) {
    layer_2_output[i] = 0;
    for (int64_t j = 0; j < 24; j++) {
      layer_2_output[i] += layer_2_weight[i][j] * layer_1_output[j];
    }
    layer_2_output[i] += layer_2_bias[i];
    if (layer_2_output[i] < 0) {
      layer_2_output[i] = 0;
    }
  }
  float layer_3_weight[18] =
    {-0.16637668f, -0.22794755f, -0.00432279f,  0.3353056f , -0.1794299f ,
      0.79640424f, -0.15697755f,  0.19460405f, -0.13104641f, -0.2885915f ,
     -0.445572f  , -0.14337765f, -0.44539693f,  1.243302f  ,  0.90636724f,
     -0.14549503f, -0.26912656f, -0.30170223f};
  float layer_3_bias = 1.3415898f;
  float layer_3_output = 0.0f;
  for (int64_t i = 0; i < 18; i++) {
    layer_3_output += layer_2_output[i] * layer_3_weight[i];
  }
  layer_3_output += layer_3_bias;
  float sigmoid = 1.0f / (1 + std::pow(2.718f, -layer_3_output));
  return std::round(sigmoid) > 0.9f;
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
      context->Output(1, output_dims) ||
      pool_size.size() == 1 &&
      !Optimizable1D(x_shape[2] + pads[0] + pads[1], output_dims[2], pool_size[0]) ||
      pool_size.size() == 2 &&
      !Optimizable2D(x_shape[2] + pads[0] + pads[2],
                     x_shape[3] + pads[1] + pads[3],
                     output_dims[2], output_dims[3],
                     pool_size[0], pool_size[1]) ||
      pool_size.size() == 3 &&
      !Optimizable3D(x_shape[2] + pads[0] + pads[3],
                     x_shape[3] + pads[1] + pads[4],
                     x_shape[4] + pads[2] + pads[5],
                     output_dims[2], output_dims[3], output_dims[4],
                     pool_size[0], pool_size[1], pool_size[2])) {

    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Arguments are not optimizable.");
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
