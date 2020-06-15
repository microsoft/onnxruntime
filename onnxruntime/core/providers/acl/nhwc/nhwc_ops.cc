// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/nhwc/nhwc_ops.h"
#include "core/providers/acl/acl_fwd.h"

// NEON
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#define CONV_ACL
#undef DEPTHWISE_CPU

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T>
thread_local std::map<OpKernel*, ACLNEConv> NhwcConv<T>::convLayers;

template <typename T>
thread_local std::map<OpKernel*, ACLNEPool> NhwcPoolBase<T>::poolLayers;

arm_compute::TensorShape ACL_NCHW2NHWC(arm_compute::TensorShape shape) {

    return arm_compute::TensorShape(shape[2], shape[0], shape[1], shape[3]);
}

template <typename T>
Status ReorderInput<T>::Compute(OpKernelContext* context) const {

  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);

  auto* Y = context->Output(0, X_shape);

  auto mm_layer = ACLCreateMemoryManager();

  ACLNEPermute* pPerm;
  ACLNEPermute tperm;
  tperm.mm_layer = std::move(mm_layer);

  tperm.in = std::make_shared<arm_compute::Tensor>();
  tperm.out = std::make_shared<arm_compute::Tensor>();

  tperm.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape(), PREF_DIM), arm_compute::Format::F32));
  tperm.out->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(Y->Shape(), PREF_DIM)), arm_compute::Format::F32));

  tperm.layer = std::make_shared<arm_compute::NEPermute>();

// ONNX NHWC
// ACL CWHN
// move to 2 0 1 3
// new ACL WHCN

  tperm.layer->configure(tperm.in.get(), tperm.out.get(), arm_compute::PermutationVector(2U, 0U, 1U));

  //ToDo cache
  pPerm = &tperm;

  const T* x_data = X->template Data<T>();
  ACLImportMemory(pPerm->in->allocator(), (void*)x_data, X->Shape().Size() * 4);

  T* y_data = Y->template MutableData<T>();
  ACLImportMemory(pPerm->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  arm_compute::Allocator alloc_mm{};
  pPerm->mm_layer->populate(alloc_mm, 1);
  pPerm->layer->run();
  pPerm->mm_layer->clear();

  pPerm->in->allocator()->free();
  pPerm->out->allocator()->free();

  return Status::OK();
}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {

  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);

  auto* Y = context->Output(0, X_shape);

  auto mm_layer = ACLCreateMemoryManager();

  ACLNEPermute* pPerm;
  ACLNEPermute tperm;
  tperm.mm_layer = std::move(mm_layer);

  tperm.in = std::make_shared<arm_compute::Tensor>();
  tperm.out = std::make_shared<arm_compute::Tensor>();

  tperm.in->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(X->Shape(), PREF_DIM)), arm_compute::Format::F32));
  tperm.out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y->Shape(), PREF_DIM), arm_compute::Format::F32));

  tperm.layer = std::make_shared<arm_compute::NEPermute>();

  tperm.layer->configure(tperm.in.get(), tperm.out.get(), arm_compute::PermutationVector(1U, 2U, 0U));

  //ToDo cache
  pPerm = &tperm;

  const T* x_data = X->template Data<T>();
  ACLImportMemory(pPerm->in->allocator(), (void*)x_data, X->Shape().Size() * 4);

  T* y_data = Y->template MutableData<T>();
  ACLImportMemory(pPerm->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  arm_compute::Allocator alloc_mm{};
  pPerm->mm_layer->populate(alloc_mm, 1);
  pPerm->layer->run();
  pPerm->mm_layer->clear();

  pPerm->in->allocator()->free();
  pPerm->out->allocator()->free();

  return Status::OK();
}

template <typename T>
arm_compute::TensorShape NhwcConv<T>::ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const {
  arm_compute::TensorShape shape = arm_compute::TensorShape(kernel->info()->tensor_shape());

  return arm_compute::TensorShape(kernel->info()->tensor_shape()[2] * kernel->info()->tensor_shape()[3],
                                  kernel->info()->tensor_shape()[0],
                                  kernel->info()->tensor_shape()[1],
                                  1);
}


template <typename T>
Status NhwcConv<T>::Compute(OpKernelContext* context) const {

  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];

  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
  if (B != nullptr) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    ORT_NOT_IMPLEMENTED("Only implemented convolution for 4D input. Number of dimensions found: ", X->Shape().NumDimensions());
  }

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;

  arm_compute::ActivationLayerInfo::ActivationFunction acl_activ_func;
  bool acl_activ_enabled = false;

  if (activation_type == "Relu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
    acl_activ_enabled = true;
  } else if (activation_type == "LeakyRelu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU;
    acl_activ_enabled = true;
  } else if (activation_type == "Tanh") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
    acl_activ_enabled = true;
  } else if (activation_type == "Sigmoid") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
    acl_activ_enabled = true;
  } else if (!activation_type.empty()) {
    ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_type);
  }

  ACLNEConv* pConv;
  ConvLayersIterator it = NhwcConv::convLayers.find((OpKernel*)this);
  if (it == NhwcConv::convLayers.end()) {

    auto mm_layer = ACLCreateMemoryManager();

    ACLNEConv tconv;
    tconv.mm_layer = std::move(mm_layer);

    tconv.in = std::make_shared<arm_compute::Tensor>();
    tconv.k = std::make_shared<arm_compute::Tensor>();
    if (B != nullptr)
      tconv.b = std::make_shared<arm_compute::Tensor>();
    tconv.out = std::make_shared<arm_compute::Tensor>();

    bool isDepthwise = (W->Shape()[1] == 1);

    tconv.in->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(X->Shape(), PREF_DIM)), arm_compute::Format::F32));
    if (isDepthwise)
      tconv.k->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape(), PREF_DIM), arm_compute::Format::F32));
    else
      tconv.k->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(W->Shape(), PREF_DIM)), arm_compute::Format::F32));
    if (B != nullptr) {
      tconv.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
    }
    tconv.out->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(Y->Shape(), PREF_DIM)), arm_compute::Format::F32));

    tconv.in.get()->info()->set_data_layout(arm_compute::DataLayout::NHWC);
    tconv.out.get()->info()->set_data_layout(arm_compute::DataLayout::NHWC);
    tconv.k.get()->info()->set_data_layout(arm_compute::DataLayout::NHWC);

    std::vector<int64_t> aclStrides(2);
    aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
    aclStrides[1] = strides[0];

    std::vector<int64_t> aclPads(4);
    if (pads.size() == 2) {
      if (strides.size() == 1) {
        aclPads[0] = 0;
        aclPads[1] = 0;
        aclPads[2] = pads[1];
        aclPads[3] = pads[0];
      } else {
        aclPads[0] = pads[1];
        aclPads[1] = pads[0];
        aclPads[2] = pads[1];
        aclPads[3] = pads[0];
      }
    } else {
      aclPads[0] = pads[1];
      aclPads[1] = pads[3];
      aclPads[2] = pads[0];
      aclPads[3] = pads[2];
    }

    arm_compute::PadStrideInfo aclPadStride = arm_compute::PadStrideInfo(aclStrides[0], aclStrides[1],
                                                                         aclPads[0], aclPads[1], aclPads[2], aclPads[3], arm_compute::DimensionRoundingType::FLOOR);
    unsigned int aclDilation0 = (dilations.size() == 2) ? dilations[1] : 1;

    if (isDepthwise) {
      tconv.k->info()->set_tensor_shape(ACLReshapeWeightsDepthwise(tconv.k.get()));

      //depthwise convolution

      if((W->Shape().GetDims()[2] != 3 || W->Shape().GetDims()[3] != 3)){
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer>();
#ifdef ACL_1902
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                       aclPadStride, 1,
                       acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo());
#else
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                       aclPadStride, 1,
                       acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                       arm_compute::Size2D(aclDilation0, dilations[0]));
#endif
        tconv.layer = std::move(layer);
      } else {
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer3x3>();
#ifdef ACL_1902
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                       aclPadStride, 1,
                       acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo());
#else
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                       aclPadStride, 1,
                       acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                       arm_compute::Size2D(aclDilation0, dilations[0]));
#endif
        tconv.layer = std::move(layer);
      }

    } else {

      //convolution
      auto layer = std::make_shared<arm_compute::NEConvolutionLayer>(mm_layer);
      layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                       aclPadStride,
                       arm_compute::WeightsInfo(), arm_compute::Size2D(aclDilation0, dilations[0]),
                       acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                       false, conv_attrs_.group);
      tconv.layer = std::move(layer);
    }

    tconv.out->info()->set_format(tconv.in->info()->format());

    std::pair<ConvLayersIterator, bool> ret;
    ret = NhwcConv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
    pConv = &ret.first->second;

  } else {
    //TODO: valildate shapes
    pConv = &it->second;
  }

  const T* x_data = X->template Data<T>();
  ACLImportMemory(pConv->in->allocator(), (void*)x_data, X->Shape().Size() * 4);

  const T* k_data = W->template Data<T>();
  ACLImportMemory(pConv->k->allocator(), (void*)k_data, W->Shape().Size() * 4);

  if (B != nullptr) {
    const T* b_data = B->template Data<T>();
    ACLImportMemory(pConv->b->allocator(), (void*)b_data, B->Shape().Size() * 4);
  }

  T* y_data = Y->template MutableData<T>();
  ACLImportMemory(pConv->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  ACLImportMemory(pConv->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  arm_compute::Allocator alloc_mm{};
  pConv->mm_layer->populate(alloc_mm, 1);
  pConv->layer->run();
  pConv->mm_layer->clear();

  pConv->in->allocator()->free();
  pConv->k->allocator()->free();
  if (B != nullptr)
    pConv->b->allocator()->free();
  pConv->out->allocator()->free();

  return Status::OK();
}

template <typename T>
Status NhwcPoolBase<T>::NhwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const auto* X = context->Input<Tensor>(0);

  const auto& x_shape = X->Shape();
  ORT_ENFORCE(x_shape.NumDimensions() == 4);

  arm_compute::Tensor in, out;

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> aclDilations(2);
  aclDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  aclDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    ORT_NOT_IMPLEMENTED("Only implemented pooling for 4D input. Number of dimensions found: ", X->Shape().NumDimensions());
  }

  std::vector<int64_t> pads = PoolBase::pool_attrs_.pads;
  std::vector<int64_t> strides = PoolBase::pool_attrs_.strides;
  std::vector<int64_t> kernel_shape = PoolBase::pool_attrs_.kernel_shape;

  if (PoolBase::pool_attrs_.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = PoolBase::pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  ACLNEPool* pPool;
  PoolLayersIterator it = NhwcPoolBase::poolLayers.find((OpKernel*)this);
  if (it == NhwcPoolBase::poolLayers.end()) {
    auto layer = std::make_shared<arm_compute::NEPoolingLayer>();

    ACLNEPool tpool;
    tpool.in = std::make_shared<arm_compute::Tensor>();
    tpool.out = std::make_shared<arm_compute::Tensor>();

    tpool.in->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(X->Shape(), PREF_DIM)), arm_compute::Format::F32));
    tpool.out->allocator()->init(arm_compute::TensorInfo(ACL_NCHW2NHWC(ACLTensorShape(Y->Shape(), PREF_DIM)), arm_compute::Format::F32));

    tpool.in.get()->info()->set_data_layout(arm_compute::DataLayout::NHWC);
    tpool.out.get()->info()->set_data_layout(arm_compute::DataLayout::NHWC);

    arm_compute::PoolingType pool_type;
    if (PoolBase::op_name_ == "GlobalAveragePool" || PoolBase::op_name_ == "AveragePool")
      pool_type = arm_compute::PoolingType::AVG;
    else if (PoolBase::op_name_ == "GlobalMaxPool" || PoolBase::op_name_ == "MaxPool")
      pool_type = arm_compute::PoolingType::MAX;
    else
      ORT_NOT_IMPLEMENTED("Not implemented type of pooling: ", PoolBase::op_name_);

    if (PoolBase::pool_attrs_.global_pooling) {
      layer->configure(tpool.in.get(), tpool.out.get(), arm_compute::PoolingLayerInfo(pool_type));
    } else {
      std::vector<int64_t> aclStrides(2);
      aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
      aclStrides[1] = strides[0];

      std::vector<int64_t> aclPads(4);
      if (pads.size() == 2) {
        if (strides.size() == 1) {
          aclPads[0] = 0;
          aclPads[1] = 0;
          aclPads[2] = pads[1];
          aclPads[3] = pads[0];
        } else {
          aclPads[0] = pads[1];
          aclPads[1] = pads[0];
          aclPads[2] = pads[1];
          aclPads[3] = pads[0];
        }
      } else {
        aclPads[0] = pads[1];
        aclPads[1] = pads[3];
        aclPads[2] = pads[0];
        aclPads[3] = pads[2];
      }

      arm_compute::PadStrideInfo aclPadStride = arm_compute::PadStrideInfo(aclStrides[0], aclStrides[1],
                                                                           aclPads[0], aclPads[1], aclPads[2], aclPads[3], arm_compute::DimensionRoundingType::FLOOR);

      std::vector<int64_t> aclKernelShape(2);
      aclKernelShape[0] = (kernel_shape.size() > 1) ? kernel_shape[1] : 1;
      aclKernelShape[1] = kernel_shape[0];

      arm_compute::Size2D aclSize(aclKernelShape[0], aclKernelShape[1]);

      bool excludePadding = (pool_type == arm_compute::PoolingType::AVG && PoolBase::pool_attrs_.count_include_pad) ? false : true;
      arm_compute::PoolingLayerInfo pool_info(pool_type, aclSize, aclPadStride, excludePadding);
      layer->configure(tpool.in.get(), tpool.out.get(), pool_info);
    }

    // allocate space for input tensor to accomodate paddings and strides
    tpool.in->allocator()->allocate();

    tpool.layer = std::move(layer);
    std::pair<PoolLayersIterator, bool> ret;
    ret = NhwcPoolBase::poolLayers.insert(std::pair<OpKernel*, ACLNEPool>((OpKernel*)this, tpool));
    pPool = &ret.first->second;
  } else {
    pPool = &it->second;
  }

  const T* x_data = X->template Data<T>();
  arm_compute::Window aclInpuWindow;
  aclInpuWindow.use_tensor_dimensions(pPool->in->info()->tensor_shape());

  arm_compute::Iterator aclInputIt(pPool->in.get(), aclInpuWindow);
  int index = 0;

  // copy input tensor into the larger buffer
  arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        *reinterpret_cast<float*>(aclInputIt.ptr()) = x_data[index];
        index++;
      },
      aclInputIt);

  T* y_data = Y->template MutableData<T>();
  ACLImportMemory(pPool->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  pPool->layer->run();

  return Status::OK();
}

template <typename T>
Status NhwcMaxPool<T>::Compute(OpKernelContext* context) const {
  return NhwcPoolBase<T>::NhwcPool(context, MlasMaximumPooling);
}

template <typename T>
Status NhwcAveragePool<T>::Compute(OpKernelContext* context) const {
  return NhwcPoolBase<T>::NhwcPool(context, PoolBase::pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad
                                                                         : MlasAveragePoolingExcludePad);
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ReorderInput,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ReorderOutput,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcConv<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxPool,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcMaxPool<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalMaxPool,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcMaxPool<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AveragePool,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcAveragePool<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalAveragePool,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcAveragePool<float>);

}  // namespace acl
}  // namespace onnxruntime
