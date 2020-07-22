// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/nn/conv.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

// ACL
#include "arm_compute/core/TensorInfo.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

#ifdef ACL_1902
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayer3x3Kernel.h"
#endif
#if defined(ACL_1905) || defined(ACL_1908)
#include "arm_compute/runtime/NEON/functions/assembly/NEDepthwiseConvolutionAssemblyDispatch.h"
#endif

#define CONV_ACL
#undef DEPTHWISE_CPU

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T>
thread_local std::map<OpKernel*, ACLNEConv> Conv<T>::convLayers;

template <typename T>
arm_compute::TensorShape Conv<T>::ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const {
  arm_compute::TensorShape shape = arm_compute::TensorShape(kernel->info()->tensor_shape());
  shape[2] = shape[2] * shape[3];
  shape[3] = 1;

  return shape;
}

#ifdef CONV_ACL
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  ACLNEConv* pConv;
  ConvLayersIterator it = Conv::convLayers.find((OpKernel*)this);
  if (it != Conv::convLayers.end()) {
    pConv = &it->second;
    if (pConv->isDepthwiseCPU == true) {
      Status s = onnxruntime::Conv<T>::Compute(context);
      return s;
    }
  }

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];

  LOGS_DEFAULT(VERBOSE) << "Conv ACL:";  
  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
  if (B != nullptr) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();

  if (X->Shape().NumDimensions() != PREF_DIM) {
    LOGS_DEFAULT(WARNING) << "ACL does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    Status s = onnxruntime::Conv<T>::Compute(context);
    return s;
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
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str();

  arm_compute::ActivationLayerInfo::ActivationFunction acl_activ_func;
  bool acl_activ_enabled = false;

  if (activation_type == "Relu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Relu fused implementation";
  } else if (activation_type == "LeakyRelu") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-LeakyRelu fused implementation";
  } else if (activation_type == "Tanh") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Tanh fused implementation";
  } else if (activation_type == "Sigmoid") {
    acl_activ_func = arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
    acl_activ_enabled = true;
    LOGS_DEFAULT(VERBOSE) << "ACL Conv-Sigmoid fused implementation";
  } else if (!activation_type.empty()) {
    ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_type);
  }

  if (it == Conv::convLayers.end()) {

    auto mm_layer = ACLCreateMemoryManager();

    ACLNEConv tconv;
    tconv.mm_layer = std::move(mm_layer);

    tconv.in = std::make_shared<arm_compute::Tensor>();
    tconv.k = std::make_shared<arm_compute::Tensor>();
    if (B != nullptr)
      tconv.b = std::make_shared<arm_compute::Tensor>();
    tconv.out = std::make_shared<arm_compute::Tensor>();

    tconv.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape(), PREF_DIM), arm_compute::Format::F32));
    tconv.k->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));
    if (B != nullptr) {
      tconv.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
    }
    tconv.out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y->Shape(), PREF_DIM), arm_compute::Format::F32));

    const arm_compute::DataLayout data_layout = tconv.in->info()->data_layout();
    const int idx_channel = arm_compute::get_data_layout_dimension_index(data_layout, arm_compute::DataLayoutDimension::CHANNEL);
    bool isDepthwise = (1 == tconv.k->info()->tensor_shape()[idx_channel]);
    tconv.isDepthwiseCPU = isDepthwise;

    std::vector<int64_t> aclStrides(2);
    aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
    aclStrides[1] = strides[0];

    std::vector<int64_t> aclPads(4);
    // The pad order in acl is: pad_left, pad_right, pad_top, pad_bottom
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

    LOGS_DEFAULT(VERBOSE) << "padding: {" << aclPads[0] << "," << aclPads[1] << "," << aclPads[2] << "," << aclPads[3] << "}";
    LOGS_DEFAULT(VERBOSE) << "strides: {" << aclStrides[0] << "," << aclStrides[1] << "}";


    if (isDepthwise) {
      LOGS_DEFAULT(VERBOSE) << "Depthwise convolution";
#ifdef DEPTHWISE_CPU
      Status s = onnxruntime::Conv<T>::Compute(context);
      std::pair<ConvLayersIterator, bool> ret;
      ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
      return s;
#else
      tconv.k->info()->set_tensor_shape(ACLReshapeWeightsDepthwise(tconv.k.get()));

      // in the configure function for NEDepthwiseConvolutionLayer3x3, there is a separation based on the optimization
#ifdef ACL_1902
      bool optimizable =
          arm_compute::NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(tconv.in->info()->tensor_shape(),
                                                                                             aclPadStride,
                                                                                             tconv.in->info()->data_type(),
                                                                                             1 /* depth multiplier */,
                                                                                             tconv.in->info()->data_layout());
#endif
#if defined(ACL_1905) || defined(ACL_1908)
      bool optimizable =
          arm_compute::NEDepthwiseConvolutionAssemblyDispatch::is_optimized_supported(tconv.in->info(),
                                                                                      tconv.k->info(),
                                                                                      aclPadStride,
                                                                                      1 /* depth multiplier */,
                                                                                      arm_compute::Size2D(aclDilation0, dilations[0]));
#endif
      if (optimizable) {
        LOGS_DEFAULT(VERBOSE) << "ACL optimized depthwise convolution";
#if defined(ACL_1902) || defined(ACL_1905)
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer3x3>();
#endif
#ifdef ACL_1908
        auto layer = std::make_shared<arm_compute::NEDepthwiseConvolutionLayerOptimized>();
#endif
#ifdef ACL_1902
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride, 1 /* depth multiplier */,
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo());
#endif
#if defined(ACL_1905) || defined(ACL_1908)
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride, 1 /* depth multiplier */,
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                         arm_compute::Size2D(aclDilation0, dilations[0]));
#endif
        tconv.layer = std::move(layer);
        tconv.isDepthwiseCPU = false;
      } else {
        LOGS_DEFAULT(VERBOSE) << "CPU depthwise convolution";
        Status s = onnxruntime::Conv<T>::Compute(context);
        std::pair<ConvLayersIterator, bool> ret;
        ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
        return s;
      }
#endif  //DEPTHWISE_CPU
    } else {
      if(tconv.k->info()->tensor_shape()[0] == 1 && tconv.k->info()->tensor_shape()[1] == 1) {
        LOGS_DEFAULT(VERBOSE) << "CPU pointwise convolution";
        Status s = onnxruntime::Conv<T>::Compute(context);
        return s;
      } else {
        if(tconv.k->info()->tensor_shape()[0] == 9 && tconv.k->info()->tensor_shape()[1] == 9) {
          LOGS_DEFAULT(WARNING) << "9x9 DirectConvolution does not have an implementation in NCHW layout; defaulting to cpu implementation";
          Status s = onnxruntime::Conv<T>::Compute(context);
          return s;
        }
        LOGS_DEFAULT(VERBOSE) << "ACL 2D convolution";
        auto layer = std::make_shared<arm_compute::NEConvolutionLayer>(mm_layer);
        layer->configure(tconv.in.get(), tconv.k.get(), (B != nullptr) ? tconv.b.get() : nullptr, tconv.out.get(),
                         aclPadStride,
                         arm_compute::WeightsInfo(), arm_compute::Size2D(aclDilation0, dilations[0]),
                         acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                         false, conv_attrs_.group);
        tconv.layer = std::move(layer);
      }
    }

    tconv.out->info()->set_format(tconv.in->info()->format());

    std::pair<ConvLayersIterator, bool> ret;
    ret = Conv::convLayers.insert(std::pair<OpKernel*, ACLNEConv>((OpKernel*)this, tconv));
    pConv = &ret.first->second;

    ACLPrintTensorShape("X", *tconv.in.get());
    ACLPrintTensorShape("Y", *tconv.out.get());

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

  arm_compute::Allocator alloc_mm{};
  pConv->mm_layer->populate(alloc_mm, 1);
  pConv->layer->run();
  pConv->mm_layer->clear();

  pConv->in->allocator()->free();
  pConv->k->allocator()->free();
  if (B != nullptr)
    pConv->b->allocator()->free();
  pConv->out->allocator()->free();

  LOGS_DEFAULT(VERBOSE) << std::endl;

  return Status::OK();
}
#else
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
  if (B != nullptr)
    LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();

  LOGS_DEFAULT(VERBOSE) << std::endl;

  Status s = onnxruntime::Conv<T>::Compute(context);
  return s;
}
#endif

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace acl
}  // namespace onnxruntime
