// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "arm_compute/core/ITensorPack.h"
#include "src/cpu/operators/CpuConv2d.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"


#define CONV_ACL
#undef DEPTHWISE_CPU

#define PREF_DIM 4

namespace onnxruntime {

namespace acl {

struct ConvConfig {
  bool isQuantized;
  bool is_channels_last;
  bool isDepthwise;
  TensorShape inShapeIn;
  TensorShape kShapeIn;
  const std::string *inType;
  const std::string *kType;
};

Status ParseConv(const onnxruntime::Node& node, ConvConfig &config) {
  onnxruntime::ProtoHelperNodeContext ctx(node);
  onnxruntime::OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  const auto inputDefs = node.InputDefs();

  config.isQuantized = node.OpType() == "QLinearConv";

  if (config.isQuantized) {
    TensorShape scaleShape;
    ORT_RETURN_IF_ERROR(GetArgShape(inputDefs[4], scaleShape));
    ORT_RETURN_IF(scaleShape.Size() > 1, "ACL execution provider does not support per-channel quantization");
  }

  config.is_channels_last = node.OpType() == "NhwcConv";
  if (!config.is_channels_last) {
    int64_t cl_ret = 0;
    attrs.GetAttr("channels_last", &cl_ret);
    config.is_channels_last = (bool) cl_ret;
  }

  int64_t group = 1;
  attrs.GetAttr("group", &group);

  const NodeArg *kDef = inputDefs[config.isQuantized? 3 : 1];

  ORT_RETURN_IF_ERROR(GetArgShape(inputDefs[0], config.inShapeIn));
  ORT_RETURN_IF_ERROR(GetArgShape(kDef, config.kShapeIn));

  ORT_RETURN_IF(config.kShapeIn.NumDimensions() > 4, "ACL execution provider supports 1D and 2D Conv only");

  config.inType = inputDefs[0]->Type();
  config.kType = kDef->Type();
  const bool mixedType = config.inType != config.kType;

  config.isDepthwise = group > 1;
  if (config.isDepthwise) {
    const size_t channels = config.inShapeIn[config.is_channels_last? config.inShapeIn.NumDimensions() - 1 : 1];
    ORT_RETURN_IF(group != channels, "ACL does not support grouping unless group == channels");
    ORT_RETURN_IF(mixedType, "ACL does not support mixed input types for depthwise Conv");
  }

  return Status::OK();
}

Status ValidateConv(const onnxruntime::Node& node) {
  ConvConfig config;
  return ParseConv(node, config);
}

arm_compute::TensorShape Conv::ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const {
  arm_compute::TensorShape shape = arm_compute::TensorShape(kernel->info()->tensor_shape());
  shape[2] = shape[2] * shape[3];
  shape[3] = 1;

  return shape;
}

Conv::Conv(const OpKernelInfo& info) : onnxruntime::OpKernel(info), conv_attrs_(info) {
  provider_ = (const_cast<ACLExecutionProvider*>(
      static_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));

  ConvConfig config;
  ORT_THROW_IF_ERROR(ParseConv(OpKernel::Node(), config));
  isQuantized = config.isQuantized;
  is_channels_last = config.is_channels_last;

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  has_bias = isQuantized? (num_inputs == 9) : (num_inputs == 3);

  const Tensor *tmp = nullptr;
  const bool kIsConst = info.TryGetConstantInput(1, &tmp);
  ORT_ENFORCE(kIsConst, "ACL does not support Conv with mutable weights");

  in = std::make_shared<arm_compute::Tensor>();
  k = std::make_shared<arm_compute::Tensor>();
  if (has_bias)
    b = std::make_shared<arm_compute::Tensor>();
  out = std::make_shared<arm_compute::Tensor>();

  const arm_compute::DataLayout data_layout = is_channels_last?
      arm_compute::DataLayout::NHWC : arm_compute::DataLayout::NCHW;

  TensorShape inShape = config.inShapeIn;
  if (is_channels_last && config.inShapeIn.NumDimensions() < 4) {
    inShape = TensorShape({config.inShapeIn[0], config.inShapeIn[1], 1, config.inShapeIn[2]});
  }

  arm_compute::DataType inType = ACLDataType(*config.inType);
  in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(inShape, PREF_DIM), 1, inType, data_layout));

  arm_compute::DataType kType = ACLDataType(*config.kType);

  TensorShapeVector kShapeVec = config.kShapeIn.AsShapeVector();
  while(kShapeVec.size() < 4) {
    kShapeVec.push_back(1);
  }

  const TensorShape kShape = is_channels_last?
      TensorShape({kShapeVec[0], kShapeVec[2], kShapeVec[3], kShapeVec[1]}) : TensorShape(kShapeVec);

  k->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(kShape), 1, kType, data_layout));

  TensorShape bShape;
  if (has_bias) {
    const Tensor *bias = nullptr;
    const bool biasIsConst = info.TryGetConstantInput(isQuantized? 8 : 2, &bias);
    ORT_ENFORCE(biasIsConst, "ACL does not support Conv with mutable bias");

    const auto bDef = OpKernel::Node().InputDefs()[isQuantized? 8 : 2];
    ORT_THROW_IF_ERROR(GetArgShape(bDef, bShape));
    arm_compute::DataType bType = ACLDataType(*bDef->Type());
    b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(bShape), 1, bType, data_layout));

    const void* b_data = bias->DataRaw();
    ORT_THROW_IF_ERROR(ACLImportMemory(b->allocator(), (void*)b_data, 0));
  }

  ORT_THROW_IF_ERROR(GetArgShape(OpKernel::Node().OutputDefs()[0], outShape));
  TensorShape outShapeACL = outShape;
  if (is_channels_last && outShape.NumDimensions() < 4) {
    outShapeACL = TensorShape({outShape[0], outShape[1], 1, outShape[2]});
  }

  out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(outShapeACL, PREF_DIM), 1, inType, data_layout));

  if (isQuantized) {
    ORT_THROW_IF_ERROR(LoadQuantizationInfo(info, in.get(), 1, 2, false));
    ORT_THROW_IF_ERROR(LoadQuantizationInfo(info, k.get(), 4, 5, false));
    ORT_THROW_IF_ERROR(LoadQuantizationInfo(info, out.get(), 6, 7, false));
  }

  LOGS_DEFAULT(VERBOSE) << "Conv ACL:";
  LOGS_DEFAULT(VERBOSE) << "X " << inShape.ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << config.kShapeIn.ToString().c_str();
  if (has_bias) {
    LOGS_DEFAULT(VERBOSE) << "B " << bShape.ToString().c_str();
  }

  ORT_THROW_IF_ERROR(conv_attrs_.ValidateInputShape(config.inShapeIn, config.kShapeIn, config.is_channels_last));

  TensorShapeVector kernel_shape;
  ORT_THROW_IF_ERROR(conv_attrs_.ComputeKernelShape(config.kShapeIn, kernel_shape));

  ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  TensorShape input_shape = config.inShapeIn.Slice(2);
  TensorShapeVector out_shape;
  ORT_THROW_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(
      input_shape, kernel_shape, strides, dilations,
      pads, out_shape));

  LOGS_DEFAULT(VERBOSE) << "Y " << outShape.ToString().c_str();

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

  const size_t idx_channel = arm_compute::get_data_layout_dimension_index(data_layout, arm_compute::DataLayoutDimension::CHANNEL);
  isDepthwiseCPU = config.isDepthwise;

  std::vector<int64_t> aclStrides(2);
  aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
  aclStrides[1] = strides[0];

  std::vector<int64_t> aclPads(4);
  // The pad order in acl is: pad_left, pad_right, pad_top, pad_bottom
  if (pads.size() == 2) {
    if (strides.size() == 1) {
      aclPads[0] = 0;
      aclPads[1] = 0;
      aclPads[2] = pads[0];
      aclPads[3] = pads[1];
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

  arm_compute::PadStrideInfo aclPadStride = arm_compute::PadStrideInfo(
      (unsigned int) aclStrides[0], (unsigned int) aclStrides[1],
      (unsigned int) aclPads[0], (unsigned int) aclPads[1],
      (unsigned int) aclPads[2], (unsigned int) aclPads[3], arm_compute::DimensionRoundingType::FLOOR);
  size_t aclDilation0 = (dilations.size() == 2) ? dilations[1] : 1;

  LOGS_DEFAULT(VERBOSE) << "padding: {" << aclPads[0] << "," << aclPads[1] << "," << aclPads[2] << "," << aclPads[3] << "}";
  LOGS_DEFAULT(VERBOSE) << "strides: {" << aclStrides[0] << "," << aclStrides[1] << "}";

  if (config.isDepthwise) {
    LOGS_DEFAULT(VERBOSE) << "Depthwise convolution";
    k->info()->set_tensor_shape(ACLReshapeWeightsDepthwise(k.get()));
    auto dl = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer>();
    dl->configure(in.get(), k.get(), (has_bias) ? b.get() : nullptr, out.get(),
                  aclPadStride, 1 /* depth multiplier */,
                  acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                  arm_compute::Size2D(aclDilation0, dilations[0]));
    depthwise_layer = std::move(dl);
    isDepthwiseCPU = false;
  } else {
    LOGS_DEFAULT(VERBOSE) << "ACL 2D convolution";
    auto cl = std::make_shared<arm_compute::cpu::CpuConv2d>();
    cl->configure(in->info(), k->info(), (has_bias) ? b->info() : nullptr, out->info(),
                  aclPadStride,
                  arm_compute::WeightsInfo(), arm_compute::Size2D(aclDilation0, dilations[0]),
                  acl_activ_enabled ? arm_compute::ActivationLayerInfo(acl_activ_func, conv_attrs_.alpha) : arm_compute::ActivationLayerInfo(),
                  provider_->info.enable_fast_math, (unsigned int) conv_attrs_.group);
    conv_layer = std::move(cl);

    memory_group = arm_compute::MemoryGroup(provider_->memory_manager);
    run_pack = {{arm_compute::ACL_SRC_0, in.get()}, {arm_compute::ACL_SRC_1, k.get()},
                {arm_compute::ACL_SRC_2, b.get()}, {arm_compute::ACL_DST, out.get()}};
    prep_pack = {{arm_compute::ACL_SRC_1, k.get()}, {arm_compute::ACL_SRC_2, b.get()}};

    PopulateWorkspace(conv_layer->workspace(), workspace, memory_group, run_pack, prep_pack);
  }

  ACLPrintTensorShape("X", *in.get());
  ACLPrintTensorShape("Y", *out.get());
}

#ifdef CONV_ACL
Status Conv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
        /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) {

  is_packed = false;
  if (isQuantized? (input_idx != 3) : ( input_idx != 1)) {
    return Status::OK();
  }

  if (!workspace.persistent_tensors.empty()) {
    size_t packedSize = 0;
    size_t alignment = 0;
    GetPackingInfo(workspace.persistent_tensors, packedSize, alignment);
    auto buffSize = packedSize + alignment;

    pkRaw = IAllocator::MakeUniquePtr<void>(alloc, buffSize, true);
    ORT_RETURN_IF_ERROR(LoadPackedTensors(workspace.persistent_tensors, pkRaw.get(), packedSize, alignment));

    if (prepacked_weights != nullptr) {
      prepacked_weights->buffers_.push_back(std::move(pkRaw));
      prepacked_weights->buffer_sizes_.push_back(buffSize);
    }

    is_packed = true;
  }

  bool free_k = false;
  const void* k_data = tensor.DataRaw();
  if (is_channels_last) {
    TensorShape shape = tensor.Shape();
    if (shape.NumDimensions() < 4) {
      shape = TensorShape({shape[0], shape[1], shape[2], 1});
    }

    arm_compute::Tensor kIn;
    kIn.allocator()->init(arm_compute::TensorInfo(ACLTensorShape(shape), 1,
        k->info()->data_type(), arm_compute::DataLayout::NCHW));
    kIn.info()->set_quantization_info(k->info()->quantization_info());

    ORT_RETURN_IF_ERROR(ACLImportMemory(kIn.allocator(), (void*)k_data, 0));
    k->allocator()->allocate();
    free_k = is_packed;
    is_packed = true;

    arm_compute::NEPermute perm_layer;
    perm_layer.configure(&kIn, k.get(), {2, 0, 1, 3});
    perm_layer.run();
  } else {
    ORT_RETURN_IF_ERROR(ACLImportMemory(k->allocator(), (void*)k_data, 0));
  }

  for (std::unique_ptr<arm_compute::Tensor> &prep_tensor : workspace.prepare_tensors) {
    prep_tensor->allocator()->allocate();
  }

  if (conv_layer) {
    conv_layer->prepare(prep_pack);
  } else {
    depthwise_layer->prepare();
  }

  for (std::unique_ptr<arm_compute::Tensor> &prep_tensor : workspace.prepare_tensors) {
    prep_tensor->allocator()->free();
  }

  if (free_k) {
    k->allocator()->free();
  }

  return Status::OK();
}

Status Conv::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                 int input_idx, /*out*/ bool& used_shared_buffers) {

  used_shared_buffers = false;
  if (isQuantized? (input_idx != 3) : ( input_idx != 1)) {
    return Status::OK();
  }

  if (!workspace.persistent_tensors.empty()) {
    size_t packedSize = 0;
    size_t alignment = 0;
    GetPackingInfo(workspace.persistent_tensors, packedSize, alignment);

    ORT_RETURN_IF_ERROR(LoadPackedTensors(workspace.persistent_tensors, prepacked_buffers[0].get(),
      packedSize, alignment));

    used_shared_buffers = true;
  }

  return Status::OK();
}

Status Conv::Compute(OpKernelContext* context) const {
  provider_->SetThreadPool(context->GetOperatorThreadPool());

  const Tensor* X = context->Input<Tensor>(0);

  Tensor* Y = context->Output(0, outShape);

  const void* x_data = X->DataRaw();
  ORT_RETURN_IF(X->Shape().Size() != 0 && in->info()->has_padding(), "Padded ACL input tensor not supported");
  ORT_RETURN_IF_ERROR(ACLImportMemory(in->allocator(), (void*)x_data, 0));

  void* y_data = Y->MutableDataRaw();
  ORT_RETURN_IF(Y->Shape().Size() != 0 && out->info()->has_padding(), "Padded ACL output tensor not supported");
  ORT_RETURN_IF_ERROR(ACLImportMemory(out->allocator(), (void*)y_data, 0));

  if (conv_layer) {
    arm_compute::MemoryGroupResourceScope scope_mg(const_cast<arm_compute::MemoryGroup&>(memory_group));
    conv_layer->run(const_cast<arm_compute::ITensorPack&>(run_pack));
  } else {
    depthwise_layer->run();
  }

  in->allocator()->free();
  k->allocator()->free();
  out->allocator()->free();

  LOGS_DEFAULT(VERBOSE) << std::endl;

  return Status::OK();
}
#endif

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    MLFloat16,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Conv);

ONNX_OPERATOR_KERNEL_EX(
    NhwcConv,
    kMSDomain,
    1,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    uint8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    int8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    uint8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    int8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);

}  // namespace acl
}  // namespace onnxruntime
