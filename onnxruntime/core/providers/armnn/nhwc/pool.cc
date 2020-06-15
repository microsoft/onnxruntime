// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/armnn/nhwc/pool.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"

#include "armnn/ArmNN.hpp"

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T, typename PoolType>
thread_local std::map<OpKernel*, armnn::NetworkId> NHWCPool<T, PoolType>::poolLayers;

template <typename T, typename PoolType>
armnn::IRuntimePtr NHWCPool<T, PoolType>::run = NHWCPool<T, PoolType>::initRuntime();

armnn::Pooling2dDescriptor createNHWCDescriptor(std::vector<int64_t> pads, std::vector<int64_t> strides, std::vector<int64_t> kernel_shape, armnn::PoolingAlgorithm pool_type, onnxruntime::PoolAttributes pool_attrs){

  std::vector<int64_t> armnnStrides(2);
  armnnStrides[0] = (strides.size() == 2) ? strides[1] : 1;
  armnnStrides[1] = strides[0];

  std::vector<int64_t> armnnKernelShape(2);
  armnnKernelShape[0] = (kernel_shape.size() > 1) ? kernel_shape[1] : 1;
  armnnKernelShape[1] = kernel_shape[0];

  std::vector<int64_t> armnnPads(4);
  if (pads.size() == 2) {
    if (strides.size() == 1) {
      armnnPads[0] = 0;
      armnnPads[1] = 0;
      armnnPads[2] = pads[1];
      armnnPads[3] = pads[0];
    } else {
      armnnPads[0] = pads[1];
      armnnPads[1] = pads[0];
      armnnPads[2] = pads[1];
      armnnPads[3] = pads[0];
    }
  } else {
    armnnPads[0] = pads[1];
    armnnPads[1] = pads[3];
    armnnPads[2] = pads[0];
    armnnPads[3] = pads[2];
  }

  armnn::Pooling2dDescriptor poolDescriptor;
  poolDescriptor.m_PoolType = pool_type;
  poolDescriptor.m_PadLeft = armnnPads[0];
  poolDescriptor.m_PadRight = armnnPads[1];
  poolDescriptor.m_PadTop = armnnPads[2];
  poolDescriptor.m_PadBottom = armnnPads[3];
  poolDescriptor.m_PoolWidth = armnnKernelShape[0];
  poolDescriptor.m_PoolHeight = armnnKernelShape[1];
  poolDescriptor.m_StrideX = armnnStrides[0];
  poolDescriptor.m_StrideY = armnnStrides[1];
  poolDescriptor.m_OutputShapeRounding = pool_attrs.ceil_mode ? armnn::OutputShapeRounding::Ceiling : armnn::OutputShapeRounding::Floor;
  poolDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
  if (pool_type == armnn::PoolingAlgorithm::Average && pool_attrs.count_include_pad)
    poolDescriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
  poolDescriptor.m_DataLayout = armnn::DataLayout::NHWC;

  return poolDescriptor;
}

template <typename T, typename PoolType>
Status NHWCPool<T, PoolType>::Compute(OpKernelContext* context) const {

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> armnnDilations(2);
  armnnDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  armnnDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    ORT_NOT_IMPLEMENTED("Only implemented pooling for 4D input. Number of dimensions found: ", X->Shape().NumDimensions());
  }

  if (armnnDilations[0] * armnnDilations[1] > 1) {
    ORT_NOT_IMPLEMENTED("ArmNN does not support dilation");
  }

  std::vector<int64_t> pads = PoolBase::pool_attrs_.pads;
  std::vector<int64_t> strides = PoolBase::pool_attrs_.strides;
  std::vector<int64_t> kernel_shape = PoolBase::pool_attrs_.kernel_shape;

  if (PoolBase::pool_attrs_.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    strides.assign(kernel_shape.size(), 0);
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = PoolBase::pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  const T* x_data = X->template Data<T>();
  T* y_data = Y->template MutableData<T>();

  armnn::NetworkId* pNetworkId;
  PoolLayersIterator it = NHWCPool::poolLayers.find((OpKernel*)this);
  if (it == NHWCPool::poolLayers.end()) {

    armnn::PoolingAlgorithm pool_type;
    if (PoolBase::op_name_ == "GlobalAveragePool" || PoolBase::op_name_ == "AveragePool"){
      pool_type = armnn::PoolingAlgorithm::Average;
    } else if (PoolBase::op_name_ == "GlobalMaxPool" || PoolBase::op_name_ == "MaxPool"){
      pool_type = armnn::PoolingAlgorithm::Max;
    } else
      ORT_NOT_IMPLEMENTED("Not implemented type of pooling: ", PoolBase::op_name_);

    armnn::NetworkId networkId;

    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::Pooling2dDescriptor poolDescriptor = createNHWCDescriptor(pads, strides, kernel_shape, pool_type, PoolBase::pool_attrs_);

    armnn::IConnectableLayer *pool_armnn = myNetwork->AddPooling2dLayer(poolDescriptor, "pool_armnn");
    armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

    inputShape = { inputShape[0],
                   inputShape[2],
                   inputShape[3],
                   inputShape[1] };

    outputShape = { outputShape[0],
                    outputShape[2],
                    outputShape[3],
                    outputShape[1] };

    armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(pool_armnn->GetInputSlot(0));
    pool_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    //Set the tensors in the network.
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    pool_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, NHWCPool::run->GetDeviceSpec());

    // Load graph into runtime
    NHWCPool::run->LoadNetwork(networkId, std::move(optNet));

    std::pair<PoolLayersIterator, bool> ret;
    ret = NHWCPool::poolLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;

  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(NHWCPool::run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(NHWCPool::run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       y_data)}};

  // Execute network
  NHWCPool::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxPool,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCPool<float, MaxPool<1>>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalMaxPool,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCPool<float, MaxPool<1>>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AveragePool,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCPool<float, AveragePool>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalAveragePool,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NHWCPool<float, AveragePool>);

}  // namespace armnn_ep
}  // namespace onnxruntime
