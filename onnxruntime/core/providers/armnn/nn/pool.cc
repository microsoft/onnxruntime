// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/armnn/nn/pool.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T, typename PoolType>
thread_local std::map<OpKernel*, armnn::NetworkId> Pool<T, PoolType>::poolLayers;

template <typename T, typename PoolType>
armnn::IRuntimePtr Pool<T, PoolType>::run = Pool<T, PoolType>::initRuntime();

template <typename T>
thread_local std::map<OpKernel*, armnn::NetworkId> MaxPoolV8<T>::maxPoolLayers;

template <typename T>
armnn::IRuntimePtr MaxPoolV8<T>::run = MaxPoolV8<T>::initRuntime();

armnn::Pooling2dDescriptor createDescriptor(std::vector<int64_t> pads, std::vector<int64_t> strides, std::vector<int64_t> kernel_shape, armnn::PoolingAlgorithm pool_type, onnxruntime::PoolAttributes pool_attrs){

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

  LOGS_DEFAULT(VERBOSE) << "padding: {" << armnnPads[0] << "," << armnnPads[1] << "," << armnnPads[2] << "," << armnnPads[3] << "}";
  LOGS_DEFAULT(VERBOSE) << "kernel shape: {" << armnnKernelShape[0] << "," << armnnKernelShape[1] << "}";
  LOGS_DEFAULT(VERBOSE) << "strides: {" << armnnStrides[0] << "," << armnnStrides[1] << "}";

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
  if (pool_type == armnn::PoolingAlgorithm::Average && pool_attrs.count_include_pad) {
    poolDescriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    LOGS_DEFAULT(VERBOSE) << "PaddingMethod: IgnoreValue";
  }
  poolDescriptor.m_DataLayout = armnn::DataLayout::NCHW;

  return poolDescriptor;
}

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> armnnDilations(2);
  armnnDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  armnnDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    Status s = onnxruntime::Pool<T, PoolType>::Compute(context);
    return s;
  }

  if (armnnDilations[0] * armnnDilations[1] > 1) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for dilation; defaulting to cpu implementation";
    Status s = onnxruntime::Pool<T, PoolType>::Compute(context);
    return s;
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
  PoolLayersIterator it = Pool::poolLayers.find((OpKernel*)this);
  if (it == Pool::poolLayers.end()) {

    armnn::PoolingAlgorithm pool_type;
    if (PoolBase::op_name_ == "GlobalAveragePool" || PoolBase::op_name_ == "AveragePool"){
      pool_type = armnn::PoolingAlgorithm::Average;
      LOGS_DEFAULT(VERBOSE) << "AveragePool";
    } else if (PoolBase::op_name_ == "GlobalMaxPool" || PoolBase::op_name_ == "MaxPool"){
      pool_type = armnn::PoolingAlgorithm::Max;
      LOGS_DEFAULT(VERBOSE) << "MaxPool";
    } else {
      LOGS_DEFAULT(WARNING) << "Pooling operation not supported in ArmNN; defaulting to cpu implementation" << std::endl;
      return onnxruntime::Pool<T, PoolType>::Compute(context);
    }

    armnn::NetworkId networkId;

    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::Pooling2dDescriptor poolDescriptor = createDescriptor(pads, strides, kernel_shape, pool_type, PoolBase::pool_attrs_);

    armnn::IConnectableLayer *pool_armnn = myNetwork->AddPooling2dLayer(poolDescriptor, "pool_armnn");
    armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

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
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Pool::run->GetDeviceSpec());

    if (optNet == nullptr) {
        LOGS_DEFAULT(WARNING) << "Got invalid operation; defaulting to cpu implementation";
        return onnxruntime::Pool<T, PoolType>::Compute(context);
    }

    // Load graph into runtime
    Pool::run->LoadNetwork(networkId, std::move(optNet));

    std::pair<PoolLayersIterator, bool> ret;
    ret = Pool::poolLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;

  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Pool::run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(Pool::run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       y_data)}};

  // Execute network
  Pool::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

template <typename T>
Status MaxPoolV8<T>::Compute(OpKernelContext* context) const {

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> armnnDilations(2);
  armnnDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  armnnDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if ((X->Shape().NumDimensions() != PREF_DIM) ||
      (armnnDilations[0] * armnnDilations[1] > 1)) {
    Status s = onnxruntime::MaxPoolV8::Compute(context);
    return s;
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
  PoolLayersIterator it = MaxPoolV8::maxPoolLayers.find((OpKernel*)this);
  if (it == MaxPoolV8::maxPoolLayers.end()) {

    armnn::NetworkId networkId;

    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::Pooling2dDescriptor poolDescriptor = createDescriptor(pads, strides, kernel_shape, armnn::PoolingAlgorithm::Max, PoolBase::pool_attrs_);

    armnn::IConnectableLayer *pool_armnn = myNetwork->AddPooling2dLayer(poolDescriptor, "pool_armnn");
    armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

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
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, MaxPoolV8::run->GetDeviceSpec());

    if (optNet == nullptr) {
        LOGS_DEFAULT(WARNING) << "Got invalid operation; defaulting to cpu implementation";
        return onnxruntime::MaxPoolV8::Compute(context);
    }

    // Load graph into runtime
    MaxPoolV8::run->LoadNetwork(networkId, std::move(optNet));

    std::pair<PoolLayersIterator, bool> ret;
    ret = MaxPoolV8::maxPoolLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;

  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(MaxPoolV8::run->GetInputTensorInfo(*pNetworkId, 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(MaxPoolV8::run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       y_data)}};

  // Execute network
  MaxPoolV8::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                  \
      op_name,                                                                              \
      kOnnxDomain,                                                                          \
      since_version,                                                                        \
      end_version,                                                                          \
      data_type,                                                                            \
      kArmNNExecutionProvider,                                                              \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),     \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      data_type,                                                                        \
      kArmNNExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10)
POOLING_KERNEL(AveragePool, float, AveragePool, 11)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1)

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                            \
      MaxPool,                                                                      \
      kOnnxDomain,                                                                  \
      8,                                                                            \
      11,                                                                           \
      float,                                                                        \
      kArmNNExecutionProvider,                                                      \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      MaxPoolV8<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime

