// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#ifdef RELU_ARMNN

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/activation/activations.h"
#include "core/providers/armnn/armnn_fwd.h"

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local std::map<OpKernel*, armnn::NetworkId> Relu<T>::reluLayers;

template <typename T>
armnn::IRuntimePtr Relu<T>::run = Relu<T>::initRuntime();

template <typename T>
Status Relu<T>::Compute(OpKernelContext* context) const {

  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());

  const T* src_data = X->template Data<T>();
  T* dst_data = Y->template MutableData<T>();

  armnn::NetworkId* pNetworkId;
  ReluLayersIterator it = Relu::reluLayers.find((OpKernel*)this);
  if (it == Relu::reluLayers.end()) {
    
    armnn::NetworkId networkId;
    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    armnn::TensorShape inputShape = ArmNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ArmNNTensorShape(Y->Shape());

    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::ReLu;

    armnn::IConnectableLayer* activation = myNetwork->AddActivationLayer(desc, "relu_armnn");

    armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    //Set the tensors in the network.
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    activation->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Relu::run->GetDeviceSpec());

    if (optNet == nullptr) {
      ORT_NOT_IMPLEMENTED("Something went wrong when creating the layer");
    }

    // Load graph into runtime
    Relu::run->LoadNetwork(networkId, std::move(optNet));

    std::pair<ReluLayersIterator, bool> ret;
    ret = Relu::reluLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;

  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Relu::run->GetInputTensorInfo(*pNetworkId, 0),
                                                          src_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(Relu::run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       dst_data)}};

  Relu::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Relu<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime

#endif
