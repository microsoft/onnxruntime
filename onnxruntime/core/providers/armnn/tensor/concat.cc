// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/armnn/tensor/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local std::map<OpKernel*, armnn::NetworkId> Concat<T>::concatLayers;

template <typename T>
armnn::IRuntimePtr Concat<T>::run = Concat<T>::initRuntime();

template <typename T>
Status Concat<T>::Compute(OpKernelContext* ctx) const {

  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(input_count);

  LOGS_DEFAULT(VERBOSE) << "Concat ArmNN:";
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
    LOGS_DEFAULT(VERBOSE) << "X[" << i << "]: " << ctx->Input<Tensor>(i)->Shape().ToString().c_str();
  }
  LOGS_DEFAULT(VERBOSE) << "axis: " << axis_;
  LOGS_DEFAULT(VERBOSE) << std::endl;

  std::vector<int64_t> output_dims = input_tensors[0]->Shape().GetDims();

  // 'Concat' mode
  if (!is_stack_) {
    // While concating, the rank of the output is the same as the input rank(s)

    // Calculate the size of the concatenated axis
    size_t concat_axis_size = 0;
    for (int64_t index = 0; index < input_count; index++) {
      concat_axis_size += input_tensors[index]->Shape()[static_cast<int>(axis_)];
    }

    output_dims[axis_] = concat_axis_size;
  } else { // 'Stack' mode
    // While stacking, the rank of the output is one more than the input rank(s).
    // Stacking may be thought of as adding an unit dimension (of value 1) in the input tensors,
    // and concatenating them on thie new axis.
    // The value in the corresponding axis of the output will be the number of inputs that are being stacked.
    output_dims.insert(output_dims.begin() + axis_, static_cast<int64_t>(input_count));
  }

  if(output_dims.size() > 4 || axis_ > 3) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    return onnxruntime::Concat::Compute(ctx);
  }

  TensorShape output_shape(output_dims);
  Tensor* Y = ctx->Output(0, output_shape);

  armnn::NetworkId* pNetworkId;
  ConcatIterator it = Concat::concatLayers.find((OpKernel*)this);
  if (it == Concat::concatLayers.end()) {

    armnn::NetworkId networkId;
    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    const unsigned int supportedNumDims = 4;
    unsigned int numConcatViews = input_count;
    armnn::OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatViews), supportedNumDims);
    concatDescriptor.SetConcatAxis(axis_);
    armnn::TensorShape mergeDims(supportedNumDims);
    unsigned int mergeDim = 0;
    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex) {
      // Copy the input tensor shape to mergeDimSizes and initialize the view origin coordinates for the current input
      mergeDims = ArmNNTensorShape(input_tensors[viewIndex]->Shape(), PREF_DIM);
      unsigned int* viewOrigin = const_cast<unsigned int*>(concatDescriptor.GetViewOrigin(viewIndex));
      std::fill(viewOrigin, viewOrigin + supportedNumDims, 0);

      // Update the view origin coordinates and the merge dimension value
      concatDescriptor.SetViewOriginCoord(viewIndex, axis_, mergeDim);
      mergeDim += mergeDims[axis_];
    }

    // Update the output shape
    mergeDims[axis_] = mergeDim;
    armnn::IConnectableLayer *layer = myNetwork->AddConcatLayer(concatDescriptor, "concat_armnn");

    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex) {
      armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(viewIndex);
      InputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(viewIndex));
      InputLayer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(ArmNNTensorShape(input_tensors[viewIndex]->Shape(), PREF_DIM), armnn::DataType::Float32));
    }

    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);
    layer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(mergeDims, armnn::DataType::Float32)); 

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Concat::run->GetDeviceSpec());

    if (optNet == nullptr) {
      LOGS_DEFAULT(WARNING) << "Got invalid operation; defaulting to cpu implementation";
      return onnxruntime::Concat::Compute(ctx);
    }

    // Load graph into runtime
    Concat::run->LoadNetwork(networkId, std::move(optNet));

    std::pair<ConcatIterator, bool> ret;
    ret = Concat::concatLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
    pNetworkId = &ret.first->second;
    
  } else {
    pNetworkId = &it->second;
  }

  armnn::InputTensors inputTensors{};
  for (int index = 0; index < input_count; ++index)
    inputTensors.push_back({index, armnn::ConstTensor(Concat::run->GetInputTensorInfo(*pNetworkId, index),
                                                       input_tensors[index]->template Data<T>())});
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(Concat::run->GetOutputTensorInfo(*pNetworkId, 0),
                                                       Y->template MutableData<T>())}};

  Concat::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    4, 10,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
