// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"

#include "tvm_runner.h"
#include "tvm_utils.h"
#include "tvm_execution_provider.h"
#include "tvm_api.h"


using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace tvm {

TVMRunner::TVMRunner(TvmExecutionProvider* ep,
                     const std::string& name,
                     const Graph& graph) :
  use_vm_(ep->options_.executor == "vm") {
    // Extract input shapes
    const ORTGraphNodes& all_nodes = graph.GetInputsIncludingInitializers();
    TVMTensorShapes input_shapes;
    size_t indx = 0;
    if (ep->options_.freeze_weights) {
      for (const auto* node : all_nodes) {
        const auto& node_name = node->Name();
        if(!graph.IsInitializedTensor(node_name)) {
          TVMTensorShape ishape;
          if(!ep->options_.input_shapes.empty() &&
              ep->options_.input_shapes.count(node_name)) {
            ishape = ep->options_.input_shapes[node_name];
            inputs_info_[indx] = ishape;
            update_output_shapes_ = true;
          } else {
            getTensorInfo(*node->Shape(), ishape, indx);
          }
          input_shapes.emplace_back(ishape);
        }
        ++indx;
      }
    } else {
      for (const auto* node : all_nodes) {
        const auto& node_name = node->Name();
        TVMTensorShape ishape;
        if(!ep->options_.input_shapes.empty() &&
            ep->options_.input_shapes.count(node_name)) {
          ishape = ep->options_.input_shapes[node_name];
          inputs_info_[indx++] = ishape;
          update_output_shapes_ = true;
        } else {
          getTensorInfo(*node->Shape(), ishape, indx++);
        }
        if(!graph.IsInitializedTensor(node_name)) {
          input_shapes.emplace_back(ishape);
        }
      }
    }

    // Get module from tvm
    mod_ = ep->CompileFunc(name, input_shapes);

    // Prepare draft for output tvm tensors
    const ORTGraphNodes& ort_outputs_info = graph.GetOutputs();
    size_t num_outputs = ort_outputs_info.size();

    if (update_output_shapes_) {
      if (!use_vm_) {
        tvm::TVMGetOutputShapes(*mod_, num_outputs, output_shapes_);
      }
    } else {
      for (auto i = 0u; i < num_outputs; i++) {
        TensorShape ort_shape = utils::GetTensorShapeFromTensorShapeProto(*ort_outputs_info[i]->Shape());
        int dims = ort_shape.NumDimensions();

        TVMTensorShape oshape(dims);
        for (int j = 0; j < dims; ++j) {
          oshape[j] = int64_t(ort_shape[j]);
        }
        output_shapes_.emplace_back(oshape);
      }
    }

    for (auto i = 0u; i < num_outputs; i++) {
      DLTensor t;
      // Draft for tensor, correct data is defined during inference
      t.strides = nullptr;
      t.byte_offset = 0;
      t.data = nullptr;
      if (!(use_vm_ && update_output_shapes_)) {
        t.ndim = output_shapes_[i].size();
        t.shape = output_shapes_[i].data();
      } else {
        t.ndim = 0;
        t.shape = nullptr;
      }

      tensors_outputs_.push_back(t);
    }
  }

common::Status TVMRunner::operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
  Ort::CustomOpApi ort{*api};

  size_t num = inputs_info_.size();
  std::vector<size_t> inds(num);
  std::vector<DLTensor> dl_tensors_inputs(num);
  size_t counter = 0u;
  for (auto& info : inputs_info_) {
    // TODO(vvchernov): decomposition declaration only available with -std=c++1z or -std=gnu++1z
    auto& i = info.first;
    auto& shape = info.second;
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
    ORT_ENFORCE(input_tensor->IsTensor());
    const Tensor& tensor = input_tensor->Get<Tensor>();
    const OrtDevice& device = tensor.Location().device;
    auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    auto tensor_type = ort.GetTensorElementType(tensor_info);
    if (!update_output_shapes_) {
      std::vector<int64_t> ort_shape = ort.GetTensorShape(tensor_info);
      ORT_ENFORCE(compare_shapes(shape, ort_shape));
    }
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

    DLTensor t;
    t.device = GetDLDevice(device);
    t.dtype = GetDataType(tensor_type);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
    t.ndim = shape.size();
    t.shape = shape.data();
    dl_tensors_inputs[counter] = t;
    inds[counter++] = i;
  }
  if (use_vm_) {
    tvm::TVM_VM_SetInputs(*mod_, inds, dl_tensors_inputs);
    // Infer once for calculating of output shapes
    if(!probe_infer_) {
      tvm::TVM_VM_Run(*mod_);
      size_t num_outputs = tensors_outputs_.size();
      tvm::TVMGetOutputShapes(*mod_, num_outputs, output_shapes_);
      for (size_t i = 0; i < num_outputs; ++i) {
        tensors_outputs_[i].ndim = output_shapes_[i].size();
        tensors_outputs_[i].shape = output_shapes_[i].data();
      }
      probe_infer_ = true;
    }
  } else {
    tvm::TVMSetInputs(*mod_, inds, dl_tensors_inputs);
  }

  size_t num_outputs = tensors_outputs_.size();
  for (auto i = 0u; i < num_outputs; i++) {
    //setup output tensor property
    OrtValue* output_tensor = ort.KernelContext_GetOutput(context,
                                                          i,
                                                          output_shapes_[i].data(),
                                                          output_shapes_[i].size());
    ORT_ENFORCE(output_tensor->IsTensor());
    const Tensor& tensor = output_tensor->Get<Tensor>();
    const OrtDevice& device = tensor.Location().device;
    auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
    auto tensor_type = ort.GetTensorElementType(tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

    tensors_outputs_[i].device = GetDLDevice(device);
    tensors_outputs_[i].dtype = GetDataType(tensor_type);
    tensors_outputs_[i].data = ort.GetTensorMutableData<void>(output_tensor);
  }

  if (use_vm_) {
    tvm::TVM_VM_Run(*mod_);
    tvm::TVM_VM_GetOutputs(*mod_, tensors_outputs_);
  } else {
    tvm::TVMRun(*mod_);
    tvm::TVMGetOutputs(*mod_, tensors_outputs_);
  }

  return Status::OK();
}

void TVMRunner::getTensorInfo(const TensorShapeProto& shape_proto,
                              TVMTensorShape& ishape,
                              size_t indx) {
  TensorShape ort_shape = utils::GetTensorShapeFromTensorShapeProto(shape_proto);
  int dims = ort_shape.NumDimensions();

  ishape.resize(dims);
  for (int j = 0; j < dims; ++j) {
    int64_t dim = int64_t(ort_shape[j]);
    ORT_ENFORCE(dim > 0, "Input dimension is not positive value (dim = " + std::to_string(dim) + "). " +
      "Please use provider options to setup input_names and input_shapes");
    ishape[j] = dim;
  }
  inputs_info_[indx] = ishape;
}

bool TVMRunner::compare_shapes(const TVMTensorShape& shape1, const TVMTensorShape& shape2) {
  size_t size = shape1.size();
  if (shape2.size() == size) {
    for (size_t i = 0; i < size; ++i) {
      if(shape1[i] != shape2[i]) {
        return false;
      }
    }
  } else {
    return false;
  }

  return true;
}

}
}
