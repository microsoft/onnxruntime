// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include "tvm_runner_impl.h"
#include "tvm_utils.h"
#include "tvm_api.h"


namespace onnxruntime {
namespace tvm {

/* ------------------------------------ RunnerImplFactory ----------------------------- */

std::shared_ptr<RunnerImpl> getTVMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
                                             const TvmEPOptions& options,
                                             const InputsInfoMap& inputs_info,
                                             const std::vector<DLTensor> output_tensors) {
    const std::string& name = options.executor;
    if (name == "graph") {
        return std::make_shared<GERunnerImpl>(mod, inputs_info, options.output_shapes, output_tensors);
    } else if (name == "vm") {
        return std::make_shared<VMRunnerImpl>(mod, inputs_info, options.output_shapes, output_tensors);
    }
    return nullptr;
}

/* ------------------------------------ RunnerImpl ------------------------------------ */

RunnerImpl::RunnerImpl(const std::shared_ptr<TvmModule>& mod,
                       const InputsInfoMap& inputs_info,
                       const TVMTensorShapes output_shapes,
                       const std::vector<DLTensor> output_tensors) :
  mod_(mod),
  inputs_info_(inputs_info),
  output_shapes_(output_shapes),
  output_tensors_(output_tensors) {
}

void RunnerImpl::convert_input_tensors2dl_tensors(Ort::CustomOpApi& ort,
                                                  OrtKernelContext* context,
                                                  std::vector<DLTensor>& dst,
                                                  std::vector<size_t>& dst_inds) {
  size_t num = inputs_info_.size();
  dst.reserve(num);
  dst_inds.reserve(num);
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
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

    DLTensor t;
    t.device = GetDLDevice(device);
    t.dtype = GetDataType(tensor_type);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
    t.ndim = shape.size();
    t.shape = shape.data();
    dst.emplace_back(t);
    dst_inds.push_back(i);
  }
}

void RunnerImpl::add_device_type_data2output_tensors(Ort::CustomOpApi& ort,
                                                     OrtKernelContext* context) {
  size_t num_outputs = output_tensors_.size();
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

    output_tensors_[i].device = GetDLDevice(device);
    output_tensors_[i].dtype = GetDataType(tensor_type);
    output_tensors_[i].data = ort.GetTensorMutableData<void>(output_tensor);
  }
}

/* ------------------------------------ GERunnerImpl ------------------------------------ */

GERunnerImpl::GERunnerImpl(const std::shared_ptr<TvmModule>& mod,
                           const InputsInfoMap& inputs_info,
                           const TVMTensorShapes output_shapes,
                           const std::vector<DLTensor> output_tensors) :
  RunnerImpl(mod, inputs_info, output_shapes, output_tensors) {
}

void GERunnerImpl::set_input(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  std::vector<size_t> inds;
  std::vector<DLTensor> dl_tensors_inputs;
  convert_input_tensors2dl_tensors(ort, context, dl_tensors_inputs, inds);

  tvm::TVMSetInputs(*mod_, inds, dl_tensors_inputs);
}

void GERunnerImpl::connect_output_tensors2ort(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  add_device_type_data2output_tensors(ort, context);
}

void GERunnerImpl::run_and_get_output() {
  tvm::TVMRun(*mod_);
  tvm::TVMGetOutputs(*mod_, output_tensors_);
}

/* ------------------------------------ VMRunnerImpl ------------------------------------ */

VMRunnerImpl::VMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
                           const InputsInfoMap& inputs_info,
                           const TVMTensorShapes output_shapes,
                           const std::vector<DLTensor> output_tensors) :
  RunnerImpl(mod, inputs_info, output_shapes, output_tensors) {
}

void VMRunnerImpl::set_input(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  std::vector<size_t> inds;
  std::vector<DLTensor> dl_tensors_inputs;
  convert_input_tensors2dl_tensors(ort, context, dl_tensors_inputs, inds);

  tvm::TVM_VM_SetInputs(*mod_, inds, dl_tensors_inputs);
}

void VMRunnerImpl::connect_output_tensors2ort(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  if(!probe_infer_) {
    infer_once_to_get_output_shapes();
  }

  add_device_type_data2output_tensors(ort, context);
}

void VMRunnerImpl::run_and_get_output() {
  tvm::TVM_VM_Run(*mod_);
  tvm::TVM_VM_GetOutputs(*mod_, output_tensors_);
}

void VMRunnerImpl::infer_once_to_get_output_shapes() {
  tvm::TVM_VM_Run(*mod_);
  size_t num_outputs = output_tensors_.size();
  // TODO(vvchernov): check it
  output_shapes_.resize(num_outputs);
  tvm::TVMGetOutputShapes(*mod_, output_shapes_);
  for (size_t i = 0; i < num_outputs; ++i) {
    output_tensors_[i].ndim = output_shapes_[i].size();
    output_tensors_[i].shape = output_shapes_[i].data();
  }
  probe_infer_ = true;
}

}   // namespace tvm
}   // namespace onnxruntime
