// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/codegen.h>

#include "core/common/common.h"

#include "tvm_api.h"

namespace onnxruntime {
namespace tvm {

using TvmIntArray = ::tvm::Array<::tvm::Integer>;
using TvmPackedFunc = ::tvm::PackedFunc;

TvmModule TVMCompile(const std::string& onnx_txt,
                     const std::string& model_path,
                     const TvmEPOptions& options,
                     int opset,
                     const TVMTensorShapes& input_shapes)
{
  ::tvm::Array<TvmIntArray> shapes;
  for (size_t i = 0; i < input_shapes.size(); ++i)
  {
    TvmIntArray shape;
    for (auto& dim : input_shapes[i])
    {
      shape.push_back(::tvm::Integer(dim));
    }
    shapes.push_back(shape);
  }

  const TvmPackedFunc* compile = ::tvm::runtime::Registry::Get("tvm_onnx_import_and_compile");
  ORT_ENFORCE(compile != nullptr, "Unable to retrieve 'tvm_onnx_import_and_compile'.");
  TvmModule mod = (*compile)(TVMByteArray{onnx_txt.data(), onnx_txt.size()},
                             model_path,
                             options.executor,
                             options.target,
                             options.target_host,
                             options.opt_level,
                             opset,
                             options.freeze_weights,
                             shapes,
                             options.to_nhwc,
                             options.tuning_file_path,
                             options.tuning_type);
  ORT_ENFORCE(mod.get() != nullptr, "Compiled TVM Module is nullptr!");
  return mod;
}

void TVMSetInputs(TvmModule& mod,
                  std::vector<size_t>& inds,
                  std::vector<DLTensor>& inputs)
{
  TvmPackedFunc set_input = mod.GetFunction("set_input", false);
  TvmPackedFunc set_input_zero_copy = mod.GetFunction("set_input_zero_copy", false);
  for (size_t i = 0; i < inds.size(); ++i)
  {
    if (reinterpret_cast<size_t>(inputs[i].data) % ::tvm::runtime::kAllocAlignment == 0) {
      set_input_zero_copy(inds[i], &inputs[i]);
    } else {
      set_input(inds[i], &inputs[i]);
    }
  }
}

void TVM_VM_SetInputs(TvmModule& mod,
                      std::vector<size_t>& inds,
                      std::vector<DLTensor>& inputs)
{
  TvmPackedFunc set_input = mod.GetFunction("set_one_input", false);
  for (size_t i = 0; i < inds.size(); ++i)
  {
    set_input("main", inds[i], &inputs[i]);
  }
}

void TVMGetOutputs(TvmModule& mod,
                   std::vector<DLTensor>& outputs)
{
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    get_output(i, &outputs[i]);
  }
}

void TVM_VM_GetOutputs(TvmModule& mod,
                       std::vector<DLTensor>& outputs)
{
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    // TODO(vvchernov): think about improvement of memory management
    ::tvm::runtime::NDArray output_array = get_output(i);
    output_array.CopyTo(&outputs[i]);
  }
}

void TVMGetOutputShapes(TvmModule& mod,
                        TVMTensorShapes& output_shapes)
{
  size_t size = output_shapes.size();
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < size; ++i) {
    ::tvm::runtime::NDArray output_array = get_output(i);
    ::tvm::runtime::ShapeTuple shape_tuple = output_array.Shape();
    size_t dims_num = shape_tuple.size();
    TensorShapeVector dims;
    for (size_t j = 0; j < dims_num; ++j) {
      dims.push_back(int64_t(shape_tuple[j]));
    }
    output_shapes[i] = dims;
  }
}

void TVMRun(TvmModule& mod)
{
  TvmPackedFunc run = mod.GetFunction("run", false);
  ORT_ENFORCE(run != nullptr, "Unable to retrieve graph executor run.");
  run();
}

void TVM_VM_Run(TvmModule& mod)
{
  TvmPackedFunc run = mod.GetFunction("invoke", false);
  ORT_ENFORCE(run != nullptr, "Unable to retrieve virtual machine invoke.");
  run("main");
}

}  // namespace tvm
}  // namespace onnxruntime
