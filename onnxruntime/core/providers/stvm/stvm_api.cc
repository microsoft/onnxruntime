// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stvm_api.h"

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>

namespace stvm {

tvm::runtime::Module TVMCompile(const std::string& onnx_txt,
                                const std::string& model_path,
                                const std::string& target,
                                const std::string& target_host,
                                int opt_level,
                                int opset,
                                bool freeze_params,
                                const std::vector<std::vector<int64_t>>& input_shapes,
                                bool nhwc,
                                const std::string& tuning_logfile,
                                const std::string& tuning_type)
{
  tvm::Array<tvm::Array<tvm::Integer>> shapes;
  for (size_t i = 0; i < input_shapes.size(); ++i)
  {
    tvm::Array<tvm::Integer> shape;
    for (auto& dim : input_shapes[i])
    {
      shape.push_back(tvm::Integer(dim));
    }
    shapes.push_back(shape);
  }

  const tvm::PackedFunc* compile = tvm::runtime::Registry::Get("tvm_onnx_import_and_compile");
  tvm::runtime::Module mod = (*compile)(
          TVMByteArray{onnx_txt.data(), onnx_txt.size()},
          model_path,
          target,
          target_host,
          opt_level,
          opset,
          freeze_params,
          shapes,
          nhwc,
          tuning_logfile,
          tuning_type);
  return mod;
}

void TVMSetInputs(tvm::runtime::Module& mod,
                  std::vector<size_t>& inds,
                  std::vector<DLTensor>& inputs)
{
  // TODO(vvchernov): set_input_zero_copy is more preferable but it does not satisfy alignment conditions.
  //tvm::PackedFunc set_input = mod.GetFunction("set_input_zero_copy", false);

  tvm::PackedFunc set_input = mod.GetFunction("set_input", false);
  for (auto& i : inds)
  {
    set_input(i, &inputs[i]);
  }
}

void TVMGetOutputShapes(tvm::runtime::Module& mod,
                        size_t num_outputs,
                        std::vector<std::vector<int64_t>>& output_shapes)
{
  output_shapes.clear();
  tvm::PackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < num_outputs; ++i) {
    tvm::runtime::NDArray output_array = get_output(i);
    tvm::runtime::ShapeTuple shape_tuple = output_array.Shape();
    size_t dims_num = shape_tuple.size();
    std::vector<int64_t> dims;
    for (size_t j = 0; j < dims_num; ++j) {
      dims.push_back(int64_t(shape_tuple[j]));
    }
    output_shapes.push_back(dims);
  }
}

void TVMRun(tvm::runtime::Module& mod,
            std::vector<DLTensor>& outputs,
            [[maybe_unused]] tvm::runtime::TVMRetValue *ret)
{
  const tvm::PackedFunc* run = tvm::runtime::Registry::Get("tvm_run");
  (*run)(mod);

  tvm::PackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    get_output(i, &outputs[i]);
  }
}

}  // namespace stvm
