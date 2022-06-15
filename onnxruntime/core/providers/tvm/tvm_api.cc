// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <glob.h>  // glob(), globfree()
#include <string.h>  // memset()
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/codegen.h>

#include "core/common/common.h"

#include "tvm_api.h"


namespace onnxruntime {
namespace tvm {

using TvmIntArray = ::tvm::Array<::tvm::Integer>;
using TvmPackedFunc = ::tvm::PackedFunc;
namespace tvm_rt = ::tvm::runtime;
namespace tvm_rt_vm = tvm_rt::vm;

TvmModule TVMCompile(const TvmEPOptions& options,
                     const std::string& onnx_txt,
                     const std::string& model_path,
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

  const TvmPackedFunc* compile = tvm_rt::Registry::Get("tvm_onnx_import_and_compile");
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

std::vector<std::string> glob(const std::string& pattern) {
  std::vector<std::string> filenames;

  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  ORT_ENFORCE(return_value == 0, "No results of glob for pattern: " + pattern);

  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(std::string(glob_result.gl_pathv[i]));
  }

  globfree(&glob_result);

  return filenames;
}

std::string filter_lib_paths(const std::vector<std::string>& lib_paths) {
  std::string lib_path;
  size_t counter = 0;
  for (const auto& path : lib_paths) {
    if (path.find("libtvm_runtime.so") != std::string::npos ||
        path.find("liboctomized_model.so") != std::string::npos) {
      ++counter;
    } else {
      lib_path = path;
    }
  }
  ORT_ENFORCE((lib_paths.size() - counter) == 1, "It should be only one shared library for model after filtering");

  return lib_path;
}

static std::unordered_map<std::string, uint64_t> str2dev_type = {
  {"llvm", 1},
  {"stackvm", 1},
  {"cpu", 1},
  {"c", 1},
  {"hybrid", 1},
  {"composite", 1},
  {"cuda", 2},
  {"nvptx", 2},
  {"cl", 4},
  {"opencl", 4},
  {"sdaccel", 4},
  {"aocl", 5},
  {"aocl_sw_emu", 5},
  {"vulkan", 7},
  {"metal", 8},
  {"vpi", 9},
  {"rocm", 10},
  {"ext_dev", 12},
  {"hexagon", 14},
  {"webgpu", 15}
};

TvmModule TVMSoCompile(const TvmEPOptions& options) {
  const std::string dir = options.so_folder;
  const std::string lib_path = filter_lib_paths(glob(dir + "/*.so"));
  const std::string consts_path = dir + "/consts";
  const auto& ro_paths = glob(dir + "/*.ro");
  ORT_ENFORCE(ro_paths.size() == 1, "It should be only one ro file in folder: " + dir);
  const std::string vm_exec_code_path = ro_paths[0];

  TvmModule lib = TvmModule::LoadFromFile(lib_path);

  std::ifstream code(vm_exec_code_path, std::ios::binary);
  std::stringstream ss;
  ss << code.rdbuf();

  auto exec_mod = tvm_rt_vm::Executable::Load(ss.str(), lib);
  const tvm_rt_vm::Executable* tmp = exec_mod.as<tvm_rt_vm::Executable>();
  auto exec = tvm_rt::GetObjectPtr<tvm_rt_vm::Executable>(const_cast<tvm_rt_vm::Executable*>(tmp));
  exec->LoadLateBoundConstantsFromFile(consts_path);

  auto vm = tvm_rt::make_object<tvm_rt_vm::VirtualMachine>();
  vm->LoadExecutable(exec);

  size_t pos = options.target.find(" ");
  const std::string dev_type_str = options.target.substr(0, pos);
  ORT_ENFORCE(!dev_type_str.empty(), "Device was not found in target string");
  uint64_t dev_type = str2dev_type[dev_type_str];
  const uint64_t cpu_type = str2dev_type["cpu"];
  // Initialize the VM for the specified device. If the device is not a CPU,
  // We'll need to add a CPU context to drive it.
  int arity;
  if (dev_type == cpu_type) {
    arity = 3;
  } else {
    arity = 6;
  }
  uint64_t alloc_type = uint64_t(tvm_rt_vm::AllocatorType::kPooled);
  // TODO(vchernov): multiple devices using and using device with specified id are not supported
  // Always use the first device of the specified type.
  uint64_t device_id = 0;
  std::vector<TVMValue> init_vals(arity);
  std::vector<int> codes(arity);
  tvm_rt::TVMArgsSetter setter(init_vals.data(), codes.data());
  setter(0, dev_type);
  setter(1, device_id);
  setter(2, alloc_type);
  // Also initialize a CPU device context.
  if (dev_type != cpu_type) {
    setter(3, cpu_type);
    setter(4, device_id);
    setter(5, alloc_type);
  }
  tvm_rt::TVMRetValue rv;
  // Call the packed func with the init arguments.
  vm->GetFunction("init", nullptr).CallPacked(tvm_rt::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

  return TvmModule(vm);
}

void TVMSetInputs(TvmModule& mod,
                  std::vector<size_t>& inds,
                  std::vector<DLTensor>& inputs)
{
  TvmPackedFunc set_input = mod.GetFunction("set_input", false);
  TvmPackedFunc set_input_zero_copy = mod.GetFunction("set_input_zero_copy", false);
  for (size_t i = 0; i < inds.size(); ++i) {
    if (reinterpret_cast<size_t>(inputs[i].data) % tvm_rt::kAllocAlignment == 0) {
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
  size_t num_total_args = inputs.size() + 1;
  std::vector<TVMValue> tvm_values(num_total_args);
  std::vector<int> tvm_type_codes(num_total_args);
  ::tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
  const std::string func_name = "main";
  setter(0, func_name.c_str());
  for (size_t k = 0; k < num_total_args - 1; ++k) {
    setter(inds[k]+1, &inputs[k]);
  }

  TvmPackedFunc set_input = mod.GetFunction("set_input", false);
  ::tvm::runtime::TVMRetValue rv;
  set_input.CallPacked(::tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_total_args), &rv);
}

void TVMGetOutputs(TvmModule& mod,
                   std::vector<DLTensor>& outputs)
{
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    get_output(i, &outputs[i]);
  }
}

void TVM_VM_GetOutputs(TvmModule& mod,
                       std::vector<DLTensor>& outputs)
{
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    // TODO(vvchernov): think about improvement of memory management
    tvm_rt::NDArray output_array = get_output(i);
    output_array.CopyTo(&outputs[i]);
  }
}

void TVMGetOutputShapes(TvmModule& mod,
                        TVMTensorShapes& output_shapes)
{
  size_t size = output_shapes.size();
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < size; ++i) {
    tvm_rt::NDArray output_array = get_output(i);
    tvm_rt::ShapeTuple shape_tuple = output_array.Shape();
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
