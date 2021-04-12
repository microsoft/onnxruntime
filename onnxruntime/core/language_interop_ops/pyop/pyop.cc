// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pyop.h"
#ifdef _WIN32
#define LIB_PYOP "onnxruntime_pywrapper.dll"
#define LOAD_PYOP_LIB(n, v, m) ORT_ENFORCE((v = LoadLibraryA(n)) != nullptr, m)
#else
#ifdef __APPLE__
#define LIB_PYOP "./libonnxruntime_pywrapper.dylib"
#else
#define LIB_PYOP "./libonnxruntime_pywrapper.so"
#endif
#define LOAD_PYOP_LIB(n, v, m) ORT_ENFORCE((v = dlopen(n, RTLD_NOW | RTLD_GLOBAL)) != nullptr, m)
#include "dlfcn.h"
#endif
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <functional>
#include <iostream>
#include <sstream>
#include <numeric>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <unordered_map>
#include "pyop_lib_proxy.h"

using namespace std;
namespace onnxruntime {

PyCustomKernel::PyCustomKernel(
    Ort::CustomOpApi ort,
    const OnnxAttrs& attrs,
    const std::string& module,
    const std::string& class_name,
    const std::string& compute,
    PyOpLogFunc logging_func)
    : ort_(ort), attrs_(attrs), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) {
  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  ORT_ENFORCE(PyOpLibProxy::GetInstance().Initialized(), "Py library not properly initialized.");
  instance_ = PyOpLibProxy::GetInstance().NewInstance(module.c_str(), class_name_.c_str(), attrs_);
  ORT_ENFORCE(instance_ != nullptr, "Python run instance_ should not be nullptr");
  PyOpLibProxy::GetInstance().PutGil(state);
  ORT_ENFORCE(nullptr != instance_, PyOpLibProxy::GetInstance().GetLastErrorMessage(err));
}

PyCustomKernel::~PyCustomKernel() {
  if (nullptr != instance_) {
    auto state = PyOpLibProxy::GetInstance().GetGil();
    PyOpLibProxy::GetInstance().ReleaseInstance(instance_);
    PyOpLibProxy::GetInstance().PutGil(state);
    instance_ = nullptr;
  }
}

// Do nothing since Custom Op does not trigger shape inference
void PyCustomKernel::GetOutputShape(OrtKernelContext*, size_t, OrtTensorTypeAndShapeInfo*) {}

Status PyCustomKernel::Compute(OrtKernelContext* context) {
  bool is_generic_python_call = false;
  if (is_generic_python_call) {
    ORT_ENFORCE(nullptr != context);
    auto inputs_count = (size_t) reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
    std::vector<OrtValue*> inputs;
    std::vector<std::unique_ptr<char[]>> outputs;
    std::vector<int32_t> outputs_elem_size;
    std::vector<std::vector<int64_t>> outputs_dim;

    for (size_t i = 0; i < inputs_count; ++i) {
      auto ort_value = ort_.KernelContext_GetInput(context, i);
      inputs.push_back(const_cast<OrtValue*>(ort_value));
    }

    std::string err;
    auto state = PyOpLibProxy::GetInstance().GetGil();
    ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonFunc(instance_, compute_.c_str(), inputs, outputs, outputs_elem_size,
                                                             outputs_dim, logging_func_),
                PyOpLibProxy::GetInstance().GetLastErrorMessage(err));
    PyOpLibProxy::GetInstance().PutGil(state);

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto ort_output = ort_.KernelContext_GetOutput(context, i, outputs_dim[i].data(), outputs_dim[i].size());
      auto output_mem_addr = ort_.GetTensorMutableData<char>(ort_output);
      auto output_len = std::accumulate(begin(outputs_dim[i]), end(outputs_dim[i]), static_cast<int64_t>(outputs_elem_size[i]), std::multiplies<int64_t>());
      memcpy(output_mem_addr, outputs[i].get(), output_len);
    }
    return Status::OK();
  }

  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  std::vector<OrtValue*> inputs;
  std::vector<void*> outputs;

  for (size_t i = 0; i < inputs_count; ++i) {
    auto ort_value = ort_.KernelContext_GetInput(context, i);
    inputs.push_back(const_cast<OrtValue*>(ort_value));
  }

  // Generate position indexes for tensor inputs. They are needed
  // when calling InvokePythonAutoGradFunc. 
  std::vector<int64_t> arg_positions;
  if (!arg_positions.size()) {
    for (int64_t i = 0; i < (int64_t)inputs.size(); ++i) {
      arg_positions.push_back(i);
    }
  }

  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();

  // There is no constants when not calling autograd.Function from PythonOp and PythonOpGrad.
  // Thus, const_args and const_arg_positions are empty.
  std::vector<void*> const_args;
  std::vector<int64_t> const_arg_positions;
  ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonAutoGradFunc(instance_, compute_.c_str(), inputs, arg_positions, outputs,
                                                                   logging_func_, const_args, const_arg_positions),
              PyOpLibProxy::GetInstance().GetLastErrorMessage(err));  //ORT_ENFORCE
  PyOpLibProxy::GetInstance().PutGil(state);

  // We had the assumption:
  // The 1st output is address of ctx.grad_fn function.
  // The 2nd output is address of OrtValue we got from Python script run.
  void* forward_ret_ortvalue_addr = outputs[1];
  auto* forward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(forward_ret_ortvalue_addr);
  ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(0, *forward_ret_ortvalue_ptr));
  // TODO: handle the output, maing sure its lifetime is still there untill the backward operation completed.
  return Status::OK();
}

PyCustomOp::PyCustomOp(const OnnxAttrs& attrs,
                       const OnnxTypes& inputs_type,
                       const OnnxTypes& outputs_type,
                       const std::string& module,
                       const std::string& class_name,
                       const std::string& compute,
                       PyOpLogFunc logging_func)
    : attrs_(attrs), inputs_type_(inputs_type), outputs_type_(outputs_type), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) {
  OrtCustomOp::version = ORT_API_VERSION;
}

void* PyCustomOp::CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo*) const {
  return new PyCustomKernel(api, attrs_, module_, class_name_, compute_, logging_func_);
}

const char* PyCustomOp::GetName() const { return "PyOp"; }

size_t PyCustomOp::GetInputTypeCount() const { return inputs_type_.size(); }
ONNXTensorElementDataType PyCustomOp::GetInputType(size_t index) const { return inputs_type_[index]; }

size_t PyCustomOp::GetOutputTypeCount() const { return outputs_type_.size(); }
ONNXTensorElementDataType PyCustomOp::GetOutputType(size_t index) const { return outputs_type_[index]; }

PyCustomOp* LoadPyOp(const ONNX_NAMESPACE::NodeProto& node_proto, PyOpLogFunc log_func) {
  OnnxAttrs onnx_attrs;
  OnnxTypes input_types, output_types;
  std::string module, class_name, compute = "compute";
  for (int j = 0; j < node_proto.attribute_size(); ++j) {
    const auto& attr = node_proto.attribute(j);
    if (utils::HasString(attr)) {
      if (attr.name() == "module")
        module = attr.s();
      else if (attr.name() == "class_name")
        class_name = attr.s();
      else if (attr.name() == "compute")
        compute = attr.s();
      else
        onnx_attrs[attr.name()] = attr.s();
    } else if (attr.ints_size() > 0) {
      if (attr.name() == "input_types") {
        for (int k = 0; k < attr.ints_size(); ++k) {
          input_types.push_back(static_cast<ONNXTensorElementDataType>(attr.ints(k)));
        }
      } else if (attr.name() == "output_types") {
        for (int k = 0; k < attr.ints_size(); ++k) {
          output_types.push_back(static_cast<ONNXTensorElementDataType>(attr.ints(k)));
        }
      }
    }
  }  //for
  ORT_ENFORCE(module != "", "PyOp module not specified");
  ORT_ENFORCE(class_name != "", "PyOp class name not specified");
  ORT_ENFORCE(!input_types.empty(), "PyOp node inputs not specified");
  ORT_ENFORCE(!output_types.empty(), "PyOp node outputs not specified");
  return new PyCustomOp(onnx_attrs, input_types, output_types, module, class_name, compute, log_func);
}
}  // namespace onnxruntime
