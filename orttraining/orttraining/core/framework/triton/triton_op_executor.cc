// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/triton/triton_op_executor.h"

#include "orttraining/core/framework/torch/dlpack_python.h"
#include "orttraining/core/framework/torch/gil.h"

namespace onnxruntime {
namespace training {
namespace framework {
namespace triton {

namespace {
void PythonObjectDeleter(PyObject* ptr) { Py_XDECREF(ptr); }
}  // namespace

void TritonOpExecutor::Initialize(PyObject* config_getter, PyObject* executor_by_name, PyObject* executor_by_onnx) {
  ORT_ENFORCE(config_getter_.get() == nullptr && config_getter != nullptr && executor_by_name_.get() == nullptr &&
              executor_by_name != nullptr && executor_by_onnx_.get() == nullptr && executor_by_onnx != nullptr);
  Py_INCREF(config_getter);
  Py_INCREF(executor_by_name);
  Py_INCREF(executor_by_onnx);
  PythonObjectPtr config_getter_ptr(config_getter, PythonObjectDeleter);
  config_getter_ = std::move(config_getter_ptr);
  PythonObjectPtr executor_by_name_ptr(executor_by_name, PythonObjectDeleter);
  executor_by_name_ = std::move(executor_by_name_ptr);
  PythonObjectPtr executor_by_onnx_ptr(executor_by_onnx, PythonObjectDeleter);
  executor_by_onnx_ = std::move(executor_by_onnx_ptr);
}

std::string TritonOpExecutor::GetConfigJson() {
  ORT_ENFORCE(IsInitialized());
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  PythonObjectPtr ret(PyObject_CallObject(config_getter_.get(), nullptr), PythonObjectDeleter);
  char* buffer = nullptr;
  Py_ssize_t length;
  buffer = const_cast<char*>(PyUnicode_AsUTF8AndSize(ret.get(), &length));
  return std::string(buffer, length);
}

void TritonOpExecutor::ExecuteByOnnx(int64_t onnx_key, const std::string& onnx_string,
                                     const InlinedVector<const OrtValue*>& inputs, InlinedVector<OrtValue>& outputs,
                                     const InlinedHashSet<size_t>& bool_outputs) {
  ORT_ENFORCE(IsInitialized());
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(2 + inputs.size())), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create args.");
  PyTuple_SetItem(args.get(), 0, PyLong_FromLongLong(static_cast<long long>(onnx_key)));
  PyTuple_SetItem(args.get(), 1, PyBytes_FromStringAndSize(onnx_string.c_str(), onnx_string.size()));
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i]) {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(2 + i), Py_None);
      Py_INCREF(Py_None);
    } else {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(2 + i), torch::ToDlpack(*inputs[i]));
    }
  }

  PythonObjectPtr ret(PyObject_CallObject(executor_by_onnx_.get(), args.get()), PythonObjectDeleter);
  if (ret == nullptr) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }
  ORT_ENFORCE(ret.get() != Py_None);
  if (PyTuple_Check(ret.get())) {
    for (size_t i = 0; i < static_cast<size_t>(PyTuple_Size(ret.get())); ++i) {
      outputs.emplace_back(torch::FromDlpack(PyTuple_GetItem(ret.get(), static_cast<Py_ssize_t>(i)),
                                             bool_outputs.find(i) != bool_outputs.end()));
    }
  } else {
    outputs.emplace_back(torch::FromDlpack(ret.get(), bool_outputs.find(0) != bool_outputs.end()));
  }
}

void TritonOpExecutor::ExecuteByFuncName(const std::string& func_name, const InlinedVector<const OrtValue*>& inputs,
                                         InlinedVector<OrtValue>& outputs, const InlinedHashSet<size_t>& bool_outputs,
                                         const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs) {
  ORT_ENFORCE(IsInitialized());
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(1 + inputs.size())), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create args.");
  PyTuple_SetItem(args.get(), 0, PyUnicode_FromString(func_name.c_str()));
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i]) {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(1 + i), Py_None);
      Py_INCREF(Py_None);
    } else {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(1 + i), torch::ToDlpack(*inputs[i]));
    }
  }

  PythonObjectPtr python_kwargs(PyDict_New(), PythonObjectDeleter);
  ORT_ENFORCE(python_kwargs, "Failed to create kwargs.");
  for (const auto& kv : kwargs) {
    if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
      std::string bool_str = kv.second.first;
      std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      int bool_value = bool_str == "" || bool_str == "false" || bool_str == "0" ? 0 : 1;
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyBool_FromLong(bool_value));
    } else if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyLong_FromLongLong(std::stoll(kv.second.first)));
    } else if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyFloat_FromDouble(std::stod(kv.second.first)));
    } else if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyUnicode_FromString(kv.second.first.c_str()));
    } else {
      ORT_THROW("Unsupported kwargs data type: ", kv.second.second);
    }
  }
  PythonObjectPtr ret(PyObject_Call(executor_by_name_.get(), args.get(), python_kwargs.get()), PythonObjectDeleter);
  if (ret == nullptr) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }
  if (ret.get() == Py_None) return;
  if (PyTuple_Check(ret.get())) {
    for (size_t i = 0; i < static_cast<size_t>(PyTuple_Size(ret.get())); ++i) {
      outputs.emplace_back(torch::FromDlpack(PyTuple_GetItem(ret.get(), static_cast<Py_ssize_t>(i)),
                                             bool_outputs.find(i) != bool_outputs.end()));
    }
  } else {
    outputs.emplace_back(torch::FromDlpack(ret.get(), bool_outputs.find(0) != bool_outputs.end()));
  }
}

}  // namespace triton
}  // namespace framework
}  // namespace training
}  // namespace onnxruntime
