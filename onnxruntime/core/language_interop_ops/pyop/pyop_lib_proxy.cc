// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "pyop_lib_proxy.h"
#include "core/platform/env.h"
#include "core/util/dlpack_convertor.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "core/torch_custom_function/torch_custom_function_register.h"

namespace onnxruntime {

class Scope {
 public:
  Scope(const std::vector<PyObject*>& objs = {}) : objs_(objs) {
    mtx_.lock();
  }
  ~Scope() {
    for (auto obj : objs_) {
      Py_XDECREF(obj);
    }
    mtx_.unlock();
  }
  void Add(PyObject* obj) {
    objs_.push_back(obj);
  }

 private:
  static std::mutex mtx_;
  std::vector<PyObject*> objs_;
};

std::mutex Scope::mtx_;

int32_t GetType(const OrtValue* input) {
  int32_t numpy_type;
  ORT_ENFORCE(nullptr != input);
  ORT_ENFORCE(input->IsTensor(), "input must be a tensor");
  auto elem_type = input->Get<Tensor>().GetElementType();

  namespace on = ONNX_NAMESPACE;
  switch (elem_type) {
    case on::TensorProto_DataType_BOOL:
      numpy_type = 0;
      break;
    case on::TensorProto_DataType_INT8:
      numpy_type = 1;
      break;
    case on::TensorProto_DataType_UINT8:
      numpy_type = 2;
      break;
    case on::TensorProto_DataType_INT16:
      numpy_type = 3;
      break;
    case on::TensorProto_DataType_UINT16:
      numpy_type = 4;
      break;
    case on::TensorProto_DataType_INT32:
      numpy_type = 5;
      break;
    case on::TensorProto_DataType_UINT32:
      numpy_type = 6;
      break;
    case on::TensorProto_DataType_INT64:
      numpy_type = 9;
      break;
    case on::TensorProto_DataType_UINT64:
      numpy_type = 10;
      break;
    case on::TensorProto_DataType_FLOAT:
      numpy_type = 11;
      break;
    case on::TensorProto_DataType_DOUBLE:
      numpy_type = 12;
      break;
    default:
      ORT_THROW("Input primitive type not supported: ", DataTypeImpl::ToString(input->Get<Tensor>().DataType()));
  }
  return numpy_type;
}

PyObject* MakePyObj(const void* data, int32_t type, const std::vector<int64_t>& dim) {
  std::vector<npy_intp> np_dim;
  for (auto d : dim) {
    np_dim.push_back(static_cast<npy_intp>(d));
  }
  auto pyObj = static_cast<PyObject*>(PyArray_EMPTY(static_cast<int>(np_dim.size()), np_dim.data(), type, 0));
  auto data_len = std::accumulate(begin(np_dim), end(np_dim),
                                  static_cast<int64_t>(PyArray_DescrFromType(type)->elsize),
                                  std::multiplies<int64_t>());
  auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
  memcpy(PyArray_DATA(np_array), data, data_len);
  return pyObj;
}

bool ExtractOutput(PyObject* pyObj,
                   std::vector<std::unique_ptr<char[]>>& outputs,
                   std::vector<int32_t>& outputs_elem_size,
                   std::vector<std::vector<int64_t>>& outputs_dim) {
  if (!PyArray_Check(pyObj)) {
    return false;
  }

  outputs_dim.push_back({});
  auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
  outputs_elem_size.push_back(static_cast<int32_t>(PyArray_ITEMSIZE(np_array)));

  for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
    outputs_dim.back().push_back(PyArray_SHAPE(np_array)[i]);
  }

  auto data_len = std::accumulate(begin(outputs_dim.back()),
                                  end(outputs_dim.back()),
                                  static_cast<int64_t>(outputs_elem_size.back()),
                                  std::multiplies<int64_t>());

  outputs.push_back(std::unique_ptr<char[]>(new char[data_len]));
  memcpy(static_cast<void*>(outputs.back().get()), PyArray_DATA(np_array), data_len);
  return true;
}

void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlmanged_tensor) {
    // the dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // the dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

bool ExtractPointerOutput(PyObject* pyObj, std::vector<void*>& outputs) {
  void* prt = PyLong_AsVoidPtr(pyObj);
  // std::cout << "ExtractPointerOutput:" << prt << std::endl;
  outputs.push_back(prt);
  return true;
}

PyOpLibProxy& PyOpLibProxy::GetInstance() {
  static PyOpLibProxy proxy;
  return proxy;
}

PyOpLibProxy::PyOpLibProxy() {
  Scope scope;
  // in theory we shouldn't initialize it any more,
  // so comment it out currently.
  // Py_Initialize();
  if (_import_array() < 0) {
    return;
  }
  auto path_list = PySys_GetObject("path");  //do not release it
  if (nullptr == path_list || !PyList_Check(path_list) ||
      PyList_Append(path_list, PyUnicode_FromString(".")) != 0) {
    return;
  }
  initialized_ = true;
}

PyOpLibProxy::~PyOpLibProxy() {
  if (initialized_) {
    Py_Finalize();
  }
}

void* PyOpLibProxy::NewInstance(const char* module, const char* class_name, const std::unordered_map<std::string, std::string>& args) {
  Scope scope;
  auto pyModule = PyImport_ImportModule(module);
  if (nullptr == pyModule) {
    return nullptr;
  }

  scope.Add(pyModule);
  auto pyClass = PyObject_GetAttrString(pyModule, class_name);
  if (nullptr == pyClass) {
    return nullptr;
  }

  scope.Add(pyClass);
  auto empty_args = PyTuple_New(0);
  scope.Add(empty_args);
  auto named_args = PyDict_New();
  scope.Add(named_args);
  for (const auto& iter : args) {
    PyDict_SetItemString(named_args, iter.first.c_str(), PyUnicode_FromString(iter.second.c_str()));
  }

  return PyObject_Call(pyClass, empty_args, named_args);
}

void* PyOpLibProxy::NewInstance(void* py_class) {
  Scope scope;
  if (nullptr == py_class) {
    return nullptr;
  }
  PyObject* pyClass = reinterpret_cast<PyObject*>(py_class);
  auto empty_args = PyTuple_New(0);
  scope.Add(empty_args);
  auto named_args = PyDict_New();
  scope.Add(named_args);

  return PyObject_Call(pyClass, empty_args, named_args);
}

void PyOpLibProxy::ReleaseInstance(void* instance) {
  Scope scope({static_cast<PyObject*>(instance)});
}

const char* PyOpLibProxy::GetLastErrorMessage(std::string& err) {
  Scope scope;
  if (PyErr_Occurred()) {
    PyObject *type, *value, *trace;
    PyErr_Fetch(&type, &value, &trace);
    if (nullptr != value) {
      auto pyVal = PyObject_Repr(value);
      scope.Add(pyVal);
      auto pyStr = PyUnicode_AsEncodedString(pyVal, "utf-8", "Error ~");
      scope.Add(pyStr);
      err = PyBytes_AS_STRING(pyStr);
    }
    PyErr_Restore(type, value, trace);
  }
  return err.c_str();
}

int32_t PyOpLibProxy::GetGil() const {
  return PyGILState_Ensure();
}

void PyOpLibProxy::PutGil(int32_t state) const {
  PyGILState_Release((PyGILState_STATE)state);
}

bool PyOpLibProxy::InvokePythonFunc(void* raw_inst,
                                    const char* function,
                                    const std::vector<OrtValue*>& inputs,
                                    std::vector<std::unique_ptr<char[]>>& outputs,
                                    std::vector<int32_t>& outputs_elem_size,
                                    std::vector<std::vector<int64_t>>& outputs_dim,
                                    std::function<void(const char*)> logging_func) {
  Scope scope;
  auto instance = static_cast<PyObject*>(raw_inst);
  if (nullptr == instance || nullptr == function) {
    logging_func("InvokePythonFunc: found invalid instance or function");
    return false;
  }

  auto pyFunc = PyObject_GetAttrString(instance, function);
  if (nullptr == pyFunc) {
    logging_func("InvokePythonFunc: failed to create function object");
    return false;
  }

  scope.Add(pyFunc);
  auto pyArgs = PyTuple_New(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    PyTuple_SetItem(pyArgs, i, MakePyObj(inputs[i], GetType(inputs[i]), inputs[i]->Get<Tensor>().Shape().GetDims()));
  }

  scope.Add(pyArgs);
  auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
  if (nullptr == pyResult) {
    logging_func("InvokePythonFunc: no result");
    return false;
  }

  scope.Add(pyResult);
  if (PyArray_Check(pyResult)) {
    ExtractOutput(pyResult, outputs, outputs_elem_size, outputs_dim);
  } else if (PyTuple_Check(pyResult)) {
    for (int32_t i = 0; i < PyTuple_Size(pyResult); ++i) {
      if (!ExtractOutput(PyTuple_GetItem(pyResult, i), outputs, outputs_elem_size, outputs_dim)) {
        logging_func("InvokePythonFunc: failed to extract output");
        return false;
      }
    }
  } else {
    logging_func("InvokePythonFunc: returned value must be numpy(s)");
    return false;
  }
  return true;
}  //bool InvokePythonFunc

void PyOpLibProxy::InvokePythonFunction(void* function) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* callback = reinterpret_cast<PyObject*>(function);
  Py_INCREF(callback);
  PyObject_CallObject(callback, nullptr);
  PyGILState_Release(gstate);
}

bool PyOpLibProxy::InvokePythonFunc(const char* module,
                                    const char* function,
                                    const std::vector<const OrtValue*>& inputs,
                                    std::vector<void*>& outputs) {
  Scope scope;
  auto pyModule = PyImport_ImportModule(module);
  if (nullptr == pyModule) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: found invalid module";
    return false;
  }

  scope.Add(pyModule);

  if (nullptr == function) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: found invalid instance or function";
    return false;
  }

  auto pyFunc = PyObject_GetAttrString(pyModule, function);
  if (nullptr == pyFunc) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: failed to create function object";
    return false;
  }

  scope.Add(pyFunc);
  auto pyArgs = PyTuple_New(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    PyTuple_SetItem(pyArgs, i, MakePyObj(inputs[i], GetType(inputs[i]), inputs[i]->Get<Tensor>().Shape().GetDims()));
  }

  scope.Add(pyArgs);
  auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
  if (nullptr == pyResult) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: no result";
    return false;
  }

  scope.Add(pyResult);
  if (PyArray_Check(pyResult)) {
    ExtractPointerOutput(pyResult, outputs);
  } else if (PyTuple_Check(pyResult)) {
    for (int32_t i = 0; i < PyTuple_Size(pyResult); ++i) {
      if (!ExtractPointerOutput(PyTuple_GetItem(pyResult, i), outputs)) {
        LOGS_DEFAULT(WARNING) << "InvokePythonFunc: failed to extract output";
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: returned value must be numpy(s)";
    return false;
  }
  return true;
}  //bool InvokePythonFunc

bool PyOpLibProxy::InvokePythonAutoGradFunc(void* function,
                                            const std::vector<const OrtValue*>& inputs,
                                            std::vector<void*>& outputs) {
  Scope scope;
  PyObject* pyFunc;

  pyFunc = static_cast<PyObject*>(function);
  if (nullptr == function || nullptr == pyFunc) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: found invalid instance or function";
    return false;
  }

  scope.Add(pyFunc);
  auto pyArgs = PyTuple_New(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*inputs[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    PyTuple_SetItem(pyArgs, i, dltensor);
  }

  scope.Add(pyArgs);
  auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
  if (nullptr == pyResult) {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: no result";
    return false;
  }

  scope.Add(pyResult);
  if (PyTuple_Check(pyResult)) {
    for (int32_t i = 0; i < PyTuple_Size(pyResult); ++i) {
      if (!ExtractPointerOutput(PyTuple_GetItem(pyResult, i), outputs)) {
        LOGS_DEFAULT(WARNING) << "InvokePythonFunc: failed to extract address from output";
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "InvokePythonFunc: returned value must be numpy(s)";
    return false;
  }
  return true;
}

bool PyOpLibProxy::InvokePythonAutoGradFunc(void* raw_inst,
                                            const char* function,
                                            const std::vector<OrtValue*>& inputs,
                                            const std::vector<int64_t>& arg_positions,
                                            std::vector<void*>& outputs,
                                            std::function<void(const char*)> logging_func,
                                            std::vector<void*>& const_args,
                                            const std::vector<int64_t>& const_arg_positions) {
  Scope scope;
  auto instance = static_cast<PyObject*>(raw_inst);
  if (nullptr == instance || nullptr == function) {
    logging_func("InvokePythonFunc: found invalid instance or function");
    return false;
  }

  // Get member function.
  auto py_func = PyObject_GetAttrString(instance, function);
  if (nullptr == py_func) {
    logging_func("InvokePythonFunc: failed to create function object");
    return false;
  }

  // Number of input arguments in the called Python function.
  const size_t num_args = inputs.size() + const_args.size();

  // Input arguments to the called Python function.
  PyObject* py_args = PyTuple_New(num_args);

  // filled_flags[i] being true means the i-th input argument has been set.
  std::vector<bool> filled_flags(num_args, false);

  // Fill argument list with non-tensor types.
  for (size_t i = 0; i < const_args.size(); ++i) {
    // Let j = const_arg_positions.at(i).
    // Here we set the j-th input argument to const_args.at(i).
    PyTuple_SetItem(py_args, const_arg_positions.at(i), reinterpret_cast<PyObject*>(const_args.at(i)));
    std::cout << "[pyop_lib_proxy.cc] const_arg " << i << ": " << std::endl;
    PyObject_Print(reinterpret_cast<PyObject*>(const_args.at(i)), stdout, 0);
    std::cout << std::endl;
  }

  // Fill argument list with tensor types.
  for (size_t i = 0; i < inputs.size(); ++i) {
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*inputs[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    PyTuple_SetItem(py_args, arg_positions.at(i), dltensor);
    std::cout << "[pyop_lib_proxy.cc] arg " << i << ": " << std::endl;
    PyObject_Print(dltensor, stdout, 0);
    std::cout << std::endl;
  }

  scope.Add(py_args);

  // Call member function.
  PyObject* py_result = PyEval_CallObject(py_func, py_args);

  if (!py_result) {
    logging_func("InvokePythonFunc: no result");
    return false;
  }

  scope.Add(py_result);
  if (PyTuple_Check(py_result)) {
    for (int i = 0; i < PyTuple_Size(py_result); ++i) {
      if (!ExtractPointerOutput(PyTuple_GetItem(py_result, i), outputs)) {
        logging_func("InvokePythonFunc: failed to extract address from output");
        return false;
      }
    }
  } else {
    logging_func("InvokePythonFunc: returned value must be numpy(s)");
    return false;
  }

  return true;
}

void PyOpLibProxy::InvokeForward(
    void* callback,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<void*>& returned_args) {
  // Set tensor indicators.
  std::cout << "[PyOpLibProxy::InvokeForward] 1" << std::endl;
  const auto len = tensor_args.size() + obj_args.size();
  PyObject* tensor_flags = PyList_New(len);
  for (size_t i = 0; i < len; ++i) {
    PyObject* zero = PyLong_FromLong(0);
    PyList_SetItem(tensor_flags, i, zero);
  }
  std::cout << "[PyOpLibProxy::InvokeForward] 2" << std::endl;
  for (const auto i : tensor_indices) {
    PyObject* one = PyLong_FromLong(1);
    PyList_SetItem(tensor_flags, i, one);
  }

  // Store them in tuple.
  std::cout << "[PyOpLibProxy::InvokeForward] 3" << std::endl;
  auto args = PyTuple_New(2 + len);
  PyTuple_SetItem(args, 0, reinterpret_cast<PyObject*>(callback));
  PyTuple_SetItem(args, 1, tensor_flags);

  std::cout << "[PyOpLibProxy::InvokeForward] 4" << std::endl;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*tensor_args[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    PyTuple_SetItem(args, 2 + tensor_indices[i], dltensor);
  }

  std::cout << "[PyOpLibProxy::InvokeForward] 5" << std::endl;
  for (size_t i = 0; i < obj_args.size(); ++i) {
    PyTuple_SetItem(args, 2 + obj_indices[i], reinterpret_cast<PyObject*>(obj_args[i]));
  }

  PyObject* runner = onnxruntime::python::OrtTorchFunctionPool::GetInstance().GetForwardRunner();

  std::cout << "[PyOpLibProxy::InvokeForward] 6" << std::endl;
  PyObject* result = PyObject_CallObject(reinterpret_cast<PyObject*>(runner), args);

  std::cout << "[PyOpLibProxy::InvokeForward] 6-1" << std::endl;
  PyObject_Print(result, stdout, 0);
  std::cout << std::endl;

  std::string err;
  GetLastErrorMessage(err);
  std::cout << "Error: " << err << std::endl;

  std::cout << "[PyOpLibProxy::InvokeForward] 7" << std::endl;
  ORT_ENFORCE(PyTuple_Check(result), "Python function must return a tuple.");
  for (int i = 0; i < PyTuple_Size(result); ++i) {
    std::cout << "[PyOpLibProxy::InvokeForward] 7-1: elem " << i << std::endl;
    PyObject_Print(PyTuple_GetItem(result, i), stdout, 0);
    std::cout << std::endl;
    ORT_ENFORCE(ExtractPointerOutput(PyTuple_GetItem(result, i), returned_args));
  }

  GetLastErrorMessage(err);
  std::cout << "Error: " << err << std::endl;
  std::cout << "[PyOpLibProxy::InvokeForward] 8" << std::endl;
}

void PyOpLibProxy::InvokeBackward(
    void* callback,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<void*>& returned_args) {
  // Set tensor indicators.
  std::cout << "[PyOpLibProxy::InvokeBackward] 1" << std::endl;
  const auto len = tensor_args.size() + obj_args.size();
  PyObject* tensor_flags = PyList_New(len);
  for (size_t i = 0; i < len; ++i) {
    PyObject* zero = PyLong_FromLong(0);
    PyList_SetItem(tensor_flags, i, zero);
  }
  std::cout << "[PyOpLibProxy::InvokeBackward] 2" << std::endl;
  for (const auto i : tensor_indices) {
    PyObject* one = PyLong_FromLong(1);
    PyList_SetItem(tensor_flags, i, one);
  }

  std::cout << "[PyOpLibProxy::InvokeBackward] 3" << std::endl;
  // Store them in tuple.
  auto args = PyTuple_New(2 + len);
  PyTuple_SetItem(args, 0, reinterpret_cast<PyObject*>(callback));
  PyTuple_SetItem(args, 1, tensor_flags);

  std::cout << "[PyOpLibProxy::InvokeBackward] 4" << std::endl;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*tensor_args[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    PyTuple_SetItem(args, 2 + tensor_indices[i], dltensor);
  }

  std::cout << "[PyOpLibProxy::InvokeBackward] 5" << std::endl;
  for (size_t i = 0; i < obj_args.size(); ++i) {
    PyTuple_SetItem(args, 2 + obj_indices[i], reinterpret_cast<PyObject*>(obj_args[i]));
  }

  std::cout << "[PyOpLibProxy::InvokeBackward] 6" << std::endl;
  PyObject* runner = onnxruntime::python::OrtTorchFunctionPool::GetInstance().GetBackwardRunner();

  std::cout << "[PyOpLibProxy::InvokeBackward] 7" << std::endl;
  PyObject* result = PyObject_CallObject(reinterpret_cast<PyObject*>(runner), args);

  std::string err;
  GetLastErrorMessage(err);
  std::cout << "Error: " << err << std::endl;
  PyObject_Print(result, stdout, 0);
  std::cout << std::endl;

  std::cout << "[PyOpLibProxy::InvokeBackward] 8" << std::endl;
  ORT_ENFORCE(PyTuple_Check(result), "Python function must return a tuple.");
  for (int i = 0; i < PyTuple_Size(result); ++i) {
    ORT_ENFORCE(ExtractPointerOutput(PyTuple_GetItem(result, i), returned_args));
  }
  std::cout << "[PyOpLibProxy::InvokeBackward] 9" << std::endl;
}

}  // namespace onnxruntime