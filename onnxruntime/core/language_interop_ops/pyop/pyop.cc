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

using namespace std;

namespace onnxruntime {

PyOpLibProxy& PyOpLibProxy::GetInstance() {
  static PyOpLibProxy proxy;
  return proxy;
}

class Scope
{
public:
    Scope(const vector<PyObject*>& objs = {}): objs_(objs) {
        mtx_.lock();
    }
    ~Scope() {
        for (auto obj: objs_) {
            Py_XDECREF(obj);
        }
        mtx_.unlock();
    }
    void Add(PyObject* obj) {
        objs_.push_back(obj);
    }
private:
    static std::mutex mtx_;
    vector<PyObject*> objs_;
};

PyOpLibProxy::PyOpLibProxy() {
    Scope scope;
    Py_Initialize();
    if (_import_array() < 0) {
        return;
    }
    auto path_list = PySys_GetObject("path");//do not release it
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

std::mutex Scope::mtx_;

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

PyObject* MakePyObj(const void* data, int32_t type, const vector<int64_t>& dim) {
    std::vector<npy_intp> np_dim;
    for (auto d: dim) {
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

bool ExtractOutput(PyObject*                   pyObj,
                   vector<unique_ptr<char[]>>& outputs,
                   vector<int32_t>&            outputs_elem_size,
                   vector<vector<int64_t>>&    outputs_dim) {
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

    outputs.push_back(unique_ptr<char[]>(new char[data_len]));
    memcpy(static_cast<void*>(outputs.back().get()), PyArray_DATA(np_array), data_len);
    return true;
}

void* PyOpLibProxy::NewInstance(const char* module, const char* class_name, const unordered_map<string, string>& args) {
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
    for (const auto& iter: args) {
        PyDict_SetItemString(named_args, iter.first.c_str(), PyUnicode_FromString(iter.second.c_str()));
    }

    return PyObject_Call(pyClass, empty_args, named_args);
}

void PyOpLibProxy::ReleaseInstance(void* instance) {
    Scope scope({static_cast<PyObject*>(instance)});
}

bool PyOpLibProxy::InvokePythonFunc(void*                            raw_inst,
                                    const char*                      function,
                                    const vector<const void*>&       inputs,
                                    const vector<int32_t>&           inputs_type,
                                    const vector<vector<int64_t>>&   inputs_dim,
                                    vector<unique_ptr<char[]>>&      outputs,
                                    vector<int32_t>&                 outputs_elem_size,
                                    vector<vector<int64_t>>&         outputs_dim,
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
        PyTuple_SetItem(pyArgs, i, MakePyObj(inputs[i], inputs_type[i], inputs_dim[i]));
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
}//bool InvokePythonFunc

PyCustomKernel::PyCustomKernel(Ort::CustomOpApi ort,
                               const OnnxAttrs& attrs,
                               const std::string& module,
                               const std::string& class_name,
                               const std::string& compute,
                               PyOpLogFunc logging_func) : ort_(ort), attrs_(attrs), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) {
  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  ORT_ENFORCE(PyOpLibProxy::GetInstance().Initialized(), "Py library not properly initialized.");
  instance_ = PyOpLibProxy::GetInstance().NewInstance(module.c_str(), class_name_.c_str(), attrs_);
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

void PyCustomKernel::Compute(OrtKernelContext* context) {
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t) reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
  std::vector<const void*> inputs;
  std::vector<std::unique_ptr<char[]>> outputs;
  std::vector<int32_t> inputs_type, outputs_elem_size;
  std::vector<std::vector<int64_t>> inputs_dim, outputs_dim;

  for (size_t i = 0; i < inputs_count; ++i) {
    auto ort_value = ort_.KernelContext_GetInput(context, i);
    inputs.push_back(const_cast<MLValue*>(ort_value)->Get<Tensor>().DataRaw());
    inputs_type.push_back(GetType(ort_value));
    inputs_dim.push_back(const_cast<MLValue*>(ort_value)->Get<Tensor>().Shape().GetDims());
  }

  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonFunc(instance_, compute_.c_str(), inputs, inputs_type,
                                                           inputs_dim, outputs, outputs_elem_size,
                                                           outputs_dim, logging_func_),
              PyOpLibProxy::GetInstance().GetLastErrorMessage(err));  //ORT_ENFORCE
  PyOpLibProxy::GetInstance().PutGil(state);

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto ort_output = ort_.KernelContext_GetOutput(context, i, outputs_dim[i].data(), outputs_dim[i].size());
    auto output_mem_addr = ort_.GetTensorMutableData<char>(ort_output);
    auto output_len = std::accumulate(begin(outputs_dim[i]), end(outputs_dim[i]), static_cast<int64_t>(outputs_elem_size[i]), std::multiplies<int64_t>());
    memcpy(output_mem_addr, outputs[i].get(), output_len);
  }
}

int32_t PyCustomKernel::GetType(const OrtValue* input) const {
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

PyCustomOp::PyCustomOp(const OnnxAttrs& attrs,
                       const OnnxTypes& inputs_type,
                       const OnnxTypes& outputs_type,
                       const std::string& module,
                       const std::string& class_name,
                       const std::string& compute,
                       PyOpLogFunc logging_func) : attrs_(attrs), inputs_type_(inputs_type), outputs_type_(outputs_type), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) { OrtCustomOp::version = ORT_API_VERSION; }

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
