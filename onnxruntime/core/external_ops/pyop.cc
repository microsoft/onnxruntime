// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

#ifdef _WIN32
#define PYOP_EXPORT extern "C" __declspec(dllexport)
#else
#define PYOP_EXPORT extern "C"
#endif

struct Finalizer
{
    ~Finalizer() {
        Py_Finalize();
    }
};

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

std::mutex Scope::mtx_;

PYOP_EXPORT bool Initialize() 
{
    Scope scope;
    Py_Initialize();
    if (_import_array() < 0) {
        return false;
    }
    auto path_list = PySys_GetObject("path");// do not release it
    if (nullptr == path_list || !PyList_Check(path_list) ||
        PyList_Append(path_list, PyUnicode_FromString(".")) != 0) {
        return false; 
    }
    static Finalizer finalizer;
    return true;
}

PYOP_EXPORT const char* GetLastErrorMessage(std::string& err)
{
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

PyObject* MakePyObj(const void* data, int32_t type, const vector<int64_t>& dim)
{
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
                   vector<vector<int64_t>>&    outputs_dim)
{
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

PYOP_EXPORT void* NewInstance(const char* module, const char* class_name, const unordered_map<string, string>& args)
{
    Scope scope; 
    auto pyModule = PyImport_ImportModule(module);
    if (nullptr == pyModule) {
        return nullptr;
    }

    scope.Add(pyModule);
    auto pyClass  = PyObject_GetAttrString(pyModule, class_name);
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

PYOP_EXPORT void ReleaseInstance(void* instance)
{
    Scope scope({static_cast<PyObject*>(instance)});
}

PYOP_EXPORT bool InvokePythonFunc(void*                            raw_inst,
                                  const char*                      function,
                                  const vector<const void*>&       inputs,
                                  const vector<int32_t>&           inputs_type,
                                  const vector<vector<int64_t>>&   inputs_dim,
                                  vector<unique_ptr<char[]>>&      outputs,
                                  vector<int32_t>&                 outputs_elem_size,
                                  vector<vector<int64_t>>&         outputs_dim,
                                  std::function<void(const char*)> logging_func = [](const char*){})
{
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
}
} //namespace PythonFunctionWrapper
