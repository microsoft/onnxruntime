// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <functional>
#include <iostream>
#include <sstream>
#include <numeric>
#include <vector>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <stdarg.h>

using Releaser = std::function<void()>;

using namespace std;
namespace PythonFuncionWrapper {

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

class Locker
{
    static std::mutex mtx_;
public:
    Locker()  { mtx_.lock();   }
    ~Locker() { mtx_.unlock(); }
};

std::mutex Locker::mtx_;

PYOP_EXPORT bool Initialize() 
{
    Py_Initialize();
    if (_import_array() < 0) {
        return false;
    }
    auto path_list = PySys_GetObject("path");
    if (nullptr == path_list || !PyList_Check(path_list)) {
        return false;
    } else {
       if (PyList_Append(path_list, PyUnicode_FromString(".")) != 0) {
           return false;
       } 
    }
    static Finalizer finalizer;
    return true;
}

PYOP_EXPORT const char* GetLastErrorMessage(std::string& err)
{
    stringstream ss;
    if (PyErr_Occurred()) {
        PyObject *type, *value, *trace;
        PyErr_Fetch(&type, &value, &trace);
        ss << "type: "  << PyBytes_AsString(type)  << endl;
        ss << "value: " << PyBytes_AsString(value) << endl;
        ss << "trace: " << PyBytes_AsString(trace) << endl;
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(trace);
        err = ss.str();
    }
    return err.c_str();
}

PyObject* MakePyObj (const void* data, int32_t type, const vector<int64_t>& dim)
{
    std::vector<npy_intp> np_dim(dim);
    auto pyObj = static_cast<PyObject*>(PyArray_EMPTY(static_cast<int>(np_dim.size()), np_dim.data(), type, 0));
    auto data_len = std::accumulate(begin(np_dim), end(np_dim),
                                    static_cast<int64_t>(PyArray_DescrFromType(type)->elsize),
                                    std::multiplies<int64_t>());
    auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
    memcpy(PyArray_DATA(np_array), data, data_len);
    return pyObj;
}

bool ExtractOutput (PyObject*                pyObj,
                    vector<const void*>&     output,
                    vector<int32_t>&         output_size,
                    vector<vector<int64_t>>& output_dim)
{
    if (!PyArray_Check(pyObj)) {
        return false;
    }

    output_dim.push_back({});
    auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
    output_size.push_back(static_cast<int32_t>(PyArray_ITEMSIZE(np_array)));

    for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
        output_dim.back().push_back(PyArray_SHAPE(np_array)[i]);
    }

    auto data_len = std::accumulate(begin(output_dim.back()),
                                    end(output_dim.back()),
                                    static_cast<int64_t>(output_size.back()),
                                    std::multiplies<int64_t>());

    auto data = new char[data_len];
    memcpy(data, PyArray_DATA(np_array), data_len);
    output.push_back(data);
    return true;
}

PYOP_EXPORT void* NewInstance (const char* module, const char* class_name, const unordered_map<string, string>& args)
{
    Locker locker;
    vector<PyObject*> allocated;
    Releaser releaser = [&allocated] () { for (auto obj: allocated) Py_XDECREF(obj); };
  
    auto pyModule = PyImport_ImportModule(module);
    if (nullptr == pyModule) {
        return nullptr;
    }

    allocated.push_back(pyModule);
    auto pyClass  = PyObject_GetAttrString(pyModule, class_name);
    if (nullptr == pyClass) {
        return nullptr;
    }

    allocated.push_back(pyClass);
    auto empty_args = PyTuple_New(0);
    allocated.push_back(empty_args);
    auto named_args = PyDict_New();
    allocated.push_back(named_args);
    for (const auto& iter: args) {
        PyDict_SetItemString(named_args, iter.first.c_str(), PyUnicode_FromString(iter.second.c_str()));
    }

    auto instance = PyObject_Call(pyClass, empty_args, named_args);
    if (nullptr == instance) {
        return nullptr;
    }

    return instance;
}

PYOP_EXPORT void ReleaseInstance (void* instance)
{
    Locker locker;
    if (nullptr != instance) {
        Py_XDECREF(static_cast<PyObject*>(instance));
        instance = nullptr;
    }
}

PYOP_EXPORT bool InvokePythonFunc (void*                            raw_inst,
                                   const char*                      function,
                                   const vector<const void*>&       input,
                                   const vector<int32_t>&           input_type,
                                   const vector<vector<int64_t>>&   input_dim,
                                   vector<const void*>&             output,
                                   vector<int32_t>&                 output_size,
                                   vector<vector<int64_t>>&         output_dim,
                                   std::function<void(const char*)> logging_func = [](const char*){})
{
    Locker locker;
    auto instance = static_cast<PyObject*>(raw_inst);
    if (nullptr == instance || nullptr == function) {
        logging_func("InvokePythonFunc: found invalid instance or function");
        return false;
    }

    vector<PyObject*> allocated;
    Releaser releaser = [&allocated] () { for (auto obj: allocated) Py_XDECREF(obj); };

    auto pyFunc = PyObject_GetAttrString(instance, function);
    if (nullptr == pyFunc) {
        logging_func("InvokePythonFunc: failed to create function object");
        return false;
    }

    allocated.push_back(pyFunc);
    auto pyArgs = PyTuple_New(input.size());
    for (int32_t i = 0; i < input.size(); ++i) {
        PyTuple_SetItem(pyArgs, i, MakePyObj(input[i], input_type[i], input_dim[i]));
    }

    allocated.push_back(pyArgs);
    auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
    if (nullptr == pyResult) {
        logging_func("InvokePythonFunc: no result");
        return false;
    }

    allocated.push_back(pyResult);

    if (PyArray_Check(pyResult)) {
        ExtractOutput(pyResult, output, output_size, output_dim);
    } else if (PyTuple_Check(pyResult)) {
        for (int32_t i = 0; i < PyTuple_Size(pyResult); ++i) {
            if (!ExtractOutput(PyTuple_GetItem(pyResult, i), output, output_size, output_dim)) {
                logging_func("InvokePythonFunc: failed to extrace output");
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
