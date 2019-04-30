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

class Scoper
{
public:
    Scoper() {
        mtx_.lock();
    }
    ~Scoper() {
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

std::mutex Scoper::mtx_;

PYOP_EXPORT bool Initialize() 
{
    Scoper scoper;
    Py_Initialize();
    if (_import_array() < 0) {
        return false;
    }
    auto path_list = PySys_GetObject("path");// do not release it
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
    Scoper scoper;
    if (PyErr_Occurred()) {
        stringstream ss;
        PyObject* type  = nullptr;
        PyObject* value = nullptr;
        PyObject* trace = nullptr;
        PyErr_Fetch(&type, &value, &trace);
        scoper.Add(type);
        scoper.Add(value);
        scoper.Add(trace);
        if (nullptr != type)  ss << "python error type: "  << PyBytes_AsString(type)  << endl;
        if (nullptr != value) ss << "python error value: " << PyBytes_AsString(value) << endl;
        if (nullptr != trace) ss << "python error trace: " << PyBytes_AsString(trace) << endl;
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
                    vector<const void*>&     outputs,
                    vector<int32_t>&         outputs_size,
                    vector<vector<int64_t>>& outputs_dim)
{
    if (!PyArray_Check(pyObj)) {
        return false;
    }

    outputs_dim.push_back({});
    auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
    outputs_size.push_back(static_cast<int32_t>(PyArray_ITEMSIZE(np_array)));

    for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
        outputs_dim.back().push_back(PyArray_SHAPE(np_array)[i]);
    }

    auto data_len = std::accumulate(begin(outputs_dim.back()),
                                    end(outputs_dim.back()),
                                    static_cast<int64_t>(outputs_size.back()),
                                    std::multiplies<int64_t>());

    outputs.push_back(new char[data_len]);
    memcpy(const_cast<void*>(outputs.back()), PyArray_DATA(np_array), data_len);
    return true;
}

PYOP_EXPORT void* NewInstance (const char* module, const char* class_name, const unordered_map<string, string>& args)
{
    Scoper scoper; 
    auto pyModule = PyImport_ImportModule(module);
    if (nullptr == pyModule) {
        return nullptr;
    }

    scoper.Add(pyModule);
    auto pyClass  = PyObject_GetAttrString(pyModule, class_name);
    if (nullptr == pyClass) {
        return nullptr;
    }

    scoper.Add(pyClass);
    auto empty_args = PyTuple_New(0);
    scoper.Add(empty_args);
    auto named_args = PyDict_New();
    scoper.Add(named_args);
    for (const auto& iter: args) {
        PyDict_SetItemString(named_args, iter.first.c_str(), PyUnicode_FromString(iter.second.c_str()));
    }

    return PyObject_Call(pyClass, empty_args, named_args);
}

PYOP_EXPORT void ReleaseInstance (void* instance)
{
    Scoper scoper;
    if (nullptr != instance) {
        Py_XDECREF(static_cast<PyObject*>(instance));
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
    Scoper scoper;
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

    scoper.Add(pyFunc);
    auto pyArgs = PyTuple_New(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        PyTuple_SetItem(pyArgs, i, MakePyObj(input[i], input_type[i], input_dim[i]));
    }

    scoper.Add(pyArgs);
    auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
    if (nullptr == pyResult) {
        logging_func("InvokePythonFunc: no result");
        return false;
    }

    scoper.Add(pyResult);
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
