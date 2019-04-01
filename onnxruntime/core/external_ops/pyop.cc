// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

using Releaser = std::function<void()>;

using namespace std;
namespace PythonFuncionWrapper {

string last_error_message = "";

struct Finalizer
{
    ~Finalizer() {
        Py_Finalize();
    }
};

extern "C" bool Initialize() 
{
    Py_Initialize();
    if (_import_array() < 0) {
        last_error_message = "Failed to initialize numpy array";
        return false;
    }
    static Finalizer finalizer;
    return true;
}

extern "C" void SetSysPath(const wchar_t* dir)
{
    PySys_SetPath(dir);
}

extern "C" const char* GetLastErrorMessage()
{
    return last_error_message.c_str();
}

PyObject* MakePyObj(const void* data, int32_t type, const vector<int64_t>& dim)
{
    std::vector<npy_intp> np_dim(dim);
    PyObject* pyObj = PyArray_EMPTY(np_dim.size(), np_dim.data(), type, 0);
    auto data_len = std::accumulate(begin(np_dim), end(np_dim),
                                    PyArray_DescrFromType(type)->elsize,
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
        last_error_message = "Returned object is not numpy";
        return false;
    }

    output_dim.push_back({});
    auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
    output_size.push_back(PyArray_ITEMSIZE(np_array));

    for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
        output_dim.back().push_back(PyArray_SHAPE(np_array)[i]);
    }

    auto data_len = output_size.back() *
                    std::accumulate(begin(output_dim.back()),
                                    end(output_dim.back()), 1,
                                    std::multiplies<int64_t>());

    auto data = new char[data_len];
    memcpy(data, PyArray_DATA(np_array), data_len);
    output.push_back(data);
    return true;
}

extern "C"  bool CallPythonFunction (const char*                    module,
                                     const char*                    function,
                                     const vector<const void*>&     input,
                                     const vector<int32_t>&         input_type,
                                     const vector<vector<int64_t>>& input_dim,
                                     vector<const void*>&           output,
                                     vector<int32_t>&               output_size,
                                     vector<vector<int64_t>>&       output_dim)
{
    vector<PyObject*> allocated;
    Releaser releaser = [&allocated] () { for (auto obj: allocated) Py_XDECREF(obj); };
  
    auto pyModule = PyImport_ImportModule(module);
    if (nullptr == pyModule) {
        last_error_message = "Failed to import module";
        return false;
    }

    allocated.push_back(pyModule);

    auto pyFunc = PyObject_GetAttrString(pyModule, function);
    if (nullptr == pyFunc) {
        last_error_message = "Failed to import function";
        return false;
    }

    allocated.push_back(pyFunc);

    if (!PyCallable_Check(pyFunc)) {
        last_error_message = "Function is not callable";
        return false;
    }

    auto pyArgs = PyTuple_New(input.size());
    for (int32_t i = 0; i < input.size(); ++i) {
        PyTuple_SetItem(pyArgs, i, MakePyObj(input[i], input_type[i], input_dim[i]));
    }

    allocated.push_back(pyArgs);

    auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
    allocated.push_back(pyResult);

    if (PyArray_Check(pyResult)) {
        ExtractOutput(pyResult, output, output_size, output_dim);
    } else if (PyTuple_Check(pyResult)) {
        for (int32_t i = 0; i < PyTuple_Size(pyResult); ++i) {
            if (!ExtractOutput(PyTuple_GetItem(pyResult, i), output, output_size, output_dim)) {
                return false;
            }
        }
    } else {
        last_error_message = "Returned object is not numpy";
        return false;
    }
    return true;
}

} //namespace PythonFunctionWrapper
