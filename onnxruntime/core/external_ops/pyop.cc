// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;
namespace PythonFuncionWrapper {

string last_error_message = "";

struct Finalizer
{
    ~Finalizer() {
        Py_Finalize();
    }
};

extern "C" bool Initialize() {
    Py_Initialize();
    if (_import_array() < 0) {
        last_error_message = "Failed to initialize numpy array";
        return false;
    }
    static Finalizer finalizer;
    return true;
}

extern "C" void SetSysPath(const wchar_t* dir) {
    PySys_SetPath(dir);
}

extern "C" const char* GetLastErrorMessage() {
    return last_error_message.c_str();
}

PyObject* MakePyObj(const void* data, int32_t type, const vector<int64_t>& dim) {
    size_t data_len = 0;
    std::vector<npy_intp> np_dim(dim);
    PyObject* pyObj = nullptr;
    switch (type) {
        case 0:
            pyObj = PyArray_EMPTY(np_dim.size(), np_dim.data(), NPY_INT32, 0);
            data_len = sizeof(int32_t) * std::accumulate(begin(dim), end(dim), 1, std::multiplies<int64_t>());
            break;
        default:
            break;
    }
    auto np_array = reinterpret_cast<PyArrayObject*>(pyObj);
    memcpy(PyArray_DATA(np_array), data, data_len);
    return pyObj;
}

extern "C"  bool CallPythonFunction (const char*                    module,
                                     const char*                    function,
                                     const vector<const void*>&     input,
                                     const vector<int32_t>&         input_type,
                                     const vector<vector<int64_t>>& input_dim,
                                     vector<const void*>&           output,
                                     vector<int32_t>&               output_type,
                                     vector<vector<int64_t>>&       output_dim) {
  
    auto pyModule = PyImport_ImportModule(module);
    if (nullptr == pyModule) {
        last_error_message = "Failed to import module";
        return false;
    }
    auto pyFunc = PyObject_GetAttrString(pyModule, function);
    if (nullptr == pyFunc) {
        last_error_message = "Failed to import function";
        return false;
    }
    if (!PyCallable_Check(pyFunc)) {
        last_error_message = "Function is not callable";
        return false;
    }

    auto pyArgs = PyTuple_New(input.size());
    for (int32_t i = 0; i < input.size(); ++i) {
        PyTuple_SetItem(pyArgs, i, MakePyObj(input[i], input_type[i], input_dim[i]));
    }

    auto pyResult = PyEval_CallObject(pyFunc, pyArgs);
    if (PyArray_Check(pyResult)) {
        output_type.push_back(0);
        output_dim.push_back({});
        auto np_array = reinterpret_cast<PyArrayObject*>(pyResult);
        for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
            output_dim.back().push_back(PyArray_SHAPE(np_array)[i]);
        }
        auto data_len = std::accumulate(begin(output_dim.back()), end(output_dim.back()), 1, std::multiplies<int64_t>());
        switch (output_type.back()) {
            case 0:
                data_len *= sizeof(int32_t);
                break;
            default:
                break;
        }
        auto data = new char[data_len];
        memcpy(data, PyArray_DATA(np_array), data_len);
        output.push_back(data);
    }

    Py_XDECREF(pyArgs);
    Py_XDECREF(pyResult);
    Py_XDECREF(pyModule);
    Py_XDECREF(pyFunc);
    return true;
}

} //namespace PythonFunctionWrapper
