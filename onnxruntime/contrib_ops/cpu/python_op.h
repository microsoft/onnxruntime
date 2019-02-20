#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

namespace onnxruntime {
namespace contrib {

class PyOp final: public OpKernel {
public:
    PyOp (const OpKernelInfo& info): OpKernel(info) {
        Py_Initialize();
        _import_array();
        std::string module, script;
        ORT_ENFORCE(info.GetAttr("module",   &module).IsOK(), "module not specified");
        ORT_ENFORCE(info.GetAttr("function", &script).IsOK(), "script not specified");
        // PySys_SetPath(L"/home/randy/onnxruntime/build/Linux/Debug/");
        PySys_SetPath(L".");
        pyModule_ = PyImport_ImportModule(module.c_str());
        ORT_ENFORCE(nullptr != pyModule_, "import python module failed");
        pyFunc_ = PyObject_GetAttrString(pyModule_, script.c_str());
        ORT_ENFORCE(nullptr != pyFunc_ && PyCallable_Check(pyFunc_), "script not callable");
    }

    Status Compute(OpKernelContext*) const override;

    ~PyOp() {
        if (nullptr != pyModule_) {
            Py_XDECREF(pyModule_);
            pyModule_ = nullptr;
        }
        if (nullptr != pyFunc_) {
            Py_XDECREF(pyFunc_);
            pyFunc_ = nullptr;
        }
        Py_Finalize();
    }
private:
    PyObject* pyModule_ = nullptr;
    PyObject* pyFunc_ = nullptr;
};

}
}
