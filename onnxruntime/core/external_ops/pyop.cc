// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/custom_ops_author.h"
#include "core/session/onnxruntime_c_api.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

class PyOp final: public OpKernel {
public:
    PyOp(const OpKernelInfo& info): OpKernel(info) {
        Py_Initialize();
        _import_array();
        std::string module, script;
        ORT_ENFORCE(info.GetAttr("module",   &module).IsOK(), "module not specified");
        ORT_ENFORCE(info.GetAttr("function", &script).IsOK(), "script not specified");
        PySys_SetPath(L".");
        pyModule_ = PyImport_ImportModule(module.c_str());
        ORT_ENFORCE(nullptr != pyModule_, "import python module failed");
        pyFunc_ = PyObject_GetAttrString(pyModule_, script.c_str());
        ORT_ENFORCE(nullptr != pyFunc_ && PyCallable_Check(pyFunc_), "script not callable");
    }

    Status Compute(OpKernelContext* context) const override {
        auto pyArgs = PyTuple_New(context->InputCount());
        for (int i = 0; i < context->InputCount(); ++i) {
            PyTuple_SetItem(pyArgs, i, FromTensor(context->Input<Tensor>(i)));
        }
        auto pyResult = PyEval_CallObject(pyFunc_, pyArgs);
        Py_DECREF(pyArgs);
        ORT_ENFORCE(PyArray_Check(pyResult));
        auto np_array = reinterpret_cast<PyArrayObject*>(pyResult);
        std::vector<int64_t> shape;
        for (int i = 0; i < PyArray_NDIM(np_array); ++i) {
            shape.push_back(PyArray_SHAPE(np_array)[i]);
        }
        auto output_tensor = context->Output(0, TensorShape(shape));
        ORT_ENFORCE(output_tensor->DataType() == DataTypeImpl::GetType<int32_t>(), "output type not int32_t");
        memcpy(output_tensor->MutableDataRaw(), PyArray_DATA(np_array), output_tensor->Size());
        Py_DECREF(pyResult);
        return Status::OK(); 
    }

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

    PyObject* FromTensor(const Tensor* tensor) const
    {
        ORT_ENFORCE(tensor->DataType() == DataTypeImpl::GetType<int32_t>(), "input type not int32_t");
        std::vector<npy_intp> dims(tensor->Shape().GetDims());
        auto obj = PyArray_EMPTY(dims.size(), dims.data(), NPY_INT32, 0);
        auto np_array = reinterpret_cast<PyArrayObject*>(obj);
        memcpy(PyArray_DATA(np_array), tensor->DataRaw(), tensor->Size());
        return PyArray_Return(np_array);
    }

    PyObject* pyModule_ = nullptr;
    PyObject* pyFunc_   = nullptr;
};

ORT_EXPORT KernelsContainer* GetAllKernels() {
  KernelsContainer* kc = new KernelsContainer;
  KernelDefBuilder def_builder;
  def_builder.SetName("PyOp")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(7)
      .Provider(onnxruntime::kCpuExecutionProvider);
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) -> OpKernel* { return new PyOp(info); };
  KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);
  kc->kernels_list.push_back(std::move(create_info));
  return kc;
}

ORT_EXPORT SchemasContainer* GetAllSchemas() {
  SchemasContainer* sc = new SchemasContainer;
  sc->domain = onnxruntime::kOnnxDomain;
  sc->baseline_opset_version = 5;
  sc->opset_version = 7;
  ONNX_NAMESPACE::OpSchema schema("PyOp", "unknown", 0);
  schema.Input(0,
               "A",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "B",
      "Second operand. With broadcasting can be of smaller size than A. "
      "If broadcasting is disabled it should be of the same size.",
      "T");
  schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(7);
  sc->schemas_list.push_back(schema);
  return sc;
}

ORT_EXPORT void FreeKernelsContainer(KernelsContainer* kc) {
  delete kc;
}

ORT_EXPORT void FreeSchemasContainer(SchemasContainer* sc) {
  delete sc;
}
