#include "python_op.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PyOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    PyOp);

PyObject* FromTensor(const Tensor* tensor)
{
    ORT_ENFORCE(tensor->DataType() == DataTypeImpl::GetType<int32_t>(), "input type not int32_t");
    std::vector<npy_intp> dims(tensor->Shape().GetDims());
    auto obj = PyArray_EMPTY(dims.size(), dims.data(), NPY_INT32, 0);
    auto np_array = reinterpret_cast<PyArrayObject*>(obj);
    memcpy(PyArray_DATA(np_array), tensor->DataRaw(), tensor->Size());
    return PyArray_Return(np_array);
}

Status PyOp::Compute(OpKernelContext* context) const 
{
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

}
}
