// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITONOP

#include "orttraining/training_ops/cpu/triton/triton_op.h"

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel_context_internal.h"
#endif
#include "orttraining/core/framework/torch/dlpack_python.h"
#include "orttraining/training_ops/cpu/triton/triton_op_executor.h"

namespace onnxruntime {
namespace contrib {

constexpr auto ToDlpack = training::framework::torch::ToDlpack;
constexpr auto FromDlpack = training::framework::torch::FromDlpack;

ONNX_OPERATOR_KERNEL_EX(TritonOp, kMSDomain, 1, kCpuExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        TritonOp);

bool TritonOp::IsBoolOutput(size_t index) const {
  ORT_ENFORCE(index < Node().OutputDefs().size(), "Output index out of range.");
  return Node().OutputDefs()[index]->TypeAsProto()->tensor_type().elem_type() ==
         ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL;
}

Status TritonOp::Compute(OpKernelContext* context) const {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());

  ORT_ENFORCE(TritonOpExecutor::Instance().IsInitialized());
  auto executor = TritonOpExecutor::Instance().GetExecutor();

  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(3 + input_size)), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create input tuple.");
  PyTuple_SetItem(args.get(), 0, PyUnicode_FromString(func_name_.c_str()));
  PyTuple_SetItem(args.get(), 1, PyLong_FromLongLong(static_cast<long long>(onnx_key_)));
  PyTuple_SetItem(args.get(), 2, PyBytes_FromStringAndSize(onnx_string_.c_str(), onnx_string_.size()));
  for (size_t i = 0; i < input_size; ++i) {
    const OrtValue* ort_value = p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    if (!ort_value) {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(i + 3), Py_None);
      Py_INCREF(Py_None);
    } else {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(i + 3),
                      ToDlpack(*p_ctx_internal->GetInputMLValue(static_cast<int>(i))));
    }
  }

  PythonObjectPtr ret(PyObject_CallObject(executor, args.get()), PythonObjectDeleter);
  if (PyTuple_Check(ret.get())) {
    ORT_ENFORCE(static_cast<size_t>(PyTuple_Size(ret.get())) == output_size, "Output size mismatch.");
    for (size_t i = 0; i < output_size; ++i) {
      ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(
          static_cast<int>(i), FromDlpack(PyTuple_GetItem(ret.get(), static_cast<Py_ssize_t>(i)), IsBoolOutput(i))));
    }
  } else {
    ORT_ENFORCE(output_size == 1, "Output size mismatch.");
    ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, FromDlpack(ret.get(), IsBoolOutput(0))));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ENABLE_TRITONOP
