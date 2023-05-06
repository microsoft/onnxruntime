// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

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
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;

  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());

  ORT_ENFORCE(TritonOpExecutor::Instance().IsInitialized());
  bool call_by_name = func_name_ != "";
  auto executor = call_by_name ? TritonOpExecutor::Instance().GetExecutorByName()
                               : TritonOpExecutor::Instance().GetExecutorByOnnx();
  size_t extra_input_size = call_by_name ? 1 : 2;

  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(extra_input_size + input_size)), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create input tuple.");
  if (call_by_name) {
    PyTuple_SetItem(args.get(), 0, PyUnicode_FromString(func_name_.c_str()));
  } else {
    PyTuple_SetItem(args.get(), 0, PyLong_FromLongLong(static_cast<long long>(onnx_key_)));
    PyTuple_SetItem(args.get(), 1, PyBytes_FromStringAndSize(onnx_string_.c_str(), onnx_string_.size()));
  }
  for (size_t i = 0; i < input_size; ++i) {
    const OrtValue* ort_value = p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    if (!ort_value) {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(extra_input_size + i), Py_None);
      Py_INCREF(Py_None);
    } else {
      PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(extra_input_size + i), ToDlpack(*ort_value));
    }
  }

  PythonObjectPtr ret(PyObject_CallObject(executor, args.get()), PythonObjectDeleter);
  if (ret == nullptr) {
    PyErr_Print();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Python function execution fails with the above information.");
  }
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

bool IsTritonOpExecutorInitialized() { return TritonOpExecutor::Instance().IsInitialized(); }

Status ExecuteTritonOpByFuncName(OpKernelContext* p_ctx, const std::string& func_name, size_t input_count,
                                 size_t output_count,
                                 const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs) {
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(p_ctx);
  ORT_ENFORCE(TritonOpExecutor::Instance().IsInitialized());
  PythonObjectPtr args(PyTuple_New(static_cast<Py_ssize_t>(1 + input_count + output_count)), PythonObjectDeleter);
  ORT_ENFORCE(args, "Failed to create input tuple.");
  PyTuple_SetItem(args.get(), 0, PyUnicode_FromString(func_name.c_str()));
  for (size_t i = 0; i < input_count; ++i) {
    PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(1 + i),
                    ToDlpack(*p_ctx_internal->GetInputMLValue(static_cast<int>(i))));
  }
  for (size_t i = 0; i < output_count; ++i) {
    PyTuple_SetItem(args.get(), static_cast<Py_ssize_t>(1 + input_count + i),
                    ToDlpack(*p_ctx_internal->GetOutputMLValue(static_cast<int>(i))));
  }
  PythonObjectPtr python_kwargs(PyDict_New(), PythonObjectDeleter);
  ORT_ENFORCE(python_kwargs, "Failed to create kwargs.");
  for (const auto& kv : kwargs) {
    if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
      std::string bool_str = kv.second.first;
      std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      int bool_value = bool_str == "" || bool_str == "false" || bool_str == "0" ? 0 : 1;
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyBool_FromLong(bool_value));
    } else if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyLong_FromLongLong(std::stoll(kv.second.first)));
    } else if (kv.second.second == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      PyDict_SetItemString(python_kwargs.get(), kv.first.c_str(), PyFloat_FromDouble(std::stod(kv.second.first)));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported kwargs data type: ", kv.second.second);
    }
  }
  PythonObjectPtr ret(PyObject_Call(TritonOpExecutor::Instance().GetExecutorByName(), args.get(), python_kwargs.get()),
                      PythonObjectDeleter);
  if (ret == nullptr) {
    PyErr_Print();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Python function execution fails with the above information.");
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ENABLE_TRITON
