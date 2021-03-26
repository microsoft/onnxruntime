// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "torch_custom_function_kernel.h"
#include "core/language_interop_ops/pyop/pyop_lib_proxy.h"
#include "core/torch_custom_function/torch_custom_function_register.h"
#include <thread>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

Status PythonOp::Compute(OpKernelContext* context) const {
  ORT_ENFORCE(context);
  std::cout << "std::this_thread::get_id() is : " << std::this_thread::get_id() << std::endl;
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(context);
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  std::vector<const OrtValue*> inputs;
  std::vector<void*> outputs;

  for (size_t i = 0; i < inputs_count; ++i) {
    inputs.push_back(ctx_internal->GetInputMLValue(i));
  }

  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  auto py_func = onnxruntime::python::OrtTorchFunctionPool::GetInstance().GetForward(name_);
  //   auto log_func = [&](const char* msg) {
  //     std::cout << "InvokePythonAutoGradFunc logging:" << msg << std::endl;
  //     //LOGS_DEFAULT(WARNING) << msg << std::endl;
  //   };
  ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonAutoGradFunc(reinterpret_cast<void*>(py_func), inputs, outputs),
              PyOpLibProxy::GetInstance().GetLastErrorMessage(err));
  PyOpLibProxy::GetInstance().PutGil(state);

  // We had the assumption:
  // The 1st output is context index of auto grad function.
  // The 2nd output is address of OrtValue we got from Python script run.
  PyObject* ctx_addr = reinterpret_cast<PyObject*>(outputs[0]);
  ORT_ENFORCE(ctx_addr, "Context object pointer should not be null");
  size_t ctx_index = onnxruntime::python::OrtTorchFunctionPool::GetInstance().RegisterContext(ctx_addr);

  Tensor* first_output_tensor = context->Output(0, {1});
  ORT_ENFORCE(first_output_tensor != nullptr, "first_output_tensor should not be null.");
  size_t* step_data_new = first_output_tensor->template MutableData<size_t>();
  *step_data_new = ctx_index;

  void* forward_ret_ortvalue_addr = outputs[1];
  auto* forward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(forward_ret_ortvalue_addr);
  ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(1, *forward_ret_ortvalue_ptr));
  return Status::OK();
}

Status PythonOpGrad::Compute(OpKernelContext* context) const {
  ORT_ENFORCE(context);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
