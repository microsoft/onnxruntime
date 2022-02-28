// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include <unordered_map>

namespace onnxruntime {
namespace eager {

onnxruntime::Status CreateEagerKernel(const void* info,
                                      const char* op_name,
                                      const char* domain,
                                      const int& version,
                                      const char** type_constraint_names,
                                      const int* type_constraint_values,
                                      const int& num_type_constraint,
                                      const void* attrs,
                                      const int& num_attrs,
                                      void** kernel) {
  if (!kernel) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid kernel pointer");
  }
  *kernel = nullptr;
  auto kernel_info = reinterpret_cast<const OpKernelInfo*>(info);
  auto ep = reinterpret_cast<const IExecutionProvider*>(kernel_info->GetExecutionProvider());
  auto kernel_registry = ep->GetKernelRegistry();
  const KernelCreateInfo* kernel_create_info{};
  std::unordered_map<std::string, MLDataType> type_constraint_map;
  for (int i = 0; i < num_type_constraint; ++i) {
    ONNX_NAMESPACE::TypeProto proto;
    proto.mutable_tensor_type()->set_elem_type(type_constraint_values[i]);
    type_constraint_map[type_constraint_names[i]] = DataTypeImpl::TypeFromProto(proto);
  }
  auto status = kernel_registry->TryFindKernel(op_name,
                                               domain,
                                               version,
                                               type_constraint_map,
                                               ep->Type(),
                                               &kernel_create_info);
  if (kernel_create_info == nullptr) {
    return status;
  }
  onnxruntime::Node node;
  auto onnx_attrs = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attrs);
  for (int i = 0; i < num_attrs; ++i) {
    node.AddAttribute(onnx_attrs[i].name(), onnx_attrs[i]);
  }
  OpKernelInfo eagar_kernel_info(node, KernelDef{}, *ep, {}, {}, {});
  std::unique_ptr<onnxruntime::OpKernel> op_kernel;
  FuncManager func_mgr;
  status = kernel_create_info->kernel_create_func(func_mgr, eagar_kernel_info, op_kernel);
  if (status.IsOK()) {
    *kernel = op_kernel.release();
  }
  return status;
}

onnxruntime::Status InvokeEagerKernel(const void* context,
                                      const void* kernel,
                                      const void* const* inputs,
                                      const int& input_len,
                                      void* const* outputs,
                                      const int& output_len) {
  auto ctx = reinterpret_cast<const OpKernelContext*>(context);
  AllocatorPtr allocator{};
  auto ret = ctx->GetTempSpaceAllocator(&allocator);
  if (!ret.IsOK()) {
    return ret;
  }
  EagerKernelContext eager_ctx(reinterpret_cast<const OrtValue* const*>(inputs),
                               input_len,
                               reinterpret_cast<OrtValue* const*>(outputs),
                               output_len,
                               allocator,
                               ctx->GetOperatorThreadPool(),
                               ctx->Logger());
  auto eager_kernel = reinterpret_cast<const OpKernel*>(kernel);
  ret = eager_kernel->Compute(&eager_ctx);
  return ret;
}

}  // namespace Eager
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateEagerKernel,
                    _In_ const void* kernel_info,
                    _In_ const char* op_name,
                    _In_ const char* domain,
                    _In_ const int& version,
                    _In_ const char** type_constraint_names,
                    _In_ const int* type_constraint_values,
                    _In_ const int& num_type_constraint,
                    _In_ const void* attrs,
                    _In_ const int& num_attrs,
                    _Outptr_ void** kernel) {
  API_IMPL_BEGIN
  auto status = onnxruntime::eager::CreateEagerKernel(kernel_info, op_name, domain, version, type_constraint_names, type_constraint_values, num_type_constraint, attrs, num_attrs, kernel);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to create eager kernel.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeEagerKernel,
                    _In_ const void* context,
                    _In_ const void* kernel,
                    _In_ const void* const* inputs,
                    _In_ const int& input_len,
                    _Inout_ void* const* outputs,
                    _In_ const int& output_len) {
  API_IMPL_BEGIN
  auto status = onnxruntime::eager::InvokeEagerKernel(context, kernel, inputs, input_len, outputs, output_len);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to invoke eager kernel.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseEagerKernel,
                    _In_ const void* kernel) {
  API_IMPL_BEGIN
  if (kernel) {
    auto eager_kernel = reinterpret_cast<const onnxruntime::OpKernel*>(kernel);
    delete eager_kernel;
  }
  return nullptr;
  API_IMPL_END
}
