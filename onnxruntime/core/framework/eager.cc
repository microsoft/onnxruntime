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

onnxruntime::Status CreateEagerOperator(const OrtKernelInfo* info,
                                        const char* op_name,
                                        const char* domain,
                                        int version,
                                        const char** type_constraint_names,
                                        const ONNXTensorElementDataType* type_constraint_values,
                                        int type_constraint_count,
                                        const void* attr_values,
                                        int attr_count,
                                        OrtEagerOperator* op) {
  *op = nullptr;
  auto kernel_info = reinterpret_cast<const OpKernelInfo*>(info);
  auto ep = reinterpret_cast<const IExecutionProvider*>(kernel_info->GetExecutionProvider());
  auto kernel_registry = ep->GetKernelRegistry();
  const KernelCreateInfo* kernel_create_info{};
  std::unordered_map<std::string, MLDataType> type_constraint_map;
  for (int i = 0; i < type_constraint_count; ++i) {
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
  auto onnx_attrs = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attr_values);
  for (int i = 0; i < attr_count; ++i) {
    node.AddAttribute(onnx_attrs[i].name(), onnx_attrs[i]);
  }
  OpKernelInfo eagar_kernel_info(node, KernelDef{}, *ep, {}, {}, {});
  std::unique_ptr<onnxruntime::OpKernel> op_kernel;
  FuncManager func_mgr;
  status = kernel_create_info->kernel_create_func(func_mgr, eagar_kernel_info, op_kernel);
  if (status.IsOK()) {
    *op = op_kernel.release();
  }
  return status;
}

onnxruntime::Status InvokeEagerOperator(const OrtKernelContext* context,
                                        const OrtEagerOperator ort_op,
                                        const OrtValue* const* input_values,
                                        int input_count,
                                        OrtValue* const* output_values,
                                        int output_count) {
  auto ctx = reinterpret_cast<const OpKernelContext*>(context);
  AllocatorPtr allocator{};
  auto ret = ctx->GetTempSpaceAllocator(&allocator);
  if (!ret.IsOK()) {
    return ret;
  }
  EagerKernelContext eager_ctx(input_values,
                               input_count,
                               output_values,
                               output_count,
                               allocator,
                               ctx->GetOperatorThreadPool(),
                               ctx->Logger());
  auto kernel = reinterpret_cast<const OpKernel*>(ort_op);
  ret = kernel->Compute(&eager_ctx);
  return ret;
}

}  // namespace eager
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateEagerOperator,
                    _In_ const OrtKernelInfo* info,
                    _In_ const char* op_name,
                    _In_ const char* domain,
                    _In_ int version,
                    _In_ const char** type_constraint_names,
                    _In_ const ONNXTensorElementDataType* type_constraint_values,
                    _In_ int type_constraint_count,
                    _In_ const void* onnx_attr_values,
                    _In_ int onnx_attr_count,
                    _Out_ OrtEagerOperator* ort_op) {
  API_IMPL_BEGIN
  if (!info || !op_name || !domain || !ort_op) {
    return CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument.");
  }
  auto status = onnxruntime::eager::CreateEagerOperator(info,
                                                        op_name,
                                                        domain,
                                                        version,
                                                        type_constraint_names,
                                                        type_constraint_values,
                                                        type_constraint_count,
                                                        onnx_attr_values,
                                                        onnx_attr_count,
                                                        ort_op);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to create eager kernel.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeEagerOperator,
                    _In_ const OrtKernelContext* context,
                    _In_ const OrtEagerOperator ort_op,
                    _In_ const OrtValue* const* input_values,
                    _In_ int input_count,
                    _Inout_ OrtValue* const* output_values,
                    _In_ int output_count) {
  API_IMPL_BEGIN
  if (!context || !ort_op || !input_values || !input_count || !output_values || !output_count) {
    return CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument.");
  }
  auto status = onnxruntime::eager::InvokeEagerOperator(context, ort_op, input_values, input_count, output_values, output_count);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to invoke eager kernel.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseEagerOperator, _Inout_ OrtEagerOperator* op) {
  API_IMPL_BEGIN
  if (op && *op) {
    delete reinterpret_cast<const onnxruntime::OpKernel*>(*op);
    *op = nullptr;
  }
  return nullptr;
  API_IMPL_END
}