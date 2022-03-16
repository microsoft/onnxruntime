// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include <unordered_map>

#ifdef ORT_MINIMAL_BUILD

ORT_API_STATUS_IMPL(OrtApis::CreateAttribute,
                    _In_ const char*,
                    _In_ const void*,
                    _In_ int,
                    _In_ ONNXTensorElementDataType,
                    _In_ int,
                    _Out_ OrtOpAttr*) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "CreateAttribute is unimplemented for minimal build.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseAttribute, _Inout_ OrtOpAttr*) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "ReleaseAttribute is unimplemented for minimal build.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateOperator,
                    _In_ const OrtKernelInfo*,
                    _In_ const char*,
                    _In_ const char*,
                    _In_ int,
                    _In_ const char**,
                    _In_ const ONNXTensorElementDataType*,
                    _In_ int,
                    _In_ const OrtOpAttr*,
                    _In_ int,
                    _Out_ OrtOp*) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "CreateOperator is unimplemented for minimal build.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeOperator,
                    _In_ const OrtKernelContext*,
                    _In_ const OrtOp,
                    _In_ const OrtValue* const*,
                    _In_ int,
                    _Inout_ OrtValue* const*,
                    _In_ int) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "InvokeOperator is unimplemented for minimal build.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseOperator, _Inout_ OrtOp*) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "ReleaseOperator is unimplemented for minimal build.");
  API_IMPL_END
}

#else

namespace onnxruntime {
namespace instant {

onnxruntime::Status CreateAttribute(const char* name, const void* data, int len, ONNXTensorElementDataType type, int is_array, OrtOpAttr* op_attr) {
  auto attr = std::make_unique<ONNX_NAMESPACE::AttributeProto>();
  attr->set_name(std::string{name});
  if (type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto floats = reinterpret_cast<const float*>(data);
    if (is_array) {
      for (int j = 0; j < len; ++j) {
        attr->add_floats(floats[j]);
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);
    } else {
      attr->set_f(floats[0]);
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
    }
  } else if (type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    auto ints = reinterpret_cast<const int*>(data);
    if (is_array) {
      for (int j = 0; j < len; ++j) {
        attr->add_ints(ints[j]);
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
    } else {
      attr->set_i(ints[0]);
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    }
  } else if (type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    if (is_array) {
      using const_chars = const char*;
      auto ss = reinterpret_cast<const const_chars*>(data);
      for (int j = 0; j < len; ++j) {
        attr->add_strings(std::string{ss[j]});
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS);
    } else {
      auto str = reinterpret_cast<const char*>(data);
      attr->set_s(std::string{str});
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
    }
  } else {
    return Status(common::ONNXRUNTIME, common::FAIL, "Invalid attribute data type.");
  }
  *op_attr = reinterpret_cast<OrtOpAttr>(attr.release());
  return Status::OK();
}

onnxruntime::Status CreateOperator(const OrtKernelInfo* info,
                                   const char* op_name,
                                   const char* domain,
                                   int version,
                                   const char** type_constraint_names,
                                   const ONNXTensorElementDataType* type_constraint_values,
                                   int type_constraint_count,
                                   const OrtOpAttr* attr_values,
                                   int attr_count,
                                   OrtOp* op) {
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
  ORT_RETURN_IF_ERROR(status);
  onnxruntime::Node node;
  for (int i = 0; i < attr_count; ++i) {
    auto attr_proto = reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(attr_values[i]);
    node.AddAttributeProto(*attr_proto);
  }
  OpKernelInfo instant_kernel_info(node, KernelDef{}, *ep, {}, {}, {});
  std::unique_ptr<onnxruntime::OpKernel> op_kernel;
  FuncManager func_mgr;
  status = kernel_create_info->kernel_create_func(func_mgr, instant_kernel_info, op_kernel);
  ORT_RETURN_IF_ERROR(status);
  *op = op_kernel.release();
  return status;
}

onnxruntime::Status InvokeOperator(_In_ const OrtKernelContext* context,
                                   _In_ const OrtOp ort_op,
                                   _In_ const OrtValue* const* input_values,
                                   _In_ int input_count,
                                   _Inout_ OrtValue* const* output_values,
                                   _In_ int output_count) {
  auto ctx = reinterpret_cast<const OpKernelContext*>(context);
  AllocatorPtr allocator{};
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  InstantKernelContext instant_ctx(input_values,
                                   input_count,
                                   output_values,
                                   output_count,
                                   allocator,
                                   ctx->GetOperatorThreadPool(),
                                   ctx->Logger());
  auto kernel = reinterpret_cast<const OpKernel*>(ort_op);
  return kernel->Compute(&instant_ctx);
}

}  // namespace instant
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateAttribute,
                    _In_ const char* name,
                    _In_ const void* data,
                    _In_ int len,
                    _In_ ONNXTensorElementDataType type,
                    _In_opt_ int is_array,
                    _Outptr_ OrtOpAttr* op_attr) {
  API_IMPL_BEGIN
  if (!name || !data || !len || !op_attr) {
    return CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument for CreateAttribute.");
  }
  auto status = onnxruntime::instant::CreateAttribute(name, data, len, type, is_array, op_attr);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to create attribute.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseAttribute, _Inout_ OrtOpAttr* op_attr) {
  API_IMPL_BEGIN
  if (op_attr && *op_attr) {
    delete reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(*op_attr);
    *op_attr = nullptr;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateOperator,
                    _In_ const OrtKernelInfo* info,
                    _In_ const char* op_name,
                    _In_ const char* domain,
                    _In_ int version,
                    _In_opt_ const char** type_constraint_names,
                    _In_opt_ const ONNXTensorElementDataType* type_constraint_values,
                    _In_opt_ int type_constraint_count,
                    _In_opt_ const OrtOpAttr* attr_values,
                    _In_opt_ int attr_count,
                    _Outptr_ OrtOp* ort_op) {
  API_IMPL_BEGIN
  if (!info || !op_name || !domain || !ort_op) {
    return CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument for CreateOperator.");
  }
  auto status = onnxruntime::instant::CreateOperator(info,
                                                     op_name,
                                                     domain,
                                                     version,
                                                     type_constraint_names,
                                                     type_constraint_values,
                                                     type_constraint_count,
                                                     attr_values,
                                                     attr_count,
                                                     ort_op);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to create operator.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeOperator,
                    _In_ const OrtKernelContext* context,
                    _In_ const OrtOp ort_op,
                    _In_ const OrtValue* const* input_values,
                    _In_ int input_count,
                    _Inout_ OrtValue* const* output_values,
                    _In_ int output_count) {
  API_IMPL_BEGIN
  if (!context || !ort_op || !input_values || !input_count || !output_values || !output_count) {
    return CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument.");
  }
  auto status = onnxruntime::instant::InvokeOperator(context, ort_op, input_values, input_count, output_values, output_count);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), "Failed to invoke operator.");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseOperator, _Inout_ OrtOp* op) {
  API_IMPL_BEGIN
  if (op && *op) {
    delete reinterpret_cast<const onnxruntime::OpKernel*>(*op);
    *op = nullptr;
  }
  return nullptr;
  API_IMPL_END
}

#endif
