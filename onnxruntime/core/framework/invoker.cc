// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/invoker.h"
#include "core/framework/data_types.h"
#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
//#include "core/session/onnxruntime_c_api.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include <unordered_map>

namespace onnxruntime {
namespace invoker {

class EagerKernelPool final {
 public:
  EagerKernelPool(const IExecutionProvider& execution_provier) : execution_provier_(execution_provier) {}

  //todo: return err msg with details;
  Status CreateKernel(const char* op_name,
                      const char* domain,
                      const int& version,
                      const char** type_constraint_names,
                      const int* type_constraint_values,
                      const int& num_type_constraint,
                      const ONNX_NAMESPACE::AttributeProto* attrs,
                      const int& num_attrs,
                      onnxruntime::OpKernel** kernel) {
    *kernel = nullptr;  //todo: validate kernel
    auto kernel_registry = execution_provier_.GetKernelRegistry();
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
                                                 execution_provier_.Type(),
                                                 &kernel_create_info);
    if (kernel_create_info == nullptr) {
      return status;
    }
    onnxruntime::Node node;
    for (int i = 0; i < num_attrs; ++i) {
      node.AddAttribute(attrs[i].name(), attrs[i]);
    }
    OpKernelInfo kernel_info(node, KernelDef{}, execution_provier_, {}, {}, {});
    std::unique_ptr<onnxruntime::OpKernel> op_kernel;
    FuncManager func_mgr;
    status = kernel_create_info->kernel_create_func(func_mgr, kernel_info, op_kernel);
    if (op_kernel) {
      *kernel = op_kernel.get();
      //op_kernels_.push_back(std::move(op_kernel));
      op_kernels_[std::move(op_kernel)] = execution_provier_.GetAllocators()[0];
    }
    return status;
  }

  //todo: disable move and copy
 private:
  //todo: make execution_provier_ const
  //std::list<std::unique_ptr<OpKernel>> op_kernels_;
  std::unordered_map<std::unique_ptr<OpKernel>, AllocatorPtr> op_kernels_;
  const IExecutionProvider& execution_provier_;
};

//todo: make key a const
std::unordered_map<const IExecutionProvider*, std::unique_ptr<EagerKernelPool>> sess_invoker_map;

int CreateKernel(const void* kernel_info,
                      const char* op_name,
                      const char* domain,
                      const int& version,
                      const char** type_constraint_names,
                      const int* type_constraint_values,
                      const int& num_type_constraint,
                      const void* attrs,
                      const int& num_attrs,
                      void** kernel) {
  auto info = reinterpret_cast<const OpKernelInfo*>(kernel_info);
  const IExecutionProvider* ep = reinterpret_cast<const IExecutionProvider*>(info->GetExecutionProvider());
  auto iter = sess_invoker_map.find(ep);  //todo: add rw lock for thread safety
  if (iter == sess_invoker_map.end()) {
    auto ret = sess_invoker_map.emplace(ep, std::make_unique<EagerKernelPool>(*ep));
    return ret.first->second->CreateKernel(op_name,
                                           domain,
                                           version,
                                           type_constraint_names,
                                           type_constraint_values,
                                           num_type_constraint,
                                           reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attrs),
                                           num_attrs,
                                           reinterpret_cast<onnxruntime::OpKernel**>(kernel)).Code();  // todo: deal with ret.second == false
  } else {
    return iter->second->CreateKernel(op_name,
                                      domain,
                                      version,
                                      type_constraint_names,
                                      type_constraint_values,
                                      num_type_constraint,
                                      reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attrs),
                                      num_attrs,
                                      reinterpret_cast<onnxruntime::OpKernel**>(kernel)).Code();
  }
}

int InvokeKernel(const void* context,
                      const void* kernel,
                      const void* const* inputs,
                      const int& input_len,
                      void* const* outputs,
                      const int& output_len) {
  auto ctx = reinterpret_cast<const OpKernelContext*>(context);
  AllocatorPtr allocator{};
  auto ret = ctx->GetTempSpaceAllocator(&allocator); //todo: deal with allocator == nullptr
  if (!ret.IsOK()) {
    return ret.Code();
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
  return ret.Code();
}

}  // namespace invoker
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
  onnxruntime::invoker::CreateKernel(kernel_info, op_name, domain, version, type_constraint_names, type_constraint_values, num_type_constraint, attrs, num_attrs, kernel);
  return nullptr;
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
  onnxruntime::invoker::InvokeKernel(context, kernel, inputs, input_len, outputs, output_len);
  return nullptr;
  API_IMPL_END
}