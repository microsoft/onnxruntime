// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/invoker.h"
#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
#include <unordered_map>

namespace onnxruntime {
namespace invoker {

class Invoker final {
 public:
  Invoker(const IExecutionProvider& execution_provier) : execution_provier_(execution_provier) {}

  //todo: return err msg with details;
  Status CreateKernel(const char* op_name, const char* domain, const int& version,
                      const char** type_names, const void** type_values, const int& num_types,
                      const char** attr_names, const int* attr_types, const void** attr_values, const int& num_attrs,
                     onnxruntime::OpKernel** kernel) {
    *kernel = nullptr; //todo: validate kernel
    auto kernel_registry = execution_provier_.GetKernelRegistry();
    // std::unique_ptr<OpKernel> kernel;
    const KernelCreateInfo* kernel_create_info{};
    std::unordered_map<std::string, MLDataType> type_constraint_map;
    for (int i = 0; i < num_types; ++i) {
      //todo: make type_values to be of Tensor data type
      type_constraint_map[type_names[i]] = static_cast<MLDataType>(type_values[i]);
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
      ONNX_NAMESPACE::AttributeProto attr;
      attr.set_name(attr_names[i]);
      if (attr_types[i] == static_cast<int>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
        attr.set_f(*static_cast<const float*>(attr_values[i]));
      } else if (attr_types[i] == static_cast<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32)) {
        attr.set_i(*static_cast<const int*>(attr_values[i]));
      }
      node.AddAttribute(type_names[i], attr);
    }
    OpKernelInfo kernel_info(node, KernelDef{}, execution_provier_, {}, {}, {});
    std::unique_ptr<onnxruntime::OpKernel> op_kernel;
    FuncManager func_mgr;
    status = kernel_create_info->kernel_create_func(func_mgr, kernel_info, op_kernel);
    if (op_kernel) {
      *kernel = op_kernel.get();
      op_kernels_.push_back(std::move(op_kernel));
    }
    return status;
  }

  /*
  void* CreateKernel(const char* domain, const char* op_name, const int& version) {
    for (auto iter = execution_providers_.begin(); iter != execution_providers_.end(); ++iter) {
      auto kernel_registry = iter->get()->GetKernelRegistry();
      // kernel_registry.get()->TryCreateKernel();
    }
    return nullptr;
  }*/
  //todo: disable move and copy
 private:
  //todo: make execution_provier_ const
  std::list<std::unique_ptr<OpKernel>> op_kernels_;
  const IExecutionProvider& execution_provier_;
};

//todo: make key a const
std::unordered_map<onnxruntime::InferenceSession*, std::unique_ptr<Invoker>> sess_invoker_map;

/*
void* CreateOp(void* sess, const char* domain, const char* op_name, const int& version) {
  onnxruntime::InferenceSession* session = reinterpret_cast<onnxruntime::InferenceSession*>(sess);
  auto iter = sess_invoker_map.find(session);  //todo: add rw lock for thread safety
  if (iter == sess_invoker_map.end()) {
    auto ret = sess_invoker_map.emplace(session, std::make_unique<Invoker>(session));
    return ret.first->second->CreateKernel(domain, op_name, version);  // todo: deal with ret.second == false
  } else {
    return iter->second->CreateKernel(domain, op_name, version);
  }
}
*/
}  // namespace invoker
}  // namespace onnxruntime