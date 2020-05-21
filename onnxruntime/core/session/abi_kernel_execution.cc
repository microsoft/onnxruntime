// Licensed under the MIT License.

#include <iostream>
#include <core/graph/model.h>
#include <core/framework/mldata_type_utils.h>
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/session/abi_kernel_execution.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/framework/customregistry.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime;

// taken from OptimizerExecutionFrame
Status
ExecutableKernelContextImpl::CreateNodeOutputMLValueImpl(__attribute__((unused)) OrtValue &ort_value, int ort_value_idx,
                                                         __attribute__((unused)) const TensorShape *shape,
                                                         __attribute__((unused)) size_t nnz) {
    std::string name;
    Status status = info_->value_name_idx_map_.GetName(ort_value_idx, name);
    if (!status.IsOK()) {
        return status;
    }
    return Status(ONNXRUNTIME, RUNTIME_EXCEPTION, "All outputs should already be allocated, but output "
                                                  + name + " was not");
}


ORT_API_STATUS_IMPL(OrtApis::CreateKernelSession,
                    _Outptr_ OrtKernelSession **session_) {
    API_IMPL_BEGIN
        ORT_ENFORCE(session_, "OrtKernelSession pointer must not be null");
        std::unique_ptr<Model> model = std::make_unique<Model>("KernelExecutionModel", true,
                                                               logging::LoggingManager::DefaultLogger());

        KernelSessionImpl *session = new KernelSessionImpl(std::move(model));
        *session_ = reinterpret_cast<OrtKernelSession *>(session);

        return ToOrtStatus(Status::OK());
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateExecutableKernelContext,
                    _In_ OrtKernelSession *session_,
                    OrtProviderType providerType,
                    _In_ const void *node_proto_, // pointer to a c++ NodeProto
                    _In_ const void *arg_to_type_map_, // pointer to a ArgToTypeMap (aka std::unordered_map<std::string, onnx::TypeProto>)
                    _Outptr_ OrtExecutableKernelContext **kernel_context_) {
    API_IMPL_BEGIN

        ORT_ENFORCE(kernel_context_, "OrtExecutableKernelContext pointer must be non-null");

        KernelSessionImpl *session = reinterpret_cast<KernelSessionImpl *>(session_);
        const ONNX_NAMESPACE::NodeProto *node_proto = static_cast<const ONNX_NAMESPACE::NodeProto *>(node_proto_);
        const ArgNameToTypeMap *arg_name_to_type_map = static_cast<const ArgNameToTypeMap *>(arg_to_type_map_);

        // add the node
        auto &graph = session->model->MainGraph();
        Node &node = graph.AddNode(*node_proto, *arg_name_to_type_map);
        Status status = graph.Resolve();
        if (!status.IsOK()) {
            return ToOrtStatus(status);
        }

        // set the execution provider
        std::unique_ptr<IExecutionProvider> execution_provider;
        switch (providerType) {
            case ORT_PROVIDER_CPU:
                node.SetExecutionProviderType(kCpuExecutionProvider);
                execution_provider = make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
                break;
            case ORT_PROVIDER_CUDA:
                // TODO
            default:
                return ToOrtStatus(Status(ONNXRUNTIME,
                                          INVALID_ARGUMENT,
                                          "Unsupported provider type (" + std::to_string(providerType) + ")"));
        }

        // create the kernel
        std::shared_ptr<KernelRegistry> registry = execution_provider->GetKernelRegistry();
        std::unique_ptr<OpKernel> op_kernel;
        status = registry->TryCreateKernel(node,
                                           *execution_provider,
                                           std::unordered_map<int, OrtValue>(),
                                           OrtValueNameIdxMap(),
                                           FuncManager(),
                                           DataTransferManager(),
                                           op_kernel);

        if (!status.IsOK()) {
            return ToOrtStatus(status);
        }

        // create the context info
        std::unique_ptr<ExecutableKernelContextImpl::Info> info = std::make_unique<ExecutableKernelContextImpl::Info>(
                std::move(op_kernel),
                logging::LoggingManager::DefaultLogger());


        ExecutableKernelContextImpl *kernel_context = new ExecutableKernelContextImpl(std::move(info));
        *kernel_context_ = reinterpret_cast<OrtExecutableKernelContext *>(kernel_context);
        return ToOrtStatus(Status::OK());
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_SetInput,
                    OrtExecutableKernelContext *context_,
                    int index,
                    OrtValue *value) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
        Status status = context->SetInput(*value, index);
        return ToOrtStatus(status);
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_SetOutput,
                    OrtExecutableKernelContext *context_,
                    int index,
                    OrtValue *value) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
        Status status = context->SetOutput(*value, index);
        return ToOrtStatus(status);
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_Compute, _Inout_ OrtExecutableKernelContext *context_) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
        context->Compute();
        return ToOrtStatus(Status::OK());
    API_IMPL_END
}

