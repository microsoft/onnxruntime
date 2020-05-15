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
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status ExecutableKernelContextImpl::CreateNodeOutputMLValueImpl(OrtValue &ort_value, int ort_value_idx,
                                                                const TensorShape *shape, size_t nnz) {
    const DataTypeImpl *ml_type = DataTypeImpl::TypeFromProto(*info_.ort_value_idx_nodearg_map_.at(ort_value_idx));
    if (ml_type == nullptr)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Tried to allocate without valid type information, ort_value index=" +
                      std::to_string(ort_value_idx));
    if (ml_type->IsSparseTensorType()) {
        auto element_type = ml_type->AsSparseTensorType()->GetElementType();
        auto container_type = DataTypeImpl::GetType<SparseTensor>();
        auto sparse = onnxruntime::make_unique<SparseTensor>(element_type, *shape, nnz, info_.GetAllocator());
        ort_value.Init(sparse.release(), container_type, container_type->GetDeleteFunc());
        return Status::OK();
    }

    if (ml_type->IsTensorSequenceType()) {
        auto element_type = ml_type->AsSequenceTensorBase()->GetElementType();
        auto p_sequence = onnxruntime::make_unique<TensorSeq>(element_type);
        auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
        ort_value.Init(p_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());
        return Status::OK();
    }

    if (!ml_type->IsTensorType()) {
        assert(ml_type->AsNonTensorTypeBase() != nullptr);
        const NonTensorTypeBase *non_tensor_type = static_cast<const NonTensorTypeBase *>(ml_type);
        auto creator = non_tensor_type->GetCreateFunc();
        ort_value.Init(creator(), non_tensor_type, non_tensor_type->GetDeleteFunc());
        return Status::OK();
    }

    // tensors
    auto element_type = static_cast<const TensorTypeBase *>(ml_type)->GetElementType();
    AllocatorPtr allocator_ptr = info_.GetAllocator();
    std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                                        *shape,
                                                                        allocator_ptr);

    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

    return Status::OK();
}


ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_Compute, _Inout_ OrtExecutableKernelContext *context_) {
    API_IMPL_BEGIN

        ExecutableKernelContextImpl::Info *info = reinterpret_cast<ExecutableKernelContextImpl::Info *>(context_);

        ExecutableKernelContextImpl context(*info);


        std::cout << "Gonna compute! " << std::endl;
        context.Compute();

        return ToOrtStatus(Status::OK());
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateExecutableKernelContext,
                    const OrtExecutableKernel *kernel_,
                    OrtExecutableKernelContext **context) {
    API_IMPL_BEGIN
        const ExecutableKernelImpl *kernel = reinterpret_cast<const ExecutableKernelImpl *>(kernel_);

        std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
                make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

        InitializedTensorSet set;

        ExecutableKernelContextImpl::Info *info = new ExecutableKernelContextImpl::Info(kernel->op_kernel,
                                                                                        logging::LoggingManager::DefaultLogger());

        *context = reinterpret_cast<OrtExecutableKernelContext *>(info);
        return ToOrtStatus(Status::OK());
    API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddInput, OrtExecutableKernelContext *context_, int index,
                    OrtValue *value) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl::Info *context = reinterpret_cast<ExecutableKernelContextImpl::Info *>(context_);
        Status status = context->AddInput(*value, index);
        return ToOrtStatus(status);
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddImplicitInput, OrtExecutableKernelContext *context_, int index,
                    OrtValue *value) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl::Info *context = reinterpret_cast<ExecutableKernelContextImpl::Info *>(context_);
        Status status = context->AddImplicitInput(*value, index);
        return ToOrtStatus(status);
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddOutput, OrtExecutableKernelContext *context_, int index,
                    OrtValue *value) {
    API_IMPL_BEGIN
        ExecutableKernelContextImpl::Info *context = reinterpret_cast<ExecutableKernelContextImpl::Info *>(context_);
        Status status = context->AddOutput(*value, index);
        return ToOrtStatus(status);
    API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateExecutableKernel,
                    _In_ const void *node_proto_,
                    _In_ const void *arg_name_to_type_map_,
                    _Outptr_ OrtExecutableKernel **context) {
    API_IMPL_BEGIN
        const ONNX_NAMESPACE::NodeProto *node_proto = static_cast<const ONNX_NAMESPACE::NodeProto *>(node_proto_);
        const ArgNameToTypeMap *arg_name_to_type_map = static_cast<const ArgNameToTypeMap *>(arg_name_to_type_map_);

        ExecutionProviders execution_providers;

        CPUExecutionProviderInfo epi;
        auto status = execution_providers.Add(kCpuExecutionProvider, make_unique<CPUExecutionProvider>(epi));

        if (!status.IsOK()) {
            return ToOrtStatus(status);
        }

        std::cout << "Registered provider!" << std::endl;

        std::cout << node_proto->name() << std::endl;

        // create a fake, one-node model
        std::unique_ptr<Model> model = std::make_unique<Model>("Model", true, logging::LoggingManager::DefaultLogger());
        auto &graph = model->MainGraph();

        Node &node = graph.AddNode(*node_proto, *arg_name_to_type_map);


        status = graph.Resolve();
        if (!status.IsOK()) {
            return ToOrtStatus(status);
        }

        node.SetExecutionProviderType(kCpuExecutionProvider);

        std::cout << "Added node!" << std::endl;

        const IExecutionProvider *provider = execution_providers.Get(node);
        if (!provider) {
            auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "node has no provider");
            LOGS_DEFAULT(ERROR) << status.ErrorMessage();
            return ToOrtStatus(status);
        }

        std::shared_ptr<KernelRegistry> registry = provider->GetKernelRegistry();
        std::unordered_map<int, OrtValue> constant_initalized_tensors;

        OrtValueNameIdxMap name_idx_map;
        FuncManager funcs_mgr;
        DataTransferManager data_transfer_mgr;

        std::unique_ptr<OpKernel> op_kernel;
        status = registry->TryCreateKernel(node, *provider, constant_initalized_tensors, name_idx_map, funcs_mgr,
                                           data_transfer_mgr, op_kernel);

        if (!status.IsOK()) {
            return ToOrtStatus(status);
        }


        ExecutableKernelImpl *context_impl = new ExecutableKernelImpl(std::move(model), std::move(op_kernel));

        *context = reinterpret_cast<OrtExecutableKernel *>(context_impl);

        return ToOrtStatus(Status::OK());
    API_IMPL_END
}
