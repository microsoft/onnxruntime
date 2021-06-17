// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include <variant>
#include <optional>

#include "core/framework/op_kernel.h"

struct AbstractOperatorDesc;
interface IMLOperatorTensor;

namespace onnxruntime
{
    class KernelDef;
    class Node;
}

namespace Windows::AI::MachineLearning::Adapter
{
    interface __declspec(uuid("5b19a18a-5ed5-4df2-a363-21b89380a698"))
    IWinmlExecutionProvider : public IUnknown
    {
    public:
        // Hold a reference to an object until preceding work in the queue is complete.  This
        // only needs to be handled by providers which hide the asynchronous nature of 
        // computation, and involve resoures which cannot be automatically by work in the
        // the provider's underlying queues.
        virtual void QueueReference(IUnknown *object) = 0;

        virtual void GetShadowCopyIfRequired(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** dataCopy) const = 0;

        virtual void GetABIDataInterface(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** abiData) const = 0;
        
        virtual uint64_t TryGetPooledAllocationId(
            IUnknown* data,
            bool isInternalOperator) = 0;

        virtual void GetABIExecutionInterface(
            bool isInternalOperator,
            IUnknown** abiExecutionObject) const = 0;

        // Whether TransitionResourcesForOperator should be called before and after executing
        // an operator registered to this provider with the specified flags
        virtual bool TransitionsRequiredForOperator(bool isInternalOperator) = 0;

        // If TransitionsRequiredForOperator returns true, should be called before and after executing
        // an operator to transition its resources to and from the appropriate state.
        virtual void TransitionResourcesForOperator(
            bool isBeforeOp,
            uint32_t resourceCount,
            IUnknown** resources) = 0;

        // Waits for flushed work, discards unflushed work, and discards associated references to 
        // prevent circular references.  Must be the last call on the object before destruction.
        virtual void Close() = 0;
    };

    using MLOperatorTensorGetter = std::function<Microsoft::WRL::ComPtr<IMLOperatorTensor>(uint32_t index)>;

    struct DmlOperatorParams
    {
        Microsoft::WRL::ComPtr<IDMLOperator> op;
        std::unique_ptr<AbstractOperatorDesc> desc;
    };

    // This is the counterpart to the MLOperatorKernelDmlProperties ABI struct which owns its memory and uses containers.
    struct DmlGraphNodeCreateInfo
    {
        bool initialized = false;

        // Mapping between DML in/out indices and kernel in/out indices
        std::vector<uint32_t> kernelInputIndices;
        std::vector<uint32_t> kernelOutputIndices;

        Microsoft::WRL::ComPtr<IDMLOperator> op;
        std::unique_ptr<AbstractOperatorDesc> desc;

        bool allowHalfPrecisionComputation = false;
    };

    using GraphNodeFactory = std::function<void(
        const onnxruntime::Node& node, 
        MLOperatorTensorGetter& constantInputGetter,
        const void* executionHandle,
        DmlGraphNodeCreateInfo* graphNodeCreateInfo
        )>;

    struct GraphNodeFactoryRegistration
    {
        GraphNodeFactory factory;
        std::optional<uint32_t> requiredInputCount;
    };

    using KernelSupportQuery = std::function<bool(const onnxruntime::Node& node)>;

    struct InternalRegistrationInfo
    {
        std::vector<uint32_t> requiredConstantCpuInputs;
        std::optional<GraphNodeFactoryRegistration> graphNodeFactoryRegistration;
        KernelSupportQuery supportQuery;

        // Many ONNX operators use 64-bit tensors, but most DML operators only support
        // 32-bit indices. This flag indicates to the graph whether it's okay to compute
        // the result using 32-bit tensors (ignoring the upper bits) via doubled strides.
        bool supportedWith64BitTensorsVia32BitStrides = false;

        // When true, the input to the current operator may come from any execution
        // provider. Otherwise it must have come from another DML node to assume it's safe
        // to use 64-bit to 32-bit striding.
        bool supportedWith64BitTensorsVia32BitStridesFromAnyEp = false;

        // Operator supports true 64-bit tensors directly, no strides needed.
        // So fallback to strided 32-bit only occurs when the device lacks 64-bit support.
        bool prefer64BitTensorsDirectly = false;
    };

    using InternalRegistrationInfoMap = std::unordered_map<onnxruntime::KernelDef*, std::shared_ptr<InternalRegistrationInfo>>;
}