// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/dml/DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/framework/op_kernel.h"
#include "core/framework/customregistry.h"
#include "core/framework/tensorprotoutils.h"
#include <wrl/client.h>
#include <wrl/implements.h>

interface IDMLOperator;

namespace WRL
{
    template <typename... TInterfaces>
    using Base = Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
        TInterfaces...
        >;
}

namespace Windows::AI::MachineLearning::Adapter
{ 

using namespace Microsoft::WRL;
        
// Inline method querying whether tensor shapes are defined, during wrappers
// of shape inference callbacks.
template <class T>
bool InputTensorShapesDefinedOnNode(const onnxruntime::OpNodeProtoHelper<T>& nodeInfo)
{
    uint32_t inputCount = nodeInfo.GetInputCount();

    for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
    {
        if (nodeInfo.GetInputType(inputIndex) && (nodeInfo.GetInputType(inputIndex)->value_case() == onnx::TypeProto::kTensorType))
        {
            if (!nodeInfo.GetInputType(inputIndex)->tensor_type().has_shape())
            {
                return false;
            }

            const auto& shape = nodeInfo.GetInputType(inputIndex)->tensor_type().shape();

            for (int input_dim = 0; input_dim < shape.dim_size(); ++input_dim)
            {
                if (!shape.dim(input_dim).has_dim_value())
                {
                    return false;
                }
            }
        }
    }

    return true;
}

::MLOperatorTensorDataType ToMLTensorDataType(onnx::TensorProto_DataType type);

// Used for default values of attributes
struct AttributeValue
{
public:
    size_t ElementCount() const;

    void GetAttribute(
        MLOperatorAttributeType attributeType,
        uint32_t elementCount,
        size_t elementByteSize,
        void* value) const;

    const std::string* GetStringAttribute(
        _In_z_ const char* attributeName,
        uint32_t elementIndex) const;

    std::string name;
    MLOperatorAttributeType type = MLOperatorAttributeType::Undefined;

    std::vector<int64_t> ints;
    std::vector<std::string> strings;
    std::vector<float> floats;
};

using AttributeMap = std::map<std::string, AttributeValue>;

// Encapsulation of shapes across different edges of an operator.    Non-tensor
// edges and unused edges have an empty array of dimensions.
class EdgeShapes
{
public:
    EdgeShapes() = default;

    EdgeShapes(size_t count) : m_shapes(count) {}

    const std::vector<uint32_t>& GetShape(size_t edgeIndex) const
    {
        return m_shapes[edgeIndex];
    }

    std::vector<uint32_t>& GetMutableShape(size_t edgeIndex)
    {
        return m_shapes[edgeIndex];
    }

    size_t EdgeCount() const { return m_shapes.size(); }

    void Reset(size_t edge_count)
    {
        m_shapes.clear();
        m_shapes.resize(edge_count);
    }

    bool operator!=(const EdgeShapes& other) const noexcept
    {
        return (m_shapes != other.m_shapes);
    }

 private:
    std::vector<std::vector<uint32_t>> m_shapes;
};

// Base class for ABI objects which may be "Closed", at which point calls will predictably
// fail or return a dummy value.  This is used for transient ABI context objects which 
// are passed to methods on kernel or inferencers, and which wrap Lotus objects whose lifetimes
// are not controlled by reference counts of the encapsulating object.
class Closable
{
public:
    virtual void Close()
    {
        m_isClosed = true;
    }

protected:
    void VerifyNotClosed() const
    {
        if (m_isClosed)
        {
            ORT_THROW_HR(E_INVALIDARG);
        }
    }

    bool IsClosed() const
    {
        return m_isClosed;
    }

private:
    bool m_isClosed = false;
};

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
class OpNodeInfoWrapper : public Base1_t, public Base2_t, public Closable
{
 public:
    OpNodeInfoWrapper() = delete;

    OpNodeInfoWrapper(
        const onnxruntime::OpNodeProtoHelper<NodeInfoImpl_t>* impl, 
        const EdgeShapes* inputShapesOverride,
        const AttributeMap* defaultAttributes,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter) :
            m_impl(impl), 
            m_inputShapesOverride(inputShapesOverride),
            m_constantInputGetter(constantInputGetter),
            m_defaultAttributes(defaultAttributes)
    {
        m_requiredConstantCpuInputs.assign(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end());
    }

    HRESULT STDMETHODCALLTYPE GetAttributeElementCount(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t* elementCount) const noexcept override;

    template <MLOperatorAttributeType T>
    HRESULT GetAttributeArrayHelper(
        _In_z_ const char* name,
        uint32_t elementCount,
        uint32_t elementByteSize,
        void* values) const;

    HRESULT STDMETHODCALLTYPE GetAttribute(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t elementCount,
        size_t elementByteSize,
        void* attributeValue) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetStringAttributeElementLength(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t* attributeElementByteLength) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetStringAttributeElement(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t attributeElementByteLength,
        char* attributeElement) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetTensorAttribute(
        _In_z_ const char* name,
        _COM_Outptr_ IMLOperatorTensor** tensor) const noexcept override;

    uint32_t STDMETHODCALLTYPE GetInputCount() const noexcept override;
    uint32_t STDMETHODCALLTYPE GetOutputCount() const noexcept override;

    HRESULT STDMETHODCALLTYPE GetInputEdgeDescription(uint32_t inputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputEdgeDescription(uint32_t outputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept;

    HRESULT STDMETHODCALLTYPE GetInputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept;
    HRESULT STDMETHODCALLTYPE GetInputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept;
    
    bool STDMETHODCALLTYPE IsInputValid(uint32_t inputIndex) const noexcept override;
    bool STDMETHODCALLTYPE IsOutputValid(uint32_t outputIndex) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetConstantInputTensor(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept;

 protected:
    // Lifetime is managed by the caller and guaranteed to outlive this class
    const onnxruntime::OpNodeProtoHelper<NodeInfoImpl_t>* m_impl = nullptr;

 private:
    template <MLOperatorAttributeType T>
    HRESULT GetAttributeHelper(
        const char* name,
        uint32_t elementByteSize,
        void* value) const;

    const std::string* GetStringAttribute(
        const char* name,
        uint32_t elementIndex) const;

    // May be null
    const EdgeShapes* m_inputShapesOverride;
    
    std::vector<uint32_t> m_requiredConstantCpuInputs;
    MLOperatorTensorGetter m_constantInputGetter;

    const AttributeMap* m_defaultAttributes = nullptr;
};

class TensorWrapper : public WRL::Base<IMLOperatorTensor>, public Closable
{
 public:
    TensorWrapper() = default;

    TensorWrapper(onnxruntime::Tensor* impl, bool is_data_handle, IWinmlExecutionProvider* provider, bool isInternalOperator);

    uint32_t STDMETHODCALLTYPE GetDimensionCount() const noexcept override;

    HRESULT STDMETHODCALLTYPE GetShape(
            uint32_t dimensionCount,
            uint32_t* dimensions) const noexcept override;

    MLOperatorTensorDataType STDMETHODCALLTYPE GetTensorDataType() const noexcept override;

    bool STDMETHODCALLTYPE IsCpuData() const noexcept override;

    bool STDMETHODCALLTYPE IsDataInterface() const noexcept override;

    void* STDMETHODCALLTYPE GetData() noexcept override;

    void STDMETHODCALLTYPE GetDataInterface(IUnknown** dataInterface) noexcept override;

    const onnxruntime::Tensor* GetInterface() const { return nullptr; }
    onnxruntime::Tensor* GetInterface() { return nullptr; }

 private:
    // Lifetime is managed by the caller and guaranteed to outlive this class
    onnxruntime::Tensor* m_impl = nullptr;
    
    ComPtr<IWinmlExecutionProvider> m_winmlExecutionProvider;
    bool m_internalOperator = false;

    void* m_tensorData = nullptr;
    ComPtr<IUnknown> m_dataInterface;
    bool m_isDataInterface = false;

    // The returned data may be a converted shadow copy, and the piece of it which
    // is returned may vary according to kernel registration options.  
    ComPtr<IUnknown> m_dataInterfaceOrShadowCopy;
    ComPtr<IUnknown> m_abiDataInterface;

};

class OnnxTensorWrapper : public WRL::Base<IMLOperatorTensor>, public Closable
{
 public:
    OnnxTensorWrapper() = default;

    OnnxTensorWrapper(onnx::TensorProto* impl);

    uint32_t STDMETHODCALLTYPE GetDimensionCount() const noexcept override;

    HRESULT STDMETHODCALLTYPE GetShape(
            uint32_t dimensionCount,
            uint32_t* dimensions) const noexcept override;

    MLOperatorTensorDataType STDMETHODCALLTYPE GetTensorDataType() const noexcept override;

    bool STDMETHODCALLTYPE IsCpuData() const noexcept override;

    bool STDMETHODCALLTYPE IsDataInterface() const noexcept override;

    void* STDMETHODCALLTYPE GetData() noexcept override;

    void STDMETHODCALLTYPE GetDataInterface(IUnknown** dataInterface) noexcept override;

    const onnxruntime::Tensor* GetInterface() const { return nullptr; }
    onnxruntime::Tensor* GetInterface() { return nullptr; }

 private:
    size_t m_tensorByteSize = 0;
    std::unique_ptr<std::byte[]> m_unpackedTensor;
    std::byte* m_dataPtr = nullptr;

    // Lifetime is managed by the caller and guaranteed to outlive this class
    onnx::TensorProto* m_impl = nullptr;
};

class OpKernelInfoWrapper : public OpNodeInfoWrapper<
    onnxruntime::ProtoHelperNodeContext, 
    WRL::Base<
        Microsoft::WRL::ChainInterfaces<IMLOperatorKernelCreationContextPrivate, IMLOperatorKernelCreationContext>,
        IMLOperatorTensorShapeDescription, IMLOperatorAttributes1>,
    onnxruntime::null_type>
{
 public:
    OpKernelInfoWrapper(
            const onnxruntime::OpKernelInfo* kerneInfo,
            IUnknown* abiExecutionObject,
            const EdgeShapes* inputShapeOverrides,
            const EdgeShapes* inferredOutputShapes,
            bool allowInputShapeQuery,
            bool allowOutputShapeQuery,
            bool isInternalOperator,
            const AttributeMap* defaultAttributes,
            gsl::span<const uint32_t> requiredConstantCpuInputs,
            MLOperatorTensorGetter& constantInputGetter 
            );

    // HasTensorShapeDescription returns false if and only if the kernel is registered using
    // MLOperatorKernelOptions::AllowDynamicInputTensorSizes.    If this flag is specified and upstream
    // shapes are known when the kernel is created, HasTensorShapeDescription still returns false.
    bool STDMETHODCALLTYPE HasTensorShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept override;

    void STDMETHODCALLTYPE GetExecutionInterface(IUnknown** executionInterface) const noexcept override;

    // IMLOperatorTensorShapeDescription methods.
    HRESULT STDMETHODCALLTYPE GetOutputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept override;
    bool STDMETHODCALLTYPE HasOutputShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept override;

    bool STDMETHODCALLTYPE IsDmlGraphNode() const noexcept override
    {
        return false;
    }

    HRESULT STDMETHODCALLTYPE SetDmlOperator(
        IDMLOperator* op,
        _In_ const DML_OPERATOR_DESC* desc,
        _In_opt_ const MLOperatorKernelDmlProperties* dmlProperties
        ) const noexcept override
    {
        return E_NOTIMPL;
    }
    
private:
    // For shape info, in addition to the info
    const EdgeShapes* m_inferredOutputShapes = nullptr;
    bool m_allowInputShapeQuery = false;
    bool m_allowOutputShapeQuery = false;

    bool m_internalOperator = false;
    ComPtr<IWinmlExecutionProvider> m_winmlProvider;

    const onnxruntime::OpKernelInfo* m_impl = nullptr;
    
    // The execution object returned through the ABI, which may vary according to kernel
    // registration options.
    ComPtr<IUnknown> m_abiExecutionObject;    
};

// OpKernelInfo used for DML graph fusion.  This uses the ONNX graph structures instead of ORT OpKernelInfo.
class DmlGraphOpKernelInfoWrapper : public OpNodeInfoWrapper<
    onnxruntime::ProtoHelperNodeContext, 
    WRL::Base<
        Microsoft::WRL::ChainInterfaces<IMLOperatorKernelCreationContextPrivate, IMLOperatorKernelCreationContext>,
        IMLOperatorTensorShapeDescription, IMLOperatorAttributes1>,
    onnxruntime::null_type>
{
 public:
    DmlGraphOpKernelInfoWrapper(
            const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> * protoHelper,
            const void* executionHandle,
            bool isInternalOperator,
            const EdgeShapes* inferredOutputShapes,
            const AttributeMap* defaultAttributes,
            DmlGraphNodeCreateInfo* graphNodeCreateInfo,
            gsl::span<const uint32_t> requiredConstantCpuInputs,
            MLOperatorTensorGetter& constantInputGetter
            );

    // HasTensorShapeDescription returns false if and only if the kernel is registered using
    // MLOperatorKernelOptions::AllowDynamicInputTensorSizes.  If this flag is specified and upstream
    // shapes are known when the kernel is created, HasTensorShapeDescription still returns false.
    bool STDMETHODCALLTYPE HasTensorShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept override;

    void STDMETHODCALLTYPE GetExecutionInterface(IUnknown** executionInterface) const noexcept override;

    // IMLOperatorTensorShapeDescription methods.
    HRESULT STDMETHODCALLTYPE GetOutputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept override;
    bool STDMETHODCALLTYPE HasOutputShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept override;
    
    bool STDMETHODCALLTYPE IsDmlGraphNode() const noexcept override;
    
    HRESULT STDMETHODCALLTYPE SetDmlOperator(
        IDMLOperator* op,
        _In_ const DML_OPERATOR_DESC* desc,
        _In_opt_ const MLOperatorKernelDmlProperties* dmlProperties
        ) const noexcept override;

private:
    void SetDmlProperties(_In_ const MLOperatorKernelDmlProperties* dmlProperties) const;

    // For shape info, in addition to the info
    const EdgeShapes* m_inferredOutputShapes = nullptr;
    ComPtr<IWinmlExecutionProvider> m_winmlProvider;
    bool m_internalOperator = false;

    // The execution object returned through the ABI, which may vary according to kernel
    // registration options.
    ComPtr<IUnknown> m_abiExecutionObject;
    DmlGraphNodeCreateInfo* m_graphNodeCreateInfo = nullptr;
};

class OpKernelContextWrapper : public WRL::Base<IMLOperatorKernelContext>, public Closable
{
 public:
    ~OpKernelContextWrapper();

    OpKernelContextWrapper(onnxruntime::OpKernelContext* context, const onnxruntime::IExecutionProvider* provider, bool isInternalOperator, const EdgeShapes* outputShapes);

    HRESULT STDMETHODCALLTYPE GetInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensor(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensor(uint32_t outputIndex, uint32_t dimensions, const uint32_t* dimensionSizes, IMLOperatorTensor** tensor) noexcept override;

    HRESULT STDMETHODCALLTYPE AllocateTemporaryData(size_t size, IUnknown** data) const;
    HRESULT STDMETHODCALLTYPE AllocateTemporaryData(size_t size, IUnknown** data, uint64_t* allocId) const;

    void STDMETHODCALLTYPE GetExecutionInterface(IUnknown** executionInterface) const noexcept override;
    
    void Close() override;

    std::vector<IMLOperatorTensor*> GetInputTensors();
    std::vector<IMLOperatorTensor*> GetOutputTensors(const EdgeShapes& outputShapes);

 protected:
    void ClearTempAllocations();
    void TransitionResourcesForOperatorIfRequired(bool isBeforeOp);

    // Lifetime is managed by the caller and guaranteed to outlive this class
    onnxruntime::OpKernelContext* m_impl = nullptr;
    const EdgeShapes* m_outputShapes = nullptr;

    std::vector<ComPtr<TensorWrapper>> m_inputTensors;
    std::vector<ComPtr<TensorWrapper>> m_outputTensors;

    const onnxruntime::IExecutionProvider* m_provider = nullptr;
    ComPtr<IWinmlExecutionProvider> m_winmlProvider;
    bool m_internalOperator = false;

    // The execution object returned to the kernel may vary according to kernel execution options
    ComPtr<IUnknown> m_providerExecutionObject;
    ComPtr<IUnknown> m_abiExecutionObject;

    // Temporary allocations created by the kernel.  These will be freed to the allocator following
    // Compute being called on the kernel.  This list is used to maintain their lifetime.
    mutable std::vector<ComPtr<IUnknown>> m_temporaryAllocations;
    mutable std::vector<ComPtr<IUnknown>> m_temporaryAbiAllocations;
};    

class AbiOpKernel : public onnxruntime::OpKernel
{
 public:
    AbiOpKernel(
            IMLOperatorKernelFactory* operatorFactory,
            const onnxruntime::OpKernelInfo& kerneInfo,
            bool requiresInputShapesAtCreation,
            bool requiresOutputShapesAtCreation,
            bool isInternalOperator,
            gsl::span<const uint32_t> requiredConstantCpuInputs,
            IMLOperatorShapeInferrer* shapeInferrer,
            const AttributeMap* defaultAttributes);

    onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override;

 protected:
    bool RequiresLazyInitialization() const { return (m_operatorFactory != nullptr) && !m_lazyInitialized; };
    void SetLazyInitialized() const { m_lazyInitialized = true; };

    EdgeShapes GetInputShapes(onnxruntime::OpKernelContext* context) const;

    bool InputTensorShapesDefined() const;
    bool InputSizesInferencedFromSchema() const;
    void InferAndVerifyOutputSizes(gsl::span<const uint32_t> requiredConstantCpuInputs, MLOperatorTensorGetter& constantInputGetter, const EdgeShapes* inputShapes, EdgeShapes& outputShapes) const;
    bool m_requiresInputShapesAtCreation = false;
    bool m_requiresOutputShapesAtCreation = false;

    mutable Microsoft::WRL::ComPtr<IMLOperatorKernel> m_kernel;

    // This is null unless the kernel requires lazy initialization
    ComPtr<IMLOperatorKernelFactory> m_operatorFactory;
    mutable volatile bool m_lazyInitialized = false;

    ComPtr<IMLOperatorShapeInferrer> m_shapeInferrer;

    // Used to determine whether anything has changed since creation when shapes or
    // inputs treated as constant by the operator are not inferred / constant.
    mutable EdgeShapes m_inputShapesOfKernelInference;

    struct TensorContent
    {
        bool isValid;
        std::vector<uint32_t> shape;
        MLOperatorTensorDataType type;
        std::vector<std::byte> data;
    };

    mutable std::vector<TensorContent> m_constantInputTensorContentsOfKernel;

    mutable std::mutex m_mutex;
    mutable EdgeShapes m_inferredOutputShapes;

    ComPtr<IWinmlExecutionProvider> m_winmlProvider;
    bool m_internalOperator = false;
    std::vector<uint32_t> m_requiredConstantCpuInputs;
    
    // The execution object returned through the ABI may vary according to kernel
    // registration options.  
    ComPtr<IUnknown> m_providerExecutionObject;
    ComPtr<IUnknown> m_abiExecutionObject;
    
    const AttributeMap* m_defaultAttributes = nullptr;
};

class MLSchemaInferenceContext final : public OpNodeInfoWrapper<
    onnx::InferenceContext, 
    WRL::Base<
        Microsoft::WRL::ChainInterfaces<IMLOperatorShapeInferenceContextPrivate, IMLOperatorShapeInferenceContext>, 
        IMLOperatorTypeInferenceContext, IMLOperatorAttributes, IMLOperatorAttributes1>,
    onnxruntime::null_type>
{
 public:
    MLSchemaInferenceContext() = delete;

    MLSchemaInferenceContext(
        onnxruntime::OpNodeProtoHelper<onnx::InferenceContext>* info, 
        onnx::InferenceContext* ctx,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& mLOperatorTensorGetter
    );

    static ComPtr<MLSchemaInferenceContext> Create(onnxruntime::OpNodeProtoHelper<onnx::InferenceContext>* info, 
        onnx::InferenceContext* ctx,
        gsl::span<const uint32_t> requiredConstantCpuInputs);

    onnx::InferenceContext* GetContext() const
    {
        return m_context;
    }

    HRESULT STDMETHODCALLTYPE SetOutputEdgeDescription(uint32_t outputIndex, const MLOperatorEdgeDescription* edgeDesc) const noexcept override;
    HRESULT STDMETHODCALLTYPE SetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, const uint32_t* dimensions) noexcept override;

 private:
    onnx::InferenceContext* m_context = nullptr;
};

class MLKernelInferenceContext final : public OpNodeInfoWrapper<
    onnxruntime::ProtoHelperNodeContext,
    WRL::Base<Microsoft::WRL::ChainInterfaces<IMLOperatorShapeInferenceContextPrivate, IMLOperatorShapeInferenceContext>, IMLOperatorAttributes, IMLOperatorAttributes1>,
    onnxruntime::null_type>
{
 public:
    MLKernelInferenceContext() = delete;
    MLKernelInferenceContext(
            onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
            const EdgeShapes* inputShapesOverride,
            EdgeShapes& inferredOutputShapes,
            const AttributeMap* defaultAttributes,
            gsl::span<const uint32_t> requiredConstantCpuInputs,
            MLOperatorTensorGetter& constantInputGetter) : 
        OpNodeInfoWrapper(info, inputShapesOverride, defaultAttributes, requiredConstantCpuInputs, constantInputGetter), 
        m_inferredOutputShapes(inferredOutputShapes)
    {
    }

    HRESULT STDMETHODCALLTYPE SetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, const uint32_t* dimensions) noexcept override;

 private:
    EdgeShapes& m_inferredOutputShapes;
};

void InferAndVerifyOutputSizes(
    const onnxruntime::Node& node,
    const AttributeMap* defaultAttributes,
    IMLOperatorShapeInferrer* shapeInferrer,
    gsl::span<const uint32_t> requiredConstantCpuInputs,
    MLOperatorTensorGetter& constantInputGetter,
    const EdgeShapes* inputShapes,
    EdgeShapes& outputShapes);

class MLSupportQueryContext final : public OpNodeInfoWrapper<
    onnxruntime::ProtoHelperNodeContext,
    WRL::Base<Microsoft::WRL::ChainInterfaces<IMLOperatorSupportQueryContextPrivate, IMLOperatorAttributes, IMLOperatorAttributes1>>,
    onnxruntime::null_type>
{
 public:
    MLSupportQueryContext() = delete;

    MLSupportQueryContext(
            onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
            const AttributeMap* defaultAttributes,
            MLOperatorTensorGetter& mLOperatorTensorGetter);

    static ComPtr<MLSupportQueryContext> Create(
            onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
            const AttributeMap* defaultAttributes);

    // TODO - ...
};

onnxruntime::MLDataType ToTensorDataType(::MLOperatorTensorDataType type);
std::string ToTypeString(MLOperatorEdgeDescription desc);
onnx::AttributeProto_AttributeType ToProto(MLOperatorAttributeType type);

bool TryGetStaticInputShapes(const onnxruntime::Node& node, EdgeShapes& inputShapes);
bool TryGetStaticOutputShapes(const onnxruntime::Node& node, EdgeShapes& outputShapes);
bool ContainsEmptyDimensions(const EdgeShapes& shapes, gsl::span<const uint32_t> ignoredShapeIndices = gsl::span<const uint32_t>());

std::tuple<std::unique_ptr<std::byte[]>, size_t> UnpackTensor(const onnx::TensorProto& initializer);
}    // namespace Windows::AI::MachineLearning::Adapter
