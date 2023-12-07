#pragma once

#include "DmlDFT.h"

// NOTE: When this operator's implementation is moved into DML, the associated FP16 fallback
//       should be removed from IsCustomOpShader(...) in
//       onnxruntime\core\providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp

enum DmlSTFTKernelInputIndex : uint32_t
{
    Signal,
    FrameStep,
    Window,
    FrameLength
};

// Helper to derive dimensions and attributes from either the STFT shape inferrer or the STFT kernel constructor.
struct DmlSTFTParameters
{
    uint32_t batchSize = 0; // size of first dimension of the signal tensor
    uint32_t signalSize = 0; // size of second dimension of the signal tensor
    uint32_t frameStep = 0; // size of step between window positions in the signal tensor
    uint32_t frameSize = 0; // size of window/frame sliced from the signal tensor
    uint32_t frameCount = 0; // number of frames sliced from the signal tensor
    uint32_t frameDftElementCount = 0; // output size of the DFT applied to each frame
    bool isOnesided = true;
    bool hasWindowTensor = false;
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

    DmlSTFTParameters(
        const OperatorHelper::IKernelInformationAdapter& kernelInfo,
        const OperatorHelper::IShapeInformationAdapter& shapeInfo)
    {
        auto& attributes = kernelInfo.GetAttributes();

        // Attribute: onesided
        this->isOnesided = attributes.template GetOptionalAttribute<bool>("onesided", true);

        // input 0: signal (required; tensor)
        {
            // Signal shape is expected to be [batch_size, signal_length, 1] or [batch_size, signal_length] for
            // real-valued input. It must be [batch_size, signal_length, 2] for complex input.
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(DmlSTFTKernelInputIndex::Signal);
            ML_CHECK_VALID_ARGUMENT(rank == 2 || rank == 3, "Signal shape must be 2D or 3D.");

            auto dims = shapeInfo.GetInputTensorShape(DmlSTFTKernelInputIndex::Signal);
            assert(dims.size() == rank);
            this->batchSize = dims[0];
            this->signalSize = dims[1];

            if (rank == 3)
            {
                ML_CHECK_VALID_ARGUMENT(dims[2] == 1, "DML STFT only accepts real-valued input.");
            }

            MLOperatorEdgeDescription edgeDesc = kernelInfo.GetInputEdgeDescription(0);

            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            this->dataType = Dml::GetDmlDataTypeFromMlDataType(edgeDesc.tensorDataType);
        }

        // input 1: frame_step (required; constant; scalar)
        {
            MLOperatorTensor tensor = kernelInfo.GetConstantInputTensor(DmlSTFTKernelInputIndex::FrameStep);
            this->frameStep = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));

            ML_CHECK_VALID_ARGUMENT(this->frameStep > 0, "The frame_step must be greater than 0.");
        }

        // input 2: window (optional; tensor)
        if (kernelInfo.IsInputValid(DmlSTFTKernelInputIndex::Window))
        {
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(DmlSTFTKernelInputIndex::Window);
            ML_CHECK_VALID_ARGUMENT(rank == 1, "Window shape must be 1D.");

            auto shape = shapeInfo.GetInputTensorShape(DmlSTFTKernelInputIndex::Window);
            this->frameSize = shape[0];

            ML_CHECK_VALID_ARGUMENT(this->frameSize <= this->signalSize, "The window size cannot be larger than the signal size.");

            this->hasWindowTensor = true;
        }

        // input 3: frame_length (optional; constant; scalar)
        if (kernelInfo.IsInputValid(DmlSTFTKernelInputIndex::FrameLength))
        {
            MLOperatorTensor tensor = kernelInfo.GetConstantInputTensor(DmlSTFTKernelInputIndex::FrameLength);
            uint32_t frameLength = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));

            ML_CHECK_VALID_ARGUMENT(
                this->frameSize == 0 || this->frameSize == frameLength,
                "The window size and frame_length must be equal, if both are provided."
            );

            ML_CHECK_VALID_ARGUMENT(frameLength <= this->signalSize, "The frame_length cannot be larger than the signal size.");

            this->frameSize = frameLength;
        }

        ML_CHECK_VALID_ARGUMENT(this->frameSize > 0, "Either the window or frame_length must be set.");

        this->frameCount = (this->signalSize - this->frameSize) / this->frameStep + 1;
        this->frameDftElementCount = this->isOnesided ? this->frameSize / 2 + 1 : this->frameSize;
    }
};

namespace DmlSTFTHelpers
{
    ComPtr<ID3D12Resource> GetResourceFromKernelContext(IMLOperatorKernelContext* context, uint32_t index, bool isInput)
    {
        ComPtr<IMLOperatorTensor> tensor;
        if (isInput)
        {
            ORT_THROW_IF_FAILED(context->GetInputTensor(index, &tensor));
        }
        else
        {
            ORT_THROW_IF_FAILED(context->GetOutputTensor(index, &tensor));
        }

        ComPtr<IUnknown> dataInterface;
        tensor->GetDataInterface(&dataInterface);

        ComPtr<ID3D12Resource> resource;
        ORT_THROW_IF_FAILED(dataInterface.As(&resource));

        return resource;
    }

    ComPtr<ID3D12Resource> GetInputResourceFromKernelContext(IMLOperatorKernelContext* context, uint32_t index)
    {
        return GetResourceFromKernelContext(context, index, true);
    }

    ComPtr<ID3D12Resource> GetOutputResourceFromKernelContext(IMLOperatorKernelContext* context, uint32_t index)
    {
        return GetResourceFromKernelContext(context, index, false);
    }
}

// Implements STFT in two steps:
//
// 1. Unfold/extract frames by sliding a window along the signal.
// 2. Execute DFT on each frame.
//
// Logically, the first step is slicing subregions (frames) of the signal at a regular step
// (frame step); if the window tensor is present, then each sliced frame is multiplied element-wise
// by the window values. Consider an example below, where the signal has 8 values [1,2,3,4,5,6,7,8],
// the window has values [2,1,3,5] (i.e. frame size = 4), and the frame step is 2. The following
// graph achieves the desired STFT calculation:
//
// [1,2,3,4,5,6,7,8] +-> (Slice 0:4) -> [1,2,3,4] -> (Mul [2,1,3,5]) -> [2,2,9,20] -> DFT -> ...   +-> (Join)
//                   |-> (Slice 2:6) -> [3,4,5,6] -> (Mul [2,1,3,5]) -> [6,4,15,30] -> DFT -> ...  |
//                   |-> (Slice 4:8) -> [5,6,7,8] -> (Mul [2,1,3,5]) -> [10,6,21,40] -> DFT -> ... |
//
//  The graph approach above is slow when there are many frames. Instead, this kernel uses a single
//  2D strided element-wise multiply to simultaneously slice the signal and scale by window values:
//
//  Input  | Shape                   | Strides
//  -------|-------------------------|---------------
//  Signal | [frameCount, frameSize] | [frameStep, 1]
//  Window | [frameCount, frameSize] | [0, 1]
//
//  In the above example:
//  Signal shape=[3,4], strides=[2,1]
//  Window shape=[3,4], strides=[0,1]
//
//  out[0,0] = signal[0*2 + 0*1] * window[0*0 + 0*1] = 1 * 2 = 2
//  out[0,1] = signal[0*2 + 1*1] * window[0*0 + 1*1] = 2 * 1 = 2
//  out[0,2] = signal[0*2 + 2*1] * window[0*0 + 2*1] = 3 * 3 = 9
//  out[0,3] = signal[0*2 + 3*1] * window[0*0 + 3*1] = 4 * 5 = 20
//  out[1,0] = signal[1*2 + 0*1] * window[1*0 + 0*1] = 3 * 2 = 6
//  out[1,1] = signal[1*2 + 1*1] * window[1*0 + 1*1] = 4 * 1 = 4
//  out[1,2] = signal[1*2 + 2*1] * window[1*0 + 2*1] = 5 * 3 = 15
//  out[1,3] = signal[1*2 + 3*1] * window[1*0 + 3*1] = 6 * 5 = 30
//  out[2,0] = signal[2*2 + 0*1] * window[2*0 + 0*1] = 5 * 2 = 10
//  out[2,1] = signal[2*2 + 1*1] * window[2*0 + 1*1] = 6 * 1 = 6
//  out[2,2] = signal[2*2 + 2*1] * window[2*0 + 2*1] = 7 * 3 = 21
//  out[2,3] = signal[2*2 + 3*1] * window[2*0 + 3*1] = 8 * 5 = 40
//
// If the window tensor is not present, then slicing is done with a 2D strided element-wise identity.
//
// The framed output will have shape [frameCount, frameSize], and this can be fed directly into a DFT
// kernel by treating the 1st dimension (frameCount) as the batch size; each frame should be processed
// independently as a full signal in DFT.
//
class DmlSTFTOperator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_d3dDevice;
    ComPtr<IDMLDevice> m_dmlDevice;
    ComPtr<Dml::IExecutionProvider> m_dmlProvider;

    struct
    {
        ComPtr<IDMLCompiledOperator> op;
        ComPtr<ID3D12DescriptorHeap> descriptorHeap;
        ComPtr<IDMLBindingTable> bindingTable;
        ComPtr<IDMLCommandRecorder> commandRecorder;
        ComPtr<ID3D12Resource> persistentResource;
        ComPtr<IUnknown> persistentResourcePoolingUnk;
        std::optional<DML_BUFFER_BINDING> persistentResourceBinding;
        bool hasWindowTensor = false;
        uint64_t signalBufferSizeInBytes = 0;
        uint64_t windowBufferSizeInBytes = 0;
        uint64_t outputBufferSizeInBytes = 0;
    } m_framingOperator;

    struct
    {
        ComPtr<GpuDFTOperator> op;
        std::array<uint32_t, 3> inputDims;
        std::array<uint32_t, 3> outputDims;
        uint32_t dftLength = 0;
    } m_dftOperator;

public:
    DmlSTFTOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IMLOperatorKernelCreationContextNodeWrapperPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

        ComPtr<IUnknown> provider;
        ORT_THROW_IF_FAILED(contextPrivate->GetExecutionProvider(&provider));
        ORT_THROW_IF_FAILED(provider.As(&m_dmlProvider));
        ORT_THROW_IF_FAILED(m_dmlProvider->GetDmlDevice(&m_dmlDevice));
        ORT_THROW_IF_FAILED(m_dmlProvider->GetD3DDevice(&m_d3dDevice));

        ComPtr<IMLOperatorTensorShapeDescription> shapeDescInfo;
        ORT_THROW_IF_FAILED(context->GetTensorShapeDescription(&shapeDescInfo));

        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};
        DmlSTFTParameters params(kernelInfo, shapeInfo);

        CompileAndInitFramingOperator(params);

        constexpr uint32_t dftAxis = 1;
        constexpr bool dftIsInverse = false;
        m_dftOperator.op = wil::MakeOrThrow<GpuDFTOperator>(
            m_d3dDevice.Get(),
            dftAxis,
            params.isOnesided,
            dftIsInverse,
            Dml::GetMlDataTypeFromDmlDataType(params.dataType)
        );

        m_dftOperator.inputDims = { params.batchSize * params.frameCount, params.frameSize, 1 };
        m_dftOperator.outputDims = { params.batchSize * params.frameCount, params.frameDftElementCount, 2 };
        m_dftOperator.dftLength = params.frameSize;
    }

    void CompileAndInitFramingOperator(const DmlSTFTParameters& params)
    {
        StackAllocator<1024> stackAllocator;

        auto FillTensorDesc = [&params, &stackAllocator](
            DML_TENSOR_DESC* tensorDesc,
            uint64_t* bufferSizeInBytes,
            std::initializer_list<uint32_t> strides = {}
            )
        {
            DmlBufferTensorDesc bufferDesc;
            bufferDesc.dataType = params.dataType;
            bufferDesc.sizes = { params.batchSize, params.frameCount, params.frameSize };
            if (strides.size() > 0) { bufferDesc.strides = strides; }
            bufferDesc.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
                bufferDesc.dataType,
                onnxruntime::narrow<uint32_t>(bufferDesc.sizes.size()),
                bufferDesc.sizes.data(),
                strides.size() > 0 ? bufferDesc.strides->data() : nullptr
            );

            *bufferSizeInBytes = bufferDesc.totalTensorSizeInBytes;
            *tensorDesc = SchemaHelpers::MakeTensorDesc(bufferDesc, &stackAllocator);
        };

        DML_TENSOR_DESC signalDesc = {};
        FillTensorDesc(&signalDesc, &m_framingOperator.signalBufferSizeInBytes, { params.signalSize, params.frameStep, 1 });

        DML_TENSOR_DESC outputDesc = {};
        FillTensorDesc(&outputDesc, &m_framingOperator.outputBufferSizeInBytes);

        ComPtr<IDMLOperator> framingOp;

        if (params.hasWindowTensor)
        {
            DML_TENSOR_DESC windowDesc = {};
            FillTensorDesc(&windowDesc, &m_framingOperator.windowBufferSizeInBytes, { 0, 0, 1 });

            DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulDesc = {};
            mulDesc.ATensor = &signalDesc;
            mulDesc.BTensor = &windowDesc;
            mulDesc.OutputTensor = &outputDesc;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulDesc };
            ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(&framingOp)));
        }
        else
        {
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identityDesc = {};
            identityDesc.InputTensor = &signalDesc;
            identityDesc.OutputTensor = &outputDesc;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identityDesc };
            ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(&framingOp)));
        }

        m_framingOperator.hasWindowTensor = params.hasWindowTensor;

        // Compile
        {
            DML_EXECUTION_FLAGS flags = params.dataType == DML_TENSOR_DATA_TYPE_FLOAT16 ?
                DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION :
                DML_EXECUTION_FLAG_NONE;

            ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(framingOp.Get(), flags, IID_PPV_ARGS(&m_framingOperator.op)));
        }

        // Initialize
        {
            uint64_t persistentResourceSize = m_framingOperator.op->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_dmlProvider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Enabled,
                    m_framingOperator.persistentResource.GetAddressOf(),
                    m_framingOperator.persistentResourcePoolingUnk.GetAddressOf()));

                m_framingOperator.persistentResourceBinding = DML_BUFFER_BINDING{
                    m_framingOperator.persistentResource.Get(),
                    0,
                    persistentResourceSize
                };
            }

            std::vector<DML_BUFFER_BINDING> initializationInputBindings(params.hasWindowTensor ? 2 : 1);
            ORT_THROW_IF_FAILED(m_dmlProvider->InitializeOperator(
                m_framingOperator.op.Get(),
                m_framingOperator.persistentResourceBinding ? &*m_framingOperator.persistentResourceBinding : nullptr,
                gsl::make_span(initializationInputBindings)
            ));
        }

        auto execBindingProps = m_framingOperator.op->GetBindingProperties();

        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        descriptorHeapDesc.NumDescriptors = execBindingProps.RequiredDescriptorCount;
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

        ORT_THROW_IF_FAILED(m_d3dDevice->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(&m_framingOperator.descriptorHeap)));

        DML_BINDING_TABLE_DESC bindingTableDesc = {};
        bindingTableDesc.Dispatchable = m_framingOperator.op.Get();
        bindingTableDesc.CPUDescriptorHandle = m_framingOperator.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        bindingTableDesc.GPUDescriptorHandle = m_framingOperator.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        bindingTableDesc.SizeInDescriptors = execBindingProps.RequiredDescriptorCount;

        ORT_THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&m_framingOperator.bindingTable)));

        ORT_THROW_IF_FAILED(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_framingOperator.commandRecorder)));
    }

    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        try
        {
            ComPtr<IUnknown> executionObject;
            context->GetExecutionInterface(executionObject.GetAddressOf());

            ComPtr<ID3D12GraphicsCommandList> commandList;
            ORT_THROW_IF_FAILED(executionObject.As(&commandList));

            ComPtr<ID3D12Resource> framingOutputResource;
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(onnxruntime::narrow<size_t>(m_framingOperator.outputBufferSizeInBytes), &framingOutputResource));
            DispatchFramingOperator(commandList.Get(), context, framingOutputResource.Get());

            ComPtr<ID3D12Resource> outputResource = DmlSTFTHelpers::GetOutputResourceFromKernelContext(context, 0);

            D3D12_RESOURCE_BARRIER uavBarrier = { CD3DX12_RESOURCE_BARRIER::UAV(nullptr) };
            commandList->ResourceBarrier(1, &uavBarrier);

            return m_dftOperator.op->Compute(
                commandList.Get(),
                context,
                framingOutputResource.Get(),
                m_dftOperator.inputDims,
                outputResource.Get(),
                m_dftOperator.outputDims,
                m_dftOperator.dftLength
            );
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    void DispatchFramingOperator(ID3D12GraphicsCommandList* commandList, IMLOperatorKernelContext* context, ID3D12Resource* outputResource)
    {
        ID3D12DescriptorHeap* descriptorHeaps[] = { m_framingOperator.descriptorHeap.Get() };
        commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

        auto bindingProps = m_framingOperator.op->GetBindingProperties();

        std::array<DML_BUFFER_BINDING, 2> inputBuffers;
        std::array<DML_BINDING_DESC, 2> inputBindings;
        uint32_t inputBindingsCount = 1;

        // NOTE: avoiding std::array for barriers to avoid buggy code analysis thinking
        // barrierCount is outside the valid range.
        D3D12_RESOURCE_BARRIER barriers[3];
        uint32_t barrierCount = 0;

        ComPtr<ID3D12Resource> signalResource = DmlSTFTHelpers::GetInputResourceFromKernelContext(context, DmlSTFTKernelInputIndex::Signal);
        inputBuffers[0] = { signalResource.Get(), 0, m_framingOperator.signalBufferSizeInBytes };
        inputBindings[0] = { DML_BINDING_TYPE_BUFFER, &inputBuffers[0] };
        barriers[barrierCount++] = CD3DX12_RESOURCE_BARRIER::Transition(signalResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        ComPtr<ID3D12Resource> windowResource;
        if (m_framingOperator.hasWindowTensor)
        {
            windowResource = DmlSTFTHelpers::GetInputResourceFromKernelContext(context, DmlSTFTKernelInputIndex::Window);
            inputBuffers[1] = { windowResource.Get(), 0, m_framingOperator.windowBufferSizeInBytes };
            inputBindings[1] = { DML_BINDING_TYPE_BUFFER, &inputBuffers[1] };
            barriers[barrierCount++] = CD3DX12_RESOURCE_BARRIER::Transition(windowResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            inputBindingsCount++;
        }

        m_framingOperator.bindingTable->BindInputs(inputBindingsCount, inputBindings.data());

        DML_BUFFER_BINDING outputBuffer = {};
        outputBuffer.Buffer = outputResource;
        outputBuffer.SizeInBytes = m_framingOperator.outputBufferSizeInBytes;
        DML_BINDING_DESC outputBinding = { DML_BINDING_TYPE_BUFFER, &outputBuffer };
        barriers[barrierCount++] = CD3DX12_RESOURCE_BARRIER::Transition(outputResource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_framingOperator.bindingTable->BindOutputs(1, &outputBinding);

        ComPtr<ID3D12Resource> tempBuffer;
        auto tempBufferSize = bindingProps.TemporaryResourceSize;
        if (tempBufferSize > 0)
        {
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(onnxruntime::narrow<size_t>(tempBufferSize), &tempBuffer));

            DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
            DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
            m_framingOperator.bindingTable->BindTemporaryResource(&bindingDesc);
        }

        auto persistentBufferSize = bindingProps.PersistentResourceSize;
        if (persistentBufferSize > 0)
        {
            DML_BUFFER_BINDING bufferBinding = { m_framingOperator.persistentResource.Get(), 0, persistentBufferSize };
            DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
            m_framingOperator.bindingTable->BindPersistentResource(&bindingDesc);
        }

        // Transition resources COMMON -> UAV
        commandList->ResourceBarrier(barrierCount, barriers);

        m_framingOperator.commandRecorder->RecordDispatch(
            commandList,
            m_framingOperator.op.Get(),
            m_framingOperator.bindingTable.Get()
        );

        // Transition resources UAV -> COMMON
        for (uint32_t barrierIndex = 0; barrierIndex < barrierCount; barrierIndex++)
        {
            std::swap(barriers[barrierIndex].Transition.StateBefore, barriers[barrierIndex].Transition.StateAfter);
        }

        commandList->ResourceBarrier(barrierCount, barriers);
    }
};

struct STFTShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
            ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

            MLShapeInferenceContext inferenceContext(context);
            OperatorHelper::KernelInformationAdapter kernelInfo{inferenceContext};
            OperatorHelper::ShapeInformationAdapter shapeInfo{inferenceContext};
            DmlSTFTParameters params(kernelInfo, shapeInfo);

            std::array<uint32_t, 4> outputDims = { params.batchSize, params.frameCount, params.frameDftElementCount, 2 };

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, onnxruntime::narrow<uint32_t>(outputDims.size()), outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class DmlSTFTOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<DmlSTFTOperator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterSTFTKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "";
        kernelDescription.name = "STFT";
        kernelDescription.minimumOperatorSetVersion = 17;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::array<MLOperatorEdgeDescription, 2> t1AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Float16) },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Float) },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::array<MLOperatorEdgeDescription, 2> t2AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Int32) },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Int64) },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::array<MLOperatorEdgeTypeConstrant, 2> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        MLOperatorAttributeNameValue onesidedAttributeValue;
        onesidedAttributeValue.name = "onesided";
        onesidedAttributeValue.type = MLOperatorAttributeType::Int;
        onesidedAttributeValue.valueCount = 1;
        static const int64_t onesided[] = { 1 };
        onesidedAttributeValue.ints = onesided;

        std::array<MLOperatorAttributeNameValue, 1> attributeDefaultValues{ onesidedAttributeValue };

        kernelDescription.defaultAttributes = attributeDefaultValues.data();
        kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<STFTShapeInferrer>();
        auto factory = wil::MakeOrThrow<DmlSTFTOperatorFactory>();

        std::array<uint32_t, 2> requiredConstantCpuInputs = { /*frame_step*/1, /*frame_length*/3 };

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // supportsGraph
            nullptr,
            requiredConstantCpuInputs.data(),
            static_cast<uint32_t>(requiredConstantCpuInputs.size())
        ));
    }
};
