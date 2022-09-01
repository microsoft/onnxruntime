#pragma once

#include "../MLOperatorAuthorImpl.h"
#include "../../../OperatorAuthorHelper/OperatorHelper.h"

#include "../External/D3DX12/d3dx12.h"

// The shader header is produced using "fxc.exe dft_shader.hlsl -E DFT -T cs_5_0 -Zi /Fh"
#include "GeneratedShaders/stockham.h"

#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

using namespace Microsoft::WRL;

namespace DFTHelpers {
    // Divides and rounds up
    inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
    {
        UINT64 temp = static_cast<UINT64>(dividend) + divisor - 1;
        return static_cast<uint32_t>(temp / divisor);
    }

    // Gets the next number of elements to dispatch to the GPU within a loop handling a large
    // total number of tensor elements and threads.
    void GetNextDispatchSize(
        uint32_t elementCount,
        uint32_t elementsPerThread,
        uint32_t numThreads,
        _Out_ uint32_t& dispatch,
        _Out_ uint32_t& pendingElementCount
    )
    {
        // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
        // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
        // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
        assert(numThreads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

        const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

        const uint32_t requiredThreadCount = CeilDivide(elementCount, elementsPerThread);

        // Compute max dispatchable elements
        const uint32_t availableThreadCount = std::min(requiredThreadCount, maxThreadsPerDispatch);

        // Compute required thread group count
        uint32_t workGroupCount1D = CeilDivide(availableThreadCount, numThreads);

        // Compute min dispatch size
        dispatch = workGroupCount1D;

        // With the dispatch size computed, compute the dispatched element count
        const uint32_t dispatchedElementCount = workGroupCount1D * numThreads * elementsPerThread;

        // Update the pending element count
        pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
    }

}

class GpuDFTOperator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    std::vector<uint32_t> m_inputDims = {};
    std::vector<uint32_t> m_outputDims = {};
    int64_t m_axis;
    bool m_isOnesided;
    bool m_isInverse;

    uint32_t m_dftLength = 0;
    uint32_t m_outputDataSize = 0;
    uint32_t m_inputDataSize = 0;
    uint32_t m_outputIdx = 0;
    uint32_t m_numPasses = 0;

    // Allocate temporary buffers if needed
    struct ResourceDesc
    {
        ComPtr<ID3D12Resource> Resource;
        std::array<uint32_t, 4> Sizes;
        std::array<uint32_t, 4> Strides;
    };
    std::vector<ResourceDesc> m_resourceLoopList = {};

    struct LoopRange
    {
        unsigned Left;
        unsigned Right;
        unsigned End;
        unsigned CalculateIndex(unsigned index)
        {
            if (index > 0 && index < End)
            {
                unsigned range = Right - Left + 1;
                index = Left + (index - 1) % range;
            }
            else if (index == End)
            {
                index = Right + 1;
            }
            return index;
        }
    };
    LoopRange m_loopRange = {};

    struct DFTShaderConstants
    {
        uint32_t StartIndex;
        uint32_t ElementCount;
        uint32_t DFTIteration;
        uint32_t IsInverse;
        uint32_t InputSizes[4];
        uint32_t InputStrides[4];
        uint32_t OutputSizes[4];
        uint32_t OutputStrides[4];
        float Scale;
        uint32_t DFTLength;
    };

public:
    GpuDFTOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));


        ORT_THROW_IF_FAILED(context->GetAttribute("axis", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&m_axis)));

        int64_t isInverseInt;
        ORT_THROW_IF_FAILED(context->GetAttribute("inverse", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isInverseInt)));
        m_isInverse = static_cast<bool>(isInverseInt);

        int64_t isOnesidedInt;
        ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));
        m_isOnesided = static_cast<bool>(isOnesidedInt);

        ComPtr<IMLOperatorTensorShapeDescription> shapeDesc;
        ORT_THROW_IF_FAILED(context->GetTensorShapeDescription(shapeDesc.GetAddressOf()));

        // Get the input and output shape sizes
        uint32_t inputDimsSize;
        ORT_THROW_IF_FAILED(shapeDesc->GetInputTensorDimensionCount(0, &inputDimsSize));
        uint32_t outputDimsSize;
        ORT_THROW_IF_FAILED(shapeDesc->GetOutputTensorDimensionCount(0, &outputDimsSize));
        ORT_THROW_HR_IF(E_FAIL, inputDimsSize != outputDimsSize);

        // Get the input shape
        m_inputDims.resize(inputDimsSize);
        ORT_THROW_IF_FAILED(shapeDesc->GetInputTensorShape(0, static_cast<uint32_t>(m_inputDims.size()), m_inputDims.data()));

        // Get the output shape
        m_outputDims.resize(outputDimsSize);
        ORT_THROW_IF_FAILED(shapeDesc->GetOutputTensorShape(0, static_cast<uint32_t>(m_outputDims.size()), m_outputDims.data()));

        // For the number of total elements in the input and output shapes
        m_outputDataSize = ComputeElementCountFromDimensions(m_outputDims);
        m_inputDataSize = ComputeElementCountFromDimensions(m_inputDims);

        // { before_dft_axis, axis, after_dft_axis, real_or_complex }
        std::array<uint32_t, 4> reshapedInputSize = { 1, 1, 1, m_inputDims.back() };
        std::array<uint32_t, 4> reshapedOutputSize = { 1, 1, 1, m_outputDims.back() };

        size_t reshapedIndex = 0;
        for (int i = 0; i < m_inputDims.size() - 1; i++)
        {
            if (i == m_axis || i == (m_axis + 1))
            {
                reshapedIndex++;
            }
            reshapedInputSize[reshapedIndex] *= m_inputDims[i];
            reshapedOutputSize[reshapedIndex] *= m_outputDims[i];
        }

        auto temporarySize = reshapedInputSize;
        temporarySize.back() = reshapedOutputSize.back();

        // Calculate elements and strides
        std::array<uint32_t, 4> reshapedInputStrides = { 1, 1, 1, 1 };
        std::array<uint32_t, 4> reshapedOutputStrides = { 1, 1, 1, 1 };
        std::array<uint32_t, 4> temporaryStrides = { 1, 1, 1, 1 };
        for (int i = static_cast<int>(m_inputDims.size()) - 2; i >= 0; i--)
        {
            reshapedInputStrides[i] = reshapedInputSize[i + 1] * reshapedInputStrides[i + 1];
            reshapedOutputStrides[i] = reshapedOutputSize[i + 1] * reshapedOutputStrides[i + 1];
            temporaryStrides[i] = temporarySize[i + 1] * temporaryStrides[i + 1];
        }

        // Get DFT Length
        ML_CHECK_VALID_ARGUMENT(m_axis < inputDimsSize)
        m_dftLength = m_inputDims[m_axis];
        if (context->IsInputValid(1))
        {
            // If dft_length is specified, then we should honor the shape.
            // If onesided this will be adjusted later on.
            ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
            ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));
            ComPtr<IMLOperatorTensor> dftLengthTensor;
            ORT_THROW_IF_FAILED(contextPrivate->GetConstantInputTensor(1, &dftLengthTensor));
            MLOperatorTensor tensor(dftLengthTensor.Get());
            m_dftLength = gsl::narrow_cast<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));
        }

        // Calculate passes
        m_numPasses = static_cast<unsigned>(log2(m_dftLength));
        bool hasOnePass = m_numPasses == 1;
        bool hasOddPasses = m_numPasses % 2;
        bool hasEvenPasses = !hasOddPasses;

        // write directly input buffer to output buffer, dont create temps
        bool writeToOutput = hasOnePass;
        // First and final are input/output buffers, but all else ocillate between 2 temp buffers
        bool oscillateBetweenTwoTemporaries = !hasOnePass && m_isOnesided;
        // First is input buffer, all else ocillate between temp and output, causing the final pass to write to the output buffer
        bool oscillateFirstOutputThenTemporary = hasOddPasses && !m_isOnesided;
        // First is input buffer, all else ocillate between output and temp, causing the final pass to write to the output buffer
        bool oscillateFirstTemporaryThenOutput = hasEvenPasses && !m_isOnesided;

        // Create the resource loop list
        // Add the input resource to the loop list
        m_resourceLoopList.push_back({});
        m_resourceLoopList.back().Resource = nullptr;
        m_resourceLoopList.back().Sizes = reshapedInputSize;
        m_resourceLoopList.back().Strides = reshapedInputStrides;

        // If 1 temporary should be placed first, or multiple temporaries, then
        // Add a temp in the list
        if (oscillateFirstTemporaryThenOutput || oscillateBetweenTwoTemporaries)
        {
            m_resourceLoopList.push_back({});
            m_resourceLoopList.back().Resource = CreateTemporaryResource(temporarySize);
            m_resourceLoopList.back().Sizes = temporarySize;
            m_resourceLoopList.back().Strides = temporaryStrides;
        }

        // If 2 temps, add another
        if (oscillateBetweenTwoTemporaries)
        {
            m_resourceLoopList.push_back({});
            m_resourceLoopList.back().Resource = CreateTemporaryResource(temporarySize);
            m_resourceLoopList.back().Sizes = temporarySize;
            m_resourceLoopList.back().Strides = temporaryStrides;
        }

        // Add output resource
        m_resourceLoopList.push_back({});
        m_resourceLoopList.back().Resource = nullptr;
        m_resourceLoopList.back().Sizes = reshapedOutputSize;
        m_resourceLoopList.back().Strides = reshapedOutputStrides;
        m_outputIdx = static_cast<uint32_t>(m_resourceLoopList.size() - 1);

        // Add the temporary after output incase of odd number of passes
        if (oscillateFirstOutputThenTemporary)
        {
            m_resourceLoopList.push_back({});
            m_resourceLoopList.back().Resource = CreateTemporaryResource(temporarySize);
            m_resourceLoopList.back().Sizes = temporarySize;
            m_resourceLoopList.back().Strides = temporaryStrides;
        }

        // Define the loop range
        if (writeToOutput) { m_loopRange = { 0, 1, m_numPasses }; }
        if (oscillateBetweenTwoTemporaries) { m_loopRange = { 1, 2, m_numPasses }; }
        if (oscillateFirstOutputThenTemporary) { m_loopRange = { 1, 2, m_numPasses + 1 }; }
        if (oscillateFirstTemporaryThenOutput) { m_loopRange = { 1, 2, m_numPasses + 1 }; }

        PrepareGpuResources();
    }

    void PrepareGpuResources()
    {
        // Compute root signature.
        const int uavCount = 2;
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (UINT i = 0; i < uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        int constantCount = 22;
        rootParameters[uavCount].InitAsConstants(constantCount, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

        ComPtr<ID3DBlob> rootSignatureBlob;
        ComPtr<ID3DBlob> rootSignatureErrorBlob;
        ORT_THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
            &desc,
            rootSignatureBlob.GetAddressOf(),
            rootSignatureErrorBlob.GetAddressOf()
        ));

        ORT_THROW_IF_FAILED(m_device->CreateRootSignature(
            0,
            rootSignatureBlob->GetBufferPointer(),
            rootSignatureBlob->GetBufferSize(),
            IID_ID3D12RootSignature,
            &m_rootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_rootSignature.Get();
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(g_DFT, sizeof(g_DFT));

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
    }

    // Keep the temporary resources around so they are not destroyed while the operator is running
    std::vector<ComPtr<ID3D12Resource>> resourceCache_ = {};
    ComPtr<ID3D12Resource> CreateTemporaryResource(std::array<uint32_t, 4>& size)
    {
        // Regardless of inverse or onesided, temp resources are always in the middle of the
        // middle of the computation passes, and as such will not be half length due to onesidedness.
        // Consequently the input size can be used. However, a correction to double the size when
        // real valued inputs are supplied must be made.
        ComPtr<ID3D12Resource> output;
        auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto bufferByteSize = sizeof(float) * std::accumulate(size.begin(), size.end(), 1, std::multiplies());
        D3D12_RESOURCE_DESC resourceDesc = {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            0,
            static_cast<uint64_t>(bufferByteSize),
            1,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
            {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };

        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&output)));

        resourceCache_.push_back(output);

        return output;
    }

    // Computes the outputs of the kernel.  This may be called multiple times
    // simultaneously within the same instance of the class.  Implementations
    // of this method must be thread-safe.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        try
        {
            // Get the input tensor
            ComPtr<IMLOperatorTensor> inputTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(0, inputTensor.GetAddressOf()));

            // Get the output tensor
            ComPtr<IMLOperatorTensor> outputTensor;
            context->GetOutputTensor(0, outputTensor.GetAddressOf());

            if (outputTensor->IsCpuData() || inputTensor->IsCpuData())
            {
                return E_UNEXPECTED;
            }

            if (outputTensor->GetTensorDataType() != MLOperatorTensorDataType::Float ||
                inputTensor->GetTensorDataType() != MLOperatorTensorDataType::Float)
            {
                return E_UNEXPECTED;
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            ComPtr<IUnknown> inputUnknown;
            ComPtr<ID3D12Resource> inputResource;
            inputTensor->GetDataInterface(inputUnknown.GetAddressOf());
            inputUnknown.As(&inputResource);

            ComPtr<IUnknown> outputUnknown;
            ComPtr<ID3D12Resource> outputResource;
            outputTensor->GetDataInterface(outputUnknown.GetAddressOf());
            outputUnknown.As(&outputResource);

            auto isPowerOfTwo = [](uint32_t n) { return (n != 0) && ((n & (n - 1)) == 0); };
            if (isPowerOfTwo(m_dftLength))
            {
                StockhamFFT(inputResource.Get(), outputResource.Get(), commandList.Get());
            }
            else {
                BluesteinZChirp(inputResource.Get(), outputResource.Get(), commandList.Get());
            }
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    void StockhamFFT(
        ID3D12Resource* inputResource,
        ID3D12Resource* outputResource,
        ID3D12GraphicsCommandList* commandList)
    {
        // Transition resources from common to UAV state
        D3D12_RESOURCE_BARRIER barriers[2];

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            inputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        commandList->ResourceBarrier(2, barriers);

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pipelineState.Get());

        // Each iteration of the below loop represents 1 level in the Stockham DFT
        // Dispatch in a loop
        DFTShaderConstants constants = {};
        constants.DFTLength = m_dftLength;
        constants.DFTIteration = 0;
        constants.IsInverse = m_isInverse;

        auto resourceLoopList = m_resourceLoopList;
        resourceLoopList[0].Resource = inputResource;
        resourceLoopList[m_outputIdx].Resource = outputResource;

        for (unsigned index = 0; index < m_numPasses; index++)
        {
            auto inIdx = m_loopRange.CalculateIndex(index);
            auto outIdx = m_loopRange.CalculateIndex(index + 1);

            auto in = resourceLoopList[inIdx].Resource.Get();
            std::copy(resourceLoopList[inIdx].Sizes.begin(), resourceLoopList[inIdx].Sizes.end(), constants.InputSizes);
            std::copy(resourceLoopList[inIdx].Strides.begin(), resourceLoopList[inIdx].Strides.end(), constants.InputStrides);

            auto out = resourceLoopList[outIdx].Resource.Get();
            std::copy(resourceLoopList[outIdx].Sizes.begin(), resourceLoopList[outIdx].Sizes.end(), constants.OutputSizes);
            std::copy(resourceLoopList[outIdx].Strides.begin(), resourceLoopList[outIdx].Strides.end(), constants.OutputStrides);

            auto isLastPass = (index == m_numPasses - 1);
            auto isLastInversePass = isLastPass && m_isInverse;
            auto dftLength = 1 << m_numPasses;
            constants.Scale = isLastInversePass ? (1.f / dftLength) : 1.f;

            auto totalElementCount =
                std::accumulate(constants.OutputSizes,
                                constants.OutputSizes + std::size(constants.OutputSizes),
                                1,
                                std::multiplies<uint32_t>());
            constants.ElementCount = totalElementCount / constants.OutputSizes[3];
            constants.DFTIteration = index + 1;
            Dispatch(in, out, constants, commandList);
        }

        // Transition resources to common state
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                inputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                outputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        commandList->ResourceBarrier(2, barriers);
    }

    void Dispatch(
        ID3D12Resource* inputResource,
        ID3D12Resource* outputResource,
        DFTShaderConstants& constants,
        ID3D12GraphicsCommandList* commandList)
    {
        D3D12_RESOURCE_BARRIER uav_barriers[2];
        uav_barriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(inputResource);
        uav_barriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(outputResource);
        commandList->ResourceBarrier(2, uav_barriers);
        // Set resource views
        commandList->SetComputeRootUnorderedAccessView(
            0, // root parameter index
            inputResource->GetGPUVirtualAddress()
        );

        commandList->SetComputeRootUnorderedAccessView(
            1, // root parameter index
            outputResource->GetGPUVirtualAddress()
        );
        auto pendingElementCount = constants.ElementCount;

        // Dispatch up to the maximum number of threads per iteration until
        // all elements are completed
        while (pendingElementCount > 0)
        {
            constants.StartIndex = constants.ElementCount - pendingElementCount;

            uint32_t dispatchSizeX;

            DFTHelpers::GetNextDispatchSize(
                pendingElementCount,
                1,
                64,
                dispatchSizeX,
                pendingElementCount
            );

            // Set root constants
            commandList->SetComputeRoot32BitConstants(
                2, // root parameter index
                22, // Constant count
                &constants,
                0 // offset
            );

            commandList->Dispatch(dispatchSizeX, 1, 1);
        }

        commandList->ResourceBarrier(2, uav_barriers);
    }

    void BluesteinZChirp(
        ID3D12Resource* /*inputResource*/,
        ID3D12Resource* /*outputResource*/,
        ID3D12GraphicsCommandList* /*commandList*/)
    {
        ORT_THROW_HR(E_NOTIMPL);
    }
};

struct DFTShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            int64_t axis;
            ORT_THROW_IF_FAILED(context->GetAttribute("axis", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&axis)));
            int64_t isInverseInt;
            ORT_THROW_IF_FAILED(context->GetAttribute("inverse", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isInverseInt)));
            int64_t isOnesidedInt;
            ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));
            bool isOnesided = static_cast<bool>(isOnesidedInt);
            bool isInverse = static_cast<bool>(isInverseInt);

            if (isInverse && isOnesided)
            {
                throw new std::exception("onesided and inverse attributes cannot be enabled at the same time");
            }

            uint32_t rank;
            ORT_THROW_IF_FAILED(context->GetInputTensorDimensionCount(0, &rank));
            if (rank == 0)
            {
                // If no shape is available for the input, skip shape inference...
                throw;
            }

            auto axisIdx = OperatorHelper::HandleNegativeAxis(static_cast<int32_t>(axis), rank);

            // In general the output shape will match the input shape exactly
            // So initialize the output shape with the input shape
            std::vector<uint32_t> inputDims(rank);
            ORT_THROW_IF_FAILED(context->GetInputTensorShape(0, rank, inputDims.data()));
            auto outputDims = inputDims;
            // The last dimension of the output shape is always 2.
            // It corresponds to the real and imaginary parts of the DFT output.
            outputDims.back() = 2;

            if (context->IsInputValid(1))
            {
                // If dft_length is specified, then we should honor the shape.
                // If onesided this will be adjusted later on.
                ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
                ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));
                ComPtr<IMLOperatorTensor> dftLengthTensor;
                ORT_THROW_IF_FAILED(contextPrivate->GetConstantInputTensor(1, &dftLengthTensor));
                MLOperatorTensor tensor(dftLengthTensor.Get());
                auto dft_length = gsl::narrow_cast<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));
                outputDims[axisIdx] = dft_length;
            }

            // When DFT is onesided, the output shape is half the size of the input shape
            // along the specified axis.
            if (isOnesided)
            {
                auto axisDimension = outputDims.at(axisIdx);
                // We need to update the output shape dimension along the specified axis,
                // but sometimes the dimension will be a free dimension or be otherwise unset.
                // Only perform inference when a input dimension value exists.
                auto originalSignalSize = axisDimension;
                auto halfSignalSize = (originalSignalSize >> 1) + 1;
                outputDims.at(axisIdx) = halfSignalSize;
            }

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, rank, outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class GpuDFTOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<GpuDFTOperator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterDFTKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "";
        kernelDescription.name = "DFT";
        kernelDescription.minimumOperatorSetVersion = 17;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::vector<MLOperatorEdgeDescription> t1AllowedEdges
        {
            //MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            //MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Double },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::vector<MLOperatorEdgeDescription> t2AllowedEdges
        {
          //  MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int32 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int64 },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        MLOperatorAttributeNameValue axisAttributeValue;
        axisAttributeValue.name = "axis";
        axisAttributeValue.type = MLOperatorAttributeType::Int;
        axisAttributeValue.valueCount = 1;
        static const int64_t axis[] = { 1 };
        axisAttributeValue.ints = axis;

        MLOperatorAttributeNameValue inverseAttributeValue;
        inverseAttributeValue.name = "inverse";
        inverseAttributeValue.type = MLOperatorAttributeType::Int;
        inverseAttributeValue.valueCount = 1;
        static const int64_t inverse[] = { 0 };
        inverseAttributeValue.ints = inverse;

        MLOperatorAttributeNameValue onesidedAttributeValue;
        onesidedAttributeValue.name = "onesided";
        onesidedAttributeValue.type = MLOperatorAttributeType::Int;
        onesidedAttributeValue.valueCount = 1;
        static const int64_t onesided[] = { 0 };
        onesidedAttributeValue.ints = onesided;

        std::vector<MLOperatorAttributeNameValue> attributeDefaultValues{
            axisAttributeValue,
            inverseAttributeValue,
            onesidedAttributeValue
        };

        kernelDescription.defaultAttributes = attributeDefaultValues.data();
        kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<DFTShapeInferrer>();
        auto factory = wil::MakeOrThrow<GpuDFTOperatorFactory>();

        std::array<uint32_t, 1> requiredConstantCpuInputs = { 1 };

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // alias
            false, // supportsGraph
            nullptr,
            requiredConstantCpuInputs.data(),
            static_cast<uint32_t>(requiredConstantCpuInputs.size())
        ));

    }
};
