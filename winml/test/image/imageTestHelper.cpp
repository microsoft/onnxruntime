#include "precomp.h"
#include "imageTestHelper.h"
#include "robuffer.h"
#include <MemoryBuffer.h>
#include <d3dx12.h>
#include "winrt/Windows.Storage.h"
#include "winrt/Windows.Storage.Streams.h"

#define FENCE_SIGNAL_VALUE 1

using namespace WEX::Common;
using namespace WEX::Logging;
using namespace WEX::TestExecution;
using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Graphics::Imaging;

namespace ImageTestHelper
{
    std::wstring GetModelFileName()
    {
        String modelFileName;
        THROW_IF_FAILED(TestData::TryGetValue(L"ModelFileName", modelFileName));
        return std::wstring(modelFileName);
    }

    std::wstring GetImageFileName()
    {
        String imageFileName;
        THROW_IF_FAILED(TestData::TryGetValue(L"ImageFileName", imageFileName));
        return std::wstring(imageFileName);
    }

    std::wstring GetInputPixelFormat()
    {
        String inputPixelFormat;          // Rgba8, Bgra8, Gray8
        THROW_IF_FAILED(TestData::TryGetValue(L"InputPixelFormat", inputPixelFormat));
        return std::wstring(inputPixelFormat);
    }

    std::wstring GetModelPixelFormat()
    {
        String modelPixelFormat;          // Rgba8, Bgra8
        THROW_IF_FAILED(TestData::TryGetValue(L"ModelPixelFormat", modelPixelFormat));
        return std::wstring(modelPixelFormat);
    }

    VideoFrameSource GetVideoFrameSource()
    {
        String videoFramesource;
        THROW_IF_FAILED(TestData::TryGetValue(L"VideoFrameSource", videoFramesource));
        if (L"SoftwareBitmap" == videoFramesource)
        {
            return VideoFrameSource::FromSoftwareBitmap;
        }
        else if (L"Direct3DSurface" == videoFramesource)
        {
            return VideoFrameSource::FromDirect3DSurface;
        }
        else
        {
            Log::Error(String().Format(L"Unsupported VideoFramesource"));
            throw std::invalid_argument("Unsupported VideoFramesource");
        }
    }
    
    VideoFrameSource GetOutputVideoFrameSource()
    {
        String videoFrameSource;
        THROW_IF_FAILED(TestData::TryGetValue(L"OutputVideoFrameSource", videoFrameSource));
        if (L"SoftwareBitmap" == videoFrameSource)
        {
            return VideoFrameSource::FromSoftwareBitmap;
        }
        else if (L"Direct3DSurface" == videoFrameSource)
        {
            return VideoFrameSource::FromDirect3DSurface;
        }
        else if (L"UnsupportedDirect3DSurface" == videoFrameSource)
        {
            return VideoFrameSource::FromUnsupportedD3DSurface;
        }
        else
        {
            Log::Error(String().Format(L"Unsupported VideoFramesource"));
            throw std::invalid_argument("Unsupported VideoFramesource");
        }
    }

    uint32_t GetBatchSize()
    {
        uint32_t batchSize;
        THROW_IF_FAILED(TestData::TryGetValue(L"BatchSize", batchSize));
        return batchSize;
    }

    std::optional<uint32_t> GetBatchSizeOverride()
    {
        bool useSessionOptions;
        THROW_IF_FAILED(TestData::TryGetValue(L"UseSessionOptions", useSessionOptions));

        if (useSessionOptions)
        {
            return std::make_optional<uint32_t>(GetBatchSize());
        }
        return {};
    }

    std::vector<std::wstring> GetInputImages()
    {
        TestDataArray<String> tempInputImages;
        THROW_IF_FAILED(TestData::TryGetValue(L"InputImages", tempInputImages));
        std::vector<std::wstring> inputImages;
        size_t count = tempInputImages.GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            inputImages.emplace_back(std::wstring(tempInputImages[i]));
        }
        return inputImages;
    }

    LearningModelDeviceKind GetDeviceKind()
    {
        String device;
        THROW_IF_FAILED(TestData::TryGetValue(L"Device", device));
        if (L"GPU" == device)
        {
            return LearningModelDeviceKind::DirectX;
        }
        else
        {
            return LearningModelDeviceKind::Cpu;
        }
    }

    OutputBindingStrategy GetOutputBindingStrategy()
    {
        // Bound, Unbound, Empty
        String outputBindingStrategy;
        THROW_IF_FAILED(TestData::TryGetValue(L"OutputBindingStrategy", outputBindingStrategy));
        if (L"bound" == outputBindingStrategy)
        {
            return OutputBindingStrategy::Bound;
        }
        else if (L"unbound" == outputBindingStrategy)
        {
            return OutputBindingStrategy::Unbound;
        }
        else
        {
            Log::Error(String().Format(L"Unsupported OutputBindingStrategy"));
            throw std::invalid_argument("Unsupported OutputBindingStrategy");
        }
    }

    EvaluationStrategy GetEvaluationStrategy()
    {
        String evaluationStrategy;        // Sync or Async
        THROW_IF_FAILED(TestData::TryGetValue(L"EvaluationStrategy", evaluationStrategy));
        if (L"Async" == evaluationStrategy)
        {
            return EvaluationStrategy::Async;
        }
        else if (L"Sync" == evaluationStrategy)
        {
            return EvaluationStrategy::Sync;
        }
        else
        {
            Log::Error(String().Format(L"unsupported evaluation strategy"));
            throw std::invalid_argument("unsupported evaluation strategy");
        }
    }

    ModelInputOutputType GetModelInputOutputType()
    {
        String modelInputOutputType;      // Image or Tensor
        THROW_IF_FAILED(TestData::TryGetValue(L"ModelInputOutputType", modelInputOutputType));
        if (L"image" == modelInputOutputType)
        {
            return ModelInputOutputType::Image;
        }
        else if (L"tensor" == modelInputOutputType)
        {
            return ModelInputOutputType::Tensor;
        }
        else
        {
            Log::Error(String().Format(L"unsupported ModelInputOutputType!!"));
            throw std::invalid_argument("unsupported ModelInputOutputType!!");
        }
    }

    InputImageSource GetInputImageSource()
    {
        String inputImageSource;
        THROW_IF_FAILED(TestData::TryGetValue(L"InputImageSource", inputImageSource));
        if (L"FromImageFeatureValue" == inputImageSource)
        {
            return InputImageSource::FromImageFeatureValue;
        }
        else if (L"FromVideoFrame" == inputImageSource)
        {
            return InputImageSource::FromVideoFrame;
        }
        else if (L"FromCPUResource" == inputImageSource)
        {
            return InputImageSource::FromCPUResource;
        }
        else if (L"FromGPUResource" == inputImageSource)
        {
            return InputImageSource::FromGPUResource;
        }
        throw std::invalid_argument("Unsupported InputImageSource");
    }

    BitmapPixelFormat GetPixelFormat(const std::wstring& inputPixelFormat)
    {
        // Return corresponding BitmapPixelFormat according to input string
        if (L"Bgra8" == inputPixelFormat || L"Bgr8" == inputPixelFormat)
        {
            return BitmapPixelFormat::Bgra8;
        }
        else if (L"Rgba8" == inputPixelFormat || L"Rgb8" == inputPixelFormat)
        {
            return BitmapPixelFormat::Rgba8;
        }
        else if (L"Gray8" == inputPixelFormat)
        {
            return BitmapPixelFormat::Gray8;
        }
        else
        {
            Log::Error(String().Format(L"Unsupported pixelFormat"));
            throw std::invalid_argument("Unsupported pixelFormat");
        }
    }

    TensorFloat LoadInputImageFromCPU(
        SoftwareBitmap softwareBitmap,
        const std::wstring& modelPixelFormat)
    {
        softwareBitmap = SoftwareBitmap::Convert(softwareBitmap, BitmapPixelFormat::Bgra8);
        BYTE* pData = nullptr;
        UINT32 size = 0;
        winrt::Windows::Graphics::Imaging::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Read));
        winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
        auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
        spByteAccess->GetBuffer(&pData, &size);

        uint32_t height = softwareBitmap.PixelHeight();
        uint32_t width = softwareBitmap.PixelWidth();

        // TODO: Need modification for Gray8
        std::vector<int64_t> shape = { 1, 3, height , width };
        float* pCPUTensor;
        uint32_t uCapacity;
        TensorFloat tf = TensorFloat::Create(shape);
        com_ptr<ITensorNative> itn = tf.as<ITensorNative>();
        itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity);
        if (BitmapPixelFormat::Bgra8 == GetPixelFormat(modelPixelFormat))
        {
            for (UINT32 i = 0; i < size; i += 4)
            {
                UINT32 pixelInd = i / 4;
                pCPUTensor[pixelInd] = (float)pData[i];
                pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
                pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i + 2];
            }
        }
        else if (BitmapPixelFormat::Rgba8 == GetPixelFormat(modelPixelFormat))
        {
            for (UINT32 i = 0; i < size; i += 4)
            {
                UINT32 pixelInd = i / 4;
                pCPUTensor[pixelInd] = (float)pData[i + 2];
                pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
                pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i];
            }
        }
        // else if()
        // TODO: for Gray8
        else
        {
            Log::Error(String().Format(L"not supportted pixelFormat"));
        }
        return tf;
    }

    TensorFloat LoadInputImageFromGPU(
        SoftwareBitmap softwareBitmap,
        const std::wstring& modelPixelFormat)
    {
        softwareBitmap = SoftwareBitmap::Convert(softwareBitmap, BitmapPixelFormat::Bgra8);
        BYTE* pData = nullptr;
        UINT32 size = 0;
        BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Read));
        winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
        com_ptr<::Windows::Foundation::IMemoryBufferByteAccess> spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
        spByteAccess->GetBuffer(&pData, &size);

        std::vector<int64_t> shape = { 1, 3, softwareBitmap.PixelHeight() , softwareBitmap.PixelWidth() };
        float* pCPUTensor;
        uint32_t uCapacity;

        // CPU tensorization
        TensorFloat tf = TensorFloat::Create(shape);
        com_ptr<ITensorNative> itn = tf.as<ITensorNative>();
        itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity);

        uint32_t height = softwareBitmap.PixelHeight();
        uint32_t width = softwareBitmap.PixelWidth();
        if (BitmapPixelFormat::Bgra8 == GetPixelFormat(modelPixelFormat))
        {
            for (UINT32 i = 0; i < size; i += 4)
            {
                UINT32 pixelInd = i / 4;
                pCPUTensor[pixelInd] = (float)pData[i];
                pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
                pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i + 2];
            }
        }
        else if (BitmapPixelFormat::Rgba8 == GetPixelFormat(modelPixelFormat))
        {
            for (UINT32 i = 0; i < size; i += 4)
            {
                UINT32 pixelInd = i / 4;
                pCPUTensor[pixelInd] = (float)pData[i + 2];
                pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
                pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i];
            }
        }
        // else if()
        // TODO: for Gray8
        else
        {
            Log::Error(String().Format(L"not supportted pixelFormat"));
        }

        // create the d3d device.
        com_ptr<ID3D12Device> pD3D12Device = nullptr;
        VERIFY_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&pD3D12Device)));

        // create the command queue.
        com_ptr<ID3D12CommandQueue> dxQueue = nullptr;
        D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        pD3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&dxQueue));
        com_ptr<ILearningModelDeviceFactoryNative> devicefactory = get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
        com_ptr<ITensorStaticsNative> tensorfactory = get_activation_factory<TensorFloat, ITensorStaticsNative>();
        com_ptr<::IUnknown> spUnk;
        devicefactory->CreateFromD3D12CommandQueue(dxQueue.get(), spUnk.put());

        // Create ID3D12GraphicsCommandList and Allocator
        D3D12_COMMAND_LIST_TYPE queuetype = dxQueue->GetDesc().Type;
        com_ptr<ID3D12CommandAllocator> alloctor;
        com_ptr<ID3D12GraphicsCommandList> cmdList;

        pD3D12Device->CreateCommandAllocator(
            queuetype,
            winrt::guid_of<ID3D12CommandAllocator>(),
            alloctor.put_void());

        pD3D12Device->CreateCommandList(
            0,
            queuetype,
            alloctor.get(),
            nullptr,
            winrt::guid_of<ID3D12CommandList>(),
            cmdList.put_void());

        // Create Committed Resource
        // 3 is number of channels we use. R G B without alpha.
        UINT64 bufferbytesize = 3 * sizeof(float) * softwareBitmap.PixelWidth()*softwareBitmap.PixelHeight();
        D3D12_HEAP_PROPERTIES heapProperties = {
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            D3D12_MEMORY_POOL_UNKNOWN,
            0,
            0
        };
        D3D12_RESOURCE_DESC resourceDesc = {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            0,
            bufferbytesize,
            1,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
        { 1, 0 },
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };

        com_ptr<ID3D12Resource> pGPUResource = nullptr;
        com_ptr<ID3D12Resource> imageUploadHeap;
        pD3D12Device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            __uuidof(ID3D12Resource),
            pGPUResource.put_void()
        );

        // Create the GPU upload buffer.
        VERIFY_NO_THROW(pD3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferbytesize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            __uuidof(ID3D12Resource),
            imageUploadHeap.put_void()));

        // Copy from Cpu to GPU
        D3D12_SUBRESOURCE_DATA CPUData = {};
        CPUData.pData = reinterpret_cast<BYTE*>(pCPUTensor);
        CPUData.RowPitch = bufferbytesize;
        CPUData.SlicePitch = bufferbytesize;
        UpdateSubresources(cmdList.get(), pGPUResource.get(), imageUploadHeap.get(), 0, 0, 1, &CPUData);

        // Close the command list and execute it to begin the initial GPU setup.
        VERIFY_NO_THROW(cmdList->Close());
        ID3D12CommandList* ppCommandLists[] = { cmdList.get() };
        dxQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

        //Create Event
        HANDLE directEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        wil::unique_event hDirectEvent(directEvent);

        //Create Fence
        Microsoft::WRL::ComPtr<ID3D12Fence> spDirectFence = nullptr;
        THROW_IF_FAILED(pD3D12Device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(spDirectFence.ReleaseAndGetAddressOf())));
        //Adds fence to queue
        THROW_IF_FAILED(dxQueue->Signal(spDirectFence.Get(), FENCE_SIGNAL_VALUE));
        THROW_IF_FAILED(spDirectFence->SetEventOnCompletion(FENCE_SIGNAL_VALUE, hDirectEvent.get()));

        //Wait for signal
        DWORD retVal = WaitForSingleObject(hDirectEvent.get(), INFINITE);
        if (retVal != WAIT_OBJECT_0)
        {
            THROW_IF_FAILED(E_UNEXPECTED);
        }

        // GPU tensorize
        com_ptr<::IUnknown> spUnkTensor;
        TensorFloat input1imagetensor(nullptr);
        int64_t shapes[4] = { 1,3, softwareBitmap.PixelWidth(), softwareBitmap.PixelHeight() };
        tensorfactory->CreateFromD3D12Resource(pGPUResource.get(), shapes, 4, spUnkTensor.put());
        spUnkTensor.try_as(input1imagetensor);

        return input1imagetensor;
    }

    bool VerifyHelper(
        VideoFrame actual,
        VideoFrame expected)
    {
        // Verify two input ImageFeatureValues are identified.
        auto softwareBitmapActual = actual.SoftwareBitmap();
        auto softwareBitmapExpected = expected.SoftwareBitmap();
        VERIFY_IS_TRUE(softwareBitmapActual.PixelHeight() == softwareBitmapExpected.PixelHeight());
        VERIFY_IS_TRUE(softwareBitmapActual.PixelWidth() == softwareBitmapExpected.PixelWidth());
        VERIFY_IS_TRUE(softwareBitmapActual.BitmapPixelFormat() == softwareBitmapExpected.BitmapPixelFormat());

        uint32_t size = 4 * softwareBitmapActual.PixelHeight() * softwareBitmapActual.PixelWidth();

        winrt::Windows::Storage::Streams::Buffer actualOutputBuffer(size);
        winrt::Windows::Storage::Streams::Buffer expectedOutputBuffer(size);

        softwareBitmapActual.CopyToBuffer(actualOutputBuffer);
        softwareBitmapExpected.CopyToBuffer(expectedOutputBuffer);

        byte* actualBytes;
        actualOutputBuffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(&actualBytes);
        byte* expectedBytes;
        expectedOutputBuffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(&expectedBytes);

        byte* pActualByte = actualBytes;
        byte* pExpectedByte = expectedBytes;

        // hard code, might need to be modified later.
        const float cMaxErrorRate = 0.06f;
        int8_t epsilon = 20;

        // Even given two same ImageFeatureValues, the comparison cannot exactly match.
        // So we use error rate.
        UINT errors = 0;
        for (uint32_t i = 0; i < size; i++, pActualByte++, pExpectedByte++)
        {
            auto diff = (*pActualByte - *pExpectedByte);
            if (diff > epsilon)
            {
                errors++;
            }
        }
        Log::Warning(String().Format(L"total errors is %d / %d, errors rate is %f", errors, size, (float)errors / size));

        return ((float)errors / size < cMaxErrorRate);
    }
}