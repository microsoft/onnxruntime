#include "precomp.h"
#include "filehelpers.h"
#include "imageTestHelper.h"
#include "Imagetests.h"
#include "robuffer.h"
#include "winrt/Windows.Storage.h"
#include "winrt/Windows.Storage.Streams.h"
#include "windows.ai.machinelearning.native.internal.h"
#include <MemoryBuffer.h>
#include <d3dx12.h>

using namespace WEX::Common;
using namespace WEX::Logging;
using namespace WEX::TestExecution;
using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

bool ImageTests::TestClassSetup()
{
    init_apartment();
    return true;
}

bool ImageTests::TestMethodSetup()
{
    m_model = nullptr;
    m_device = nullptr;
    m_session = nullptr;

    return true;
}

void ImageTests::LoadModel(const std::wstring& modelPath)
{
    std::wstring fullPath = FileHelpers::GetModulePath() + modelPath;
    VERIFY_NO_THROW(m_model = LearningModel::LoadFromFilePath(fullPath));
}

// MNIST model expects 28x28 data with 
void ImageTests::mnistImageTests()
{
    GPUTEST;
    String fileName;
    unsigned int label;
    THROW_IF_FAILED(TestData::TryGetValue(L"FileName", fileName));
    THROW_IF_FAILED(TestData::TryGetValue(L"Label", label));

    LoadModel(L"mnist.onnx");
    auto imageFeatureValue = FileHelpers::LoadImageFeatureValue(std::wstring(fileName));

    LearningModelDevice device(LearningModelDeviceKind::Cpu);
    m_session = LearningModelSession(m_model, device);
    LearningModelBinding binding(m_session);
    binding.Bind(L"Input3", imageFeatureValue);

    auto result = m_session.EvaluateAsync(binding, L"0").get();

    auto vector = result.Outputs().Lookup(L"Plus214_Output_0").as<TensorFloat>().GetAsVectorView();

    unsigned int maxLabel = 0;
    float maxVal = 0;
    for (unsigned int i = 0; i < vector.Size(); ++i)
    {
        float val = vector.GetAt(i);
        if (val > maxVal)
        {
            maxVal = val;
            maxLabel = i;
        }
    }
    Log::Comment(String().Format(L"Expected Label %d", label));
    Log::Comment(String().Format(L"Evaluated Label %d", maxLabel));
    VERIFY_IS_TRUE(maxLabel == label);
}

void ImageTests::PrepareModelSessionBinding(
    const std::wstring& modelFileName,
    LearningModelDeviceKind deviceKind,
    std::optional<uint32_t> optimizedBatchSize)
{
    LoadModel(std::wstring(modelFileName));
    VERIFY_NO_THROW(m_device = LearningModelDevice(deviceKind));
    if (optimizedBatchSize.has_value())
    {
        LearningModelSessionOptions options;
        options.BatchSizeOverride(optimizedBatchSize.value());
        VERIFY_NO_THROW(m_session = LearningModelSession(m_model, m_device, options));
    }
    else
    {
        VERIFY_NO_THROW(m_session = LearningModelSession(m_model, m_device));
    }
    VERIFY_NO_THROW(m_modelBinding = LearningModelBinding(m_session));
}

bool ImageTests::BindInputValue(
    const std::wstring& imageFileName,
    const std::wstring& inputPixelFormat,
    const std::wstring& modelPixelFormat,
    InputImageSource inputImageSource,
    LearningModelDeviceKind deviceKind
    )
{
    std::wstring fullImagePath = FileHelpers::GetModulePath() + imageFileName;
    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    BitmapDecoder bitmapDecoder = BitmapDecoder::CreateAsync(stream).get();
    SoftwareBitmap softwareBitmap = bitmapDecoder.GetSoftwareBitmapAsync().get();

    // Convert the input image to PixelFormat specified
    softwareBitmap = SoftwareBitmap::Convert(
        softwareBitmap,
        ImageTestHelper::GetPixelFormat(inputPixelFormat));

    auto inputFeature = m_model.InputFeatures().First();
    if (InputImageSource::FromImageFeatureValue == inputImageSource)
    {
        VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
        ImageFeatureValue imageInputTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
        VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), imageInputTensor));
    }
    else if (InputImageSource::FromVideoFrame == inputImageSource)
    {
        VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
        VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), frame));
    }
    else if (InputImageSource::FromCPUResource == inputImageSource)
    {
        TensorFloat tensorFloat = ImageTestHelper::LoadInputImageFromCPU(softwareBitmap, modelPixelFormat);
        VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), tensorFloat));
    }
    else if (InputImageSource::FromGPUResource == inputImageSource)
    {
        TensorFloat tensorFloat = ImageTestHelper::LoadInputImageFromGPU(softwareBitmap, modelPixelFormat);
        if (LearningModelDeviceKind::Cpu == deviceKind)
        {
            VERIFY_THROWS_SPECIFIC(m_modelBinding.Bind(inputFeature.Current().Name(), tensorFloat),
                winrt::hresult_error,
                [](const winrt::hresult_error& e) -> bool
            {
                return e.code() == WINML_ERR_INVALID_BINDING;
            });
            return false;
        }
        VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), tensorFloat));
    }
    return true;
}

VideoFrame ImageTests::BindImageOutput( 
    ModelInputOutputType modelInputOutputType,
    OutputBindingStrategy outputBindingStrategy,
    const std::wstring& modelPixelFormat)
{
    std::wstring outputDataBindingName = std::wstring(m_model.OutputFeatures().First().Current().Name());
    VideoFrame frame = nullptr;
    if (OutputBindingStrategy::Bound == outputBindingStrategy)
    {
        if (ModelInputOutputType::Image == modelInputOutputType)
        {
            ImageFeatureDescriptor outputImageDescriptor = nullptr;
            VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(outputImageDescriptor));
            SoftwareBitmap bitmap(
                ImageTestHelper::GetPixelFormat(modelPixelFormat),
                outputImageDescriptor.Height(),
                outputImageDescriptor.Width());
            frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);
            
        }
        else if (ModelInputOutputType::Tensor == modelInputOutputType)
        {
            TensorFeatureDescriptor outputTensorDescriptor = nullptr;
            VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(outputTensorDescriptor));
            auto outputTensorShape = outputTensorDescriptor.Shape();
            SoftwareBitmap bitmap(
                ImageTestHelper::GetPixelFormat(modelPixelFormat),
                outputTensorShape.GetAt(3),
                outputTensorShape.GetAt(2));
            frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);
        }
        auto outputTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
        VERIFY_NO_THROW(m_modelBinding.Bind(outputDataBindingName, outputTensor));
    }

    // Else for Unbound
    return frame;
}

IVector<VideoFrame> ImageTests::BindImageOutput(
    ModelInputOutputType modelInputOutputType,
    OutputBindingStrategy outputBindingStrategy,
    VideoFrameSource outputVideoFrameSource,
    const std::wstring& modelPixelFormat,
    const uint32_t& batchSize)
{
    std::wstring outputDataBindingName = std::wstring(m_model.OutputFeatures().First().Current().Name());
    uint32_t width = 0, height = 0;
    if (ModelInputOutputType::Image == modelInputOutputType)
    {
        ImageFeatureDescriptor outputImageDescriptor = nullptr;
        VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(outputImageDescriptor));
        width = outputImageDescriptor.Width();
        height = outputImageDescriptor.Height();
    }
    else
    {
        TensorFeatureDescriptor outputTensorDescriptor = nullptr;
        VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(outputTensorDescriptor));
        auto outputTensorShape = outputTensorDescriptor.Shape();
        width = outputTensorShape.GetAt(3);
        height = outputTensorShape.GetAt(2);
    }
    IVector<VideoFrame> outputVideoFrames;
    if (OutputBindingStrategy::Bound == outputBindingStrategy)
    {
        std::vector<VideoFrame> outputFrames = {};
        for (uint32_t i = 0; i < batchSize; ++i)
        {
            VideoFrame videoFrame = nullptr;
            if (VideoFrameSource::FromSoftwareBitmap == outputVideoFrameSource)
            {
                videoFrame = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(ImageTestHelper::GetPixelFormat(modelPixelFormat), width, height));
            }
            else if (VideoFrameSource::FromDirect3DSurface == outputVideoFrameSource)
            {
                videoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8A8UIntNormalized, width, height);
            }
            else if (VideoFrameSource::FromUnsupportedD3DSurface == outputVideoFrameSource)
            {
                videoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, width, height);
            }
            outputFrames.emplace_back(videoFrame);
        }
        outputVideoFrames = winrt::single_threaded_vector(std::move(outputFrames));
        VERIFY_NO_THROW(m_modelBinding.Bind(outputDataBindingName, outputVideoFrames));

    }

    // Else for Unbound
    return outputVideoFrames;
}

void ImageTests::EvaluateTest()
{
    EvaluationStrategy evaluationStrategy = ImageTestHelper::GetEvaluationStrategy();
    if (EvaluationStrategy::Async == evaluationStrategy)
    {
        VERIFY_NO_THROW(m_result = m_session.EvaluateAsync(m_modelBinding, L"").get());
    }
    else if (EvaluationStrategy::Sync == evaluationStrategy)
    {
        VERIFY_NO_THROW(m_result = m_session.Evaluate(m_modelBinding, L""));
    }
}

void ImageTests::VerifyResults(
    VideoFrame outputTensor, 
    const std::wstring& bmImagePath,
    const std::wstring& modelPixelFormat)
{
    SoftwareBitmap bm_softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(bmImagePath);
    bm_softwareBitmap = SoftwareBitmap::Convert(
        bm_softwareBitmap,
        ImageTestHelper::GetPixelFormat(modelPixelFormat));
    VideoFrame bm_videoFrame = VideoFrame::CreateWithSoftwareBitmap(bm_softwareBitmap);

    VERIFY_IS_TRUE(ImageTestHelper::VerifyHelper(outputTensor, bm_videoFrame));
}

bool ShouldSkip(
    const std::wstring& modelFileName,
    const std::wstring& imageFileName,
    const LearningModelDeviceKind deviceKind,
    const InputImageSource inputImageSource
    )
{
    // Case that the tensor's shape doesn't match model's shape should be skiped
    if ((L"1080.jpg" == imageFileName || L"kitten_224.png" == imageFileName) &&
        (InputImageSource::FromGPUResource == inputImageSource || InputImageSource::FromCPUResource == inputImageSource))
    {
        return true;
    }

    // Case that the images's shape doesn't match model's shape which expects free dimension should be skiped.
    // Because the fns-candy is not real model that can handle free dimensional input
    if ((L"1080.jpg" == imageFileName || L"kitten_224.png" == imageFileName) &&
        L"fns-candy_Bgr8_freeDimInput.onnx" == modelFileName)
    {
        return true;
    }

    return false;
}

void ImageTests::ImageTest() 
{
    std::wstring modelFileName = ImageTestHelper::GetModelFileName();
    std::wstring imageFileName = ImageTestHelper::GetImageFileName();
    std::wstring inputPixelFormat = ImageTestHelper::GetInputPixelFormat();
    std::wstring modelPixelFormat = ImageTestHelper::GetModelPixelFormat();

    LearningModelDeviceKind deviceKind = ImageTestHelper::GetDeviceKind();
    InputImageSource inputImageSource = ImageTestHelper::GetInputImageSource();

    if (ShouldSkip(modelFileName, imageFileName, deviceKind, inputImageSource)) 
    {
        Log::Result(TestResults::Skipped, L"This test is disabled");
        return;
    }

    if (LearningModelDeviceKind::Cpu != deviceKind || InputImageSource::FromGPUResource == inputImageSource) 
    {
        GPUTEST;
    }

    PrepareModelSessionBinding(modelFileName, deviceKind, {});

    bool toContinue = BindInputValue(
        imageFileName,
        inputPixelFormat,
        modelPixelFormat,
        inputImageSource,
        deviceKind);

    if (!toContinue) return;
   
    OutputBindingStrategy outputBindingStrategy = ImageTestHelper::GetOutputBindingStrategy();
    ModelInputOutputType modelInputOutputType = ImageTestHelper::GetModelInputOutputType();

    VideoFrame outputVideoFrame = BindImageOutput(
        modelInputOutputType,
        outputBindingStrategy,
        std::wstring(modelPixelFormat));

    EvaluateTest();

    // benchmark used to compare with the output from model
    std::wstring benchmarkFileName = std::wstring(
        modelPixelFormat + L'_' + inputPixelFormat + L'_' + imageFileName);

    // Verify the output by comparing with the benchmark image
    std::wstring bmImagePath = FileHelpers::GetModulePath() + L"groundTruth\\" + benchmarkFileName;
    if (OutputBindingStrategy::Unbound == outputBindingStrategy)
    {
        std::wstring outputDataBindingName = std::wstring(m_model.OutputFeatures().First().Current().Name());
        auto imageFV = m_result.Outputs().Lookup(outputDataBindingName).try_as<ImageFeatureValue>();
        if (imageFV == nullptr)
        {
            return;
        }
        outputVideoFrame = imageFV.VideoFrame();
    }
    VerifyResults(outputVideoFrame, bmImagePath, modelPixelFormat);
}

void ImageTests::BatchSupport() 
{
    std::wstring modelFileName = ImageTestHelper::GetModelFileName();
    std::vector<std::wstring> inputImages = ImageTestHelper::GetInputImages();
    uint32_t batchSize = ImageTestHelper::GetBatchSize();
    std::optional<uint32_t> optimizedBatchSize = ImageTestHelper::GetBatchSizeOverride();
    VideoFrameSource videoFrameSource = ImageTestHelper::GetVideoFrameSource();
    VideoFrameSource outputVideoFrameSource = ImageTestHelper::GetOutputVideoFrameSource();

    LearningModelDeviceKind deviceKind = ImageTestHelper::GetDeviceKind();

    if (VideoFrameSource::FromDirect3DSurface == videoFrameSource && LearningModelDeviceKind::Cpu == deviceKind)
    {
        return;
    }

    // create model, device and session
    PrepareModelSessionBinding(modelFileName, deviceKind, optimizedBatchSize);

    // create the input videoFrames
    std::vector<VideoFrame> inputFrames = {};
    if (inputImages.empty())
    {
        for (uint32_t i = 0; i < batchSize; ++i)
        {
            if (VideoFrameSource::FromDirect3DSurface == videoFrameSource)
            {
                VideoFrame videoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, 720, 720);
                inputFrames.emplace_back(videoFrame);
            }
            else
            {
                VideoFrame videoFrame = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, 720, 720));
                inputFrames.emplace_back(videoFrame);
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < batchSize; ++i)
        {
            std::wstring fullImagePath = FileHelpers::GetModulePath() + inputImages[i];
            StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
            IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
            SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
            VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

            if (VideoFrameSource::FromDirect3DSurface == videoFrameSource) {
                uint32_t width = softwareBitmap.PixelWidth();
                uint32_t height = softwareBitmap.PixelHeight();
                auto D3DVideoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, width, height);
                frame.CopyToAsync(D3DVideoFrame);
                inputFrames.emplace_back(D3DVideoFrame);
            }
            else
            {
                inputFrames.emplace_back(frame);
            }
        }
    }
    auto videoFrames = winrt::single_threaded_vector(std::move(inputFrames));

    auto inputFeatureDescriptor = m_model.InputFeatures().First();
    VERIFY_NO_THROW(m_modelBinding.Bind(inputFeatureDescriptor.Current().Name(), videoFrames));

    OutputBindingStrategy outputBindingStrategy = ImageTestHelper::GetOutputBindingStrategy();
    ModelInputOutputType modelInputOutputType = ImageTestHelper::GetModelInputOutputType();

    auto outputVideoFrames = BindImageOutput(
        modelInputOutputType,
        outputBindingStrategy,
        outputVideoFrameSource,
        L"Bgra8",
        batchSize);

    EvaluateTest();

    // benchmark used to compare with the output from model
    if (OutputBindingStrategy::Unbound == outputBindingStrategy)
    {
        std::wstring outputDataBindingName = std::wstring(m_model.OutputFeatures().First().Current().Name());
        outputVideoFrames = m_result.Outputs().Lookup(outputDataBindingName).try_as<IVector<VideoFrame>>();
        if (outputVideoFrames == nullptr)
        {
            return;
        }
    }
    if (!inputImages.empty())
    {
        for (uint32_t i = 0; i < batchSize; ++i)
        {
            std::wstring bmImagePath = FileHelpers::GetModulePath() + L"batchGroundTruth\\" + inputImages[i];
            if (VideoFrameSource::FromSoftwareBitmap != outputVideoFrameSource && OutputBindingStrategy::Unbound != outputBindingStrategy)
            {
                VideoFrame D3DVideoFrame = outputVideoFrames.GetAt(i);
                VideoFrame SBVideoFrame(BitmapPixelFormat::Bgra8, 720, 720);
                D3DVideoFrame.as<IVideoFrame>().CopyToAsync(SBVideoFrame).get();
                VerifyResults(SBVideoFrame, bmImagePath, L"Bgra8");
            }
            else 
            {
                VerifyResults(outputVideoFrames.GetAt(i), bmImagePath, L"Bgra8");
            }
        }
    }
} 

void ImageTests::LoadBindEvalModelWithoutImageMetadata()
{
    GPUTEST;

    LoadModel(L"squeezenet_tensor_input.onnx");

    auto featureValue = FileHelpers::LoadImageFeatureValue(L"doritos_227.png");

    LearningModelSession modelSession(m_model);
    LearningModelBinding modelBinding(modelSession);

    modelBinding.Bind(L"data", featureValue);
    auto result = modelSession.Evaluate(modelBinding, L"");
}


void ImageTests::LoadBindModelWithoutImageMetadata()
{
    GPUTEST;

    // Model expecting a tensor instead of an image
    LoadModel(L"squeezenet_tensor_input.onnx");

    LearningModelSession modelSession(m_model);
    LearningModelBinding modelBinding(modelSession);

    // Should work on images (by falling back to RGB8)
    auto featureValue = FileHelpers::LoadImageFeatureValue(L"doritos_227.png");
    modelBinding.Bind(L"data", featureValue);

    // Should work on tensors
    auto tensor = TensorFloat::CreateFromIterable({ 1, 3, 227, 227 }, winrt::single_threaded_vector<float>(std::vector<float>(3 * 227 * 227)));
    modelBinding.Bind(L"data", tensor);
}

void ImageTests::LoadInvalidBindModelWithoutImageMetadata()
{
    GPUTEST;

    LoadModel(L"squeezenet_tensor_input.onnx");

    LearningModelSession modelSession(m_model);
    LearningModelBinding modelBinding(modelSession);

    // expect not fail if image dimensions are bigger than required
    auto featureValue = FileHelpers::LoadImageFeatureValue(L"1080.jpg");
    VERIFY_NO_THROW(modelBinding.Bind(L"data", featureValue));

    // expect fail if tensor is of wrong type
    auto tensorUint8 = TensorUInt8Bit::CreateFromIterable({ 1, 3, 227, 227 }, winrt::single_threaded_vector<uint8_t>(std::vector<uint8_t>(3 * 227 * 227)));
    VERIFY_THROWS_SPECIFIC(modelBinding.Bind(L"data", tensorUint8),
        winrt::hresult_error,
        [](const winrt::hresult_error& e) -> bool
    {
        return e.code() == WINML_ERR_INVALID_BINDING;
    });

    // Should fail if tensor has smaller dimensions/type
    auto tensor = TensorFloat::CreateFromIterable({ 1, 3, 22, 22 }, winrt::single_threaded_vector<float>(std::vector<float>(3 * 22 * 22)));
    VERIFY_THROWS_SPECIFIC(modelBinding.Bind(L"data", tensor),
        winrt::hresult_error,
        [](const winrt::hresult_error& e) -> bool
    {
        return e.code() == WINML_ERR_SIZE_MISMATCH;
    });
}

void ImageTests::ImageMetaDataTest()
{
    // supported image metadata
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_0_255.onnx", BitmapAlphaMode::Premultiplied, BitmapPixelFormat::Bgra8, true);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataRgb8_SRGB_0_255.onnx", BitmapAlphaMode::Premultiplied, BitmapPixelFormat::Rgba8, true);

    // unsupported image metadata
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgra8_SRGB_0_255.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Bgra8, false);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataRgba8_SRGB_0_255.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Rgba8, false);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_0_1.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Bgra8, false);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_1_1.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Bgra8, false);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_16_235.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Bgra8, false);
    ValidateOutputImageMetaData(L"Add_ImageNet1920WithImageMetadataBgr8_LINEAR_0_255.onnx", BitmapAlphaMode::Straight, BitmapPixelFormat::Bgra8, false);
}

void ImageTests::ValidateOutputImageMetaData(std::wstring path, BitmapAlphaMode expectedmode, BitmapPixelFormat expectedformat, bool supported)
{
    VERIFY_NO_THROW(LoadModel(path));
    //input does not have image metadata and output does

    VERIFY_IS_TRUE(m_model.OutputFeatures().First().HasCurrent());

    std::wstring name(m_model.OutputFeatures().First().Current().Name());
    std::wstring expectedTensorName = L"add_3";
    VERIFY_ARE_EQUAL(name, expectedTensorName);

    ImageFeatureDescriptor imageDescriptor = nullptr;
    if (supported)
    {
        VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(imageDescriptor));
        VERIFY_IS_TRUE(imageDescriptor != nullptr);

        auto tensorName = imageDescriptor.Name();
        VERIFY_ARE_EQUAL(tensorName, expectedTensorName);

        auto modelDataKind = imageDescriptor.Kind();
        VERIFY_ARE_EQUAL(modelDataKind, LearningModelFeatureKind::Image);

        VERIFY_IS_TRUE(imageDescriptor.IsRequired());

        VERIFY_ARE_EQUAL(imageDescriptor.Width(), 1920);
        VERIFY_ARE_EQUAL(imageDescriptor.Height(), 1080);
        VERIFY_ARE_EQUAL(imageDescriptor.BitmapAlphaMode(), expectedmode);
        VERIFY_ARE_EQUAL(imageDescriptor.BitmapPixelFormat(), expectedformat);
    }
    else
    {
        //not an image descriptor. a regular tensor
        VERIFY_THROWS_SPECIFIC(m_model.OutputFeatures().First().Current().as(imageDescriptor),
            winrt::hresult_no_interface
            , [](const winrt::hresult_no_interface& e) -> bool
        {
            return e.code() == E_NOINTERFACE;
        });
        TensorFeatureDescriptor tensorDescriptor = nullptr;
        VERIFY_NO_THROW(m_model.OutputFeatures().First().Current().as(tensorDescriptor));

        // Make sure we fail binding ImageFeatureValue
        LearningModelSession session(m_model);
        LearningModelBinding binding(session);
        auto ifv = FileHelpers::LoadImageFeatureValue(L"1080.jpg");
        VERIFY_THROWS_SPECIFIC(binding.Bind(L"add_3", ifv),
            winrt::hresult_error,
            [](const winrt::hresult_error& e) -> bool
        {
            return e.code() == WINML_ERR_INVALID_BINDING;
        }
        );
    }

}

static void RunConsecutiveImageBindingOnGpu(ImageFeatureValue & image1, ImageFeatureValue & image2)
{
    static const wchar_t* modelFileName = L"Add_ImageNet1920.onnx";
    std::wstring modulePath = FileHelpers::GetModulePath();

    // WinML model creation
    LearningModel model(nullptr);
    std::wstring fullModelPath = modulePath + modelFileName;
    VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    LearningModelDeviceKind deviceKind = LearningModelDeviceKind::DirectX;
    LearningModelSession modelSession(model, LearningModelDevice(deviceKind));
    LearningModelBinding modelBinding(modelSession);

    //Input Binding
    auto feature = model.InputFeatures().First();
    VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), image1));
    feature.MoveNext();
    VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), image2));
}

static ImageFeatureValue CreateImageFeatureValue(std::wstring fullImagePath)
{
    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    ImageFeatureValue imageInputTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
    return imageInputTensor;
}

//Tests if GPU will throw TDR if the same image feature value is binded back to back for two different inputs to a model
void ImageTests::ImageBindingTwiceSameFeatureValueOnGpu()
{
    GPUTEST;
    std::wstring modulePath = FileHelpers::GetModulePath();
    static const wchar_t* inputDataImageFileName = L"1080.jpg";

    std::wstring fullImagePath = modulePath + inputDataImageFileName;
    ImageFeatureValue input_norm = CreateImageFeatureValue(fullImagePath);

    RunConsecutiveImageBindingOnGpu(input_norm, input_norm);
}

//Tests if GPU will throw TDR if 2 different image feature values are binded back to back for two different inputs to a model
void ImageTests::ImageBindingTwiceDifferentFeatureValueOnGpu()
{
    GPUTEST;
    std::wstring modulePath = FileHelpers::GetModulePath();
    static const wchar_t* inputDataImageFileName = L"1080.jpg";

    std::wstring fullImagePath = modulePath + inputDataImageFileName;
    ImageFeatureValue input_norm = CreateImageFeatureValue(fullImagePath);
    ImageFeatureValue input_norm_1 = CreateImageFeatureValue(fullImagePath);

    RunConsecutiveImageBindingOnGpu(input_norm, input_norm_1);
}

static void RunImageBindingInputAndOutput(bool bindInputAsIInspectable)
{
    static const wchar_t* modelFileName = L"Add_ImageNet1920.onnx";
    std::wstring modulePath = FileHelpers::GetModulePath();
    static const wchar_t* inputDataImageFileName = L"1080.jpg";
    static const wchar_t* outputDataImageFileName = L"out_Add_ImageNet_1080.jpg";

    // WinML model creation
    LearningModel model(nullptr);
    std::wstring fullModelPath = modulePath + modelFileName;
    VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    LearningModelDeviceKind deviceKind = LearningModelDeviceKind::DirectX;
    LearningModelSession modelSession(model, LearningModelDevice(deviceKind));
    LearningModelBinding modelBinding(modelSession);

    std::wstring fullImagePath = modulePath + inputDataImageFileName;

    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

    if (bindInputAsIInspectable)
    {
        auto feature = model.InputFeatures().First();
        VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), frame));
        feature.MoveNext();
        VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), frame));
    }
    else
    {
        ImageFeatureValue inputimagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);
        auto feature = model.InputFeatures().First();
        VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), inputimagetensor));
        feature.MoveNext();
        VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), inputimagetensor));
    }

    auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
    auto outputtensorshape = outputtensordescriptor.Shape();
    VideoFrame outputimage(BitmapPixelFormat::Rgba8, outputtensorshape.GetAt(3), outputtensorshape.GetAt(2));
    ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

    VERIFY_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

    // Evaluate the model
    winrt::hstring correlationId;
    modelSession.EvaluateAsync(modelBinding, correlationId).get();

    //check the output video frame object
    StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
    StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
    IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();

    BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
    // Set the software bitmap
    encoder.SetSoftwareBitmap(outputimage.SoftwareBitmap());

    encoder.FlushAsync().get();

    BYTE* pData = nullptr;
    UINT32 uiCapacity = 0;
    winrt::Windows::Graphics::Imaging::BitmapBuffer spBitmapBuffer(outputimage.SoftwareBitmap().LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Read));
    winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
    auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
    VERIFY_SUCCEEDED(spByteAccess->GetBuffer(&pData, &uiCapacity));
    VERIFY_ARE_NOT_EQUAL(pData[0], 0);
}

void ImageTests::ImageBindingInputAndOutput()
{
    GPUTEST;
    RunImageBindingInputAndOutput(false /*bindInputAsIInspectable*/);
}

void ImageTests::ImageBindingInputAndOutput_BindInputTensorAsInspectable()
{
    GPUTEST;
    RunImageBindingInputAndOutput(true /*bindInputAsIInspectable*/);
}

void ImageTests::TestImageBindingStyleTransfer(const wchar_t* modelFileName, const wchar_t* inputDataImageFileName, wchar_t* outputDataImageFileName)
{
    GPUTEST;

    //this test only checks that the operation completed succefully without crashing

    std::wstring modulePath = FileHelpers::GetModulePath();

    // WinML model creation
    LearningModel model(nullptr);
    std::wstring fullModelPath = modulePath + modelFileName;
    VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    LearningModelDeviceKind deviceKind = LearningModelDeviceKind::DirectX;
    LearningModelDevice device = nullptr;
    VERIFY_NO_THROW(device = LearningModelDevice(deviceKind));
    LearningModelSession modelSession = nullptr;
    VERIFY_NO_THROW(modelSession = LearningModelSession(model, device));
    LearningModelBinding modelBinding = nullptr;
    VERIFY_NO_THROW(modelBinding = LearningModelBinding(modelSession));

    std::wstring fullImagePath = modulePath + inputDataImageFileName;

    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    //aizBUG:3762 Cannot bind the same tensor to 2 different input. will deal with this in a later check in
    ImageFeatureValue input1imagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);

    auto feature = model.InputFeatures().First();
    VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), input1imagetensor));

    auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
    auto outputtensorshape = outputtensordescriptor.Shape();
    VideoFrame outputimage(BitmapPixelFormat::Rgba8, outputtensorshape.GetAt(3), outputtensorshape.GetAt(2));
    ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

    VERIFY_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

    // Evaluate the model
    winrt::hstring correlationId;
    VERIFY_NO_THROW(modelSession.EvaluateAsync(modelBinding, correlationId).get());

    //check the output video frame object
    StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
    StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
    IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();

    BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
    // Set the software bitmap
    encoder.SetSoftwareBitmap(outputimage.SoftwareBitmap());

    encoder.FlushAsync().get();

}

void ImageTests::ImageBindingStyleTransfer()
{
    //this test only checks that the operation completed succefully without crashing
    TestImageBindingStyleTransfer(L"fns-candy.onnx", L"fish_720.png", L"out_fish_720_StyleTransfer.jpg");
}

void ImageTests::ImageBindingAsGPUTensor()
{
    GPUTEST;

    static const wchar_t* modelFileName = L"fns-candy.onnx";
    std::wstring modulePath = FileHelpers::GetModulePath();
    static const wchar_t* inputDataImageFileName = L"fish_720.png";
    static const wchar_t* outputDataImageFileName = L"out_fish_720_StyleTransfer.jpg";

    // WinML model creation
    LearningModel model(nullptr);
    std::wstring fullModelPath = modulePath + modelFileName;
    VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));

    ID3D12Device* pD3D12Device = nullptr;
    VERIFY_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&pD3D12Device)));
    ID3D12CommandQueue* dxQueue = nullptr;
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    pD3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&dxQueue));
    auto devicefactory = get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
    auto tensorfactory = get_activation_factory<TensorFloat, ITensorStaticsNative>();


    com_ptr<::IUnknown> spUnk;
    devicefactory->CreateFromD3D12CommandQueue(dxQueue, spUnk.put());

    LearningModelDevice dmlDeviceCustom = nullptr;
    VERIFY_NO_THROW(spUnk.as(dmlDeviceCustom));
    LearningModelSession dmlSessionCustom = nullptr;
    VERIFY_NO_THROW(dmlSessionCustom = LearningModelSession(model, dmlDeviceCustom));

    LearningModelBinding modelBinding(dmlSessionCustom);

    std::wstring fullImagePath = modulePath + inputDataImageFileName;

    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();

    UINT64 bufferbytesize = softwareBitmap.PixelWidth()*softwareBitmap.PixelHeight() * 3 * sizeof(float);
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
    pD3D12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        __uuidof(ID3D12Resource),
        pGPUResource.put_void()
    );
    com_ptr<::IUnknown> spUnkTensor;
    TensorFloat input1imagetensor(nullptr);
    __int64 shape[4] = { 1,3, softwareBitmap.PixelWidth(), softwareBitmap.PixelHeight() };
    tensorfactory->CreateFromD3D12Resource(pGPUResource.get(), shape, 4, spUnkTensor.put());
    spUnkTensor.try_as(input1imagetensor);

    auto feature = model.InputFeatures().First();
    VERIFY_NO_THROW(modelBinding.Bind(feature.Current().Name(), input1imagetensor));

    auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
    auto outputtensorshape = outputtensordescriptor.Shape();
    VideoFrame outputimage(BitmapPixelFormat::Rgba8, outputtensorshape.GetAt(3), outputtensorshape.GetAt(2));
    ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

    VERIFY_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

    // Evaluate the model
    winrt::hstring correlationId;
    dmlSessionCustom.EvaluateAsync(modelBinding, correlationId).get();

    //check the output video frame object
    StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
    StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
    IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();

    BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
    // Set the software bitmap
    encoder.SetSoftwareBitmap(outputimage.SoftwareBitmap());

    encoder.FlushAsync().get();
}

void ImageTests::GetCleanSession(LearningModelDeviceKind deviceKind, std::wstring modelFilePath, LearningModelDevice &device, LearningModelSession &session)
{
    LearningModel model(nullptr);
    std::wstring fullModelPath = modelFilePath;
    VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    VERIFY_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::DirectX));
    VERIFY_NO_THROW(session = LearningModelSession(model, device));
}

void ImageTests::BindInputToSession(BindingLocation bindLocation, std::wstring inputDataLocation, LearningModelSession& session, LearningModelBinding& binding)
{
    StorageFile imagefile = StorageFile::GetFileFromPathAsync(inputDataLocation).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame cpuVideoFrame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    if (bindLocation == BindingLocation::CPU)
    {
        ImageFeatureValue inputImageFeatureValue = ImageFeatureValue::CreateFromVideoFrame(cpuVideoFrame);
        VERIFY_NO_THROW(binding.Bind(session.Model().InputFeatures().First().Current().Name(), inputImageFeatureValue));
    }
    else
    {
        DirectXPixelFormat format = DirectXPixelFormat::B8G8R8X8UIntNormalized;
        VideoFrame gpuVideoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized,
                                                                               softwareBitmap.PixelWidth(),
                                                                               softwareBitmap.PixelHeight(),
                                                                               session.Device().Direct3D11Device());
        cpuVideoFrame.CopyToAsync(gpuVideoFrame).get();
        ImageFeatureValue inputImageFeatureValue = ImageFeatureValue::CreateFromVideoFrame(gpuVideoFrame);
        VERIFY_NO_THROW(binding.Bind(session.Model().InputFeatures().First().Current().Name(), inputImageFeatureValue));
    }
}

void ImageTests::BindOutputToSession(BindingLocation bindLocation, LearningModelSession& session, LearningModelBinding& binding)
{
    auto outputtensordescriptor = session.Model().OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
    auto outputtensorshape = outputtensordescriptor.Shape();
    if (bindLocation == BindingLocation::CPU)
    {
        VideoFrame outputimage(BitmapPixelFormat::Rgba8, outputtensorshape.GetAt(3), outputtensorshape.GetAt(2));
        ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
        VERIFY_NO_THROW(binding.Bind(session.Model().OutputFeatures().First().Current().Name(), outputTensor));
    }
    else
    {
        VideoFrame outputimage = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized,
                                                                               outputtensorshape.GetAt(3),
                                                                               outputtensorshape.GetAt(2));
        ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
        VERIFY_NO_THROW(binding.Bind(session.Model().OutputFeatures().First().Current().Name(), outputTensor));
    }
}

void ImageTests::SynchronizeGPUWorkloads(const wchar_t* modelFileName, const wchar_t* inputDataImageFileName)
{
    //this test only checks that the operations complete succefully without crashing
    GPUTEST;
    std::wstring modulePath = FileHelpers::GetModulePath();
    LearningModelDevice device = nullptr;
    LearningModelSession session = nullptr;
    LearningModelBinding binding = nullptr;

    /*
     * lazy dx11 loading scenarios:
     */ 

    // Scenario 1
    GetCleanSession(LearningModelDeviceKind::DirectX, modulePath + modelFileName, device, session);

    // input: CPU, output: CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is not initialized
    VERIFY_IS_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    // input: CPU, output: GPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::GPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // Scenario 2
    GetCleanSession(LearningModelDeviceKind::DirectX, modulePath + modelFileName, device, session);

    // input: CPU, output: CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is not initialized
    VERIFY_IS_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    // input: GPU, output: CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::GPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is initialized
    VERIFY_IS_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    // Scenario 3
    GetCleanSession(LearningModelDeviceKind::DirectX, modulePath + modelFileName, device, session);

    // input: CPU, output: CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is not initialized
    VERIFY_IS_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    // input: GPU, output: GPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::GPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::GPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is initialized
    VERIFY_IS_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    /* 
     * non lazy dx11 loading scenarios:
     */ 
    
    // Scenario 1
    GetCleanSession(LearningModelDeviceKind::DirectX, modulePath + modelFileName, device, session);

    // input: GPU, output: CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::GPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // ---> verify that 11 stack is initialized
    VERIFY_IS_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

    // input : CPU, output : CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // Scenario 2
    GetCleanSession(LearningModelDeviceKind::DirectX, modulePath + modelFileName, device, session);

    // input: CPU, output: GPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::GPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));

    // input : CPU, output : CPU
    VERIFY_NO_THROW(binding = LearningModelBinding(session));
    BindInputToSession(BindingLocation::CPU, modulePath + inputDataImageFileName, session, binding);
    BindOutputToSession(BindingLocation::CPU, session, binding);
    VERIFY_NO_THROW(session.Evaluate(binding, L""));
}

void ImageTests::SynchronizeGPUWorkloads()
{
    SynchronizeGPUWorkloads(L"fns-candy.onnx", L"fish_720.png");
}