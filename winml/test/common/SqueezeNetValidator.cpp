#include "std.h"
#include "SqueezeNetValidator.h"
#include "protobufHelpers.h"
#include "fileHelpers.h"
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

#include "WinMLProfiler.h"

using namespace WEX::Common;
using namespace WEX::Logging;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Media;

namespace WinML::Engine::Test{

enum WINML_RUNTIME_TEST_PERF
{
    PREP_TEST = 0,
    CREATE_RUNTIME,
    LOAD_MODEL,
    CREATE_EVAL_CONTEXT,
    RUN_TEST,
    BIND_VALUE,
    EVAL_MODEL,
    EVAL_MODEL_FIRST_RUN,
    kCount
};

static std::vector<std::wstring> WINML_RUNTIME_TEST_PERF_NAMES =
{
    L"PREP TEST            ",
    L"  CREATE RUNTIME     ",
    L"  LOAD MODEL         ",
    L"  CREATE EVAL CONTEXT",
    L"RUN TEST             ",
    L"  BIND VALUE         ",
    L"  EVAL MODEL         ",
    L"  EVAL MODEL 1st Run "
};

#define MAX_PROFILING_LOOP 100
Profiler<WINML_RUNTIME_TEST_PERF> g_RuntimeProfiler;


static void BindImage(
    LearningModelBinding binding,
    const wchar_t* name,
    const wchar_t* fullImagePath,
    bool bindAsInspectable = false)
{
    auto imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    auto stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    auto decoder = BitmapDecoder::CreateAsync(stream).get();
    auto softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
    auto frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

    if (bindAsInspectable)
    {
        VERIFY_NO_THROW(binding.Bind(name, frame));
    }
    else
    {
        auto imagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);
        VERIFY_NO_THROW(binding.Bind(name, imagetensor));
    }
}

static void BindTensor(
    LearningModelBinding binding,
    const wchar_t* name,
    ITensor inputTensor,
    bool bindAsInspectable = false)
{
    VERIFY_IS_NOT_NULL(inputTensor);

    if (bindAsInspectable)
    {
        VERIFY_NO_THROW(binding.Bind(name, inputTensor.as<TensorFloat>().GetAsVectorView()));
    }
    else
    {
        VERIFY_NO_THROW(binding.Bind(name, inputTensor));
    }
}

template <typename T>
ITensor BindOutput(
    OutputBindingStrategy strategy,
    LearningModelBinding binding,
    const wchar_t* name,
    const IVectorView<int64_t> shape = nullptr
)
{
    ITensor outputTensor = nullptr;
    switch (strategy)
    {
    case OutputBindingStrategy::Bound:
        outputTensor = T::Create(shape);
        VERIFY_NO_THROW(binding.Bind(name, outputTensor));
        break;
    case OutputBindingStrategy::Empty:
        outputTensor = T::Create();
        VERIFY_NO_THROW(binding.Bind(name, outputTensor));
        break;
    case OutputBindingStrategy::Unbound:
        __fallthrough;
    default:
        break;
    }

    return outputTensor;
}

ImageFeatureValue BindImageOutput(
    OutputBindingStrategy strategy,
    LearningModelBinding binding,
    const wchar_t* name
)
{
    ImageFeatureValue outputTensor = nullptr;
    switch (strategy)
    {
    case OutputBindingStrategy::Bound:
    {
        SoftwareBitmap bitmap(BitmapPixelFormat::Bgra8, 720, 720);
        VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);
        outputTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
        VERIFY_NO_THROW(binding.Bind(name, outputTensor));
        break;
    }
    case OutputBindingStrategy::Unbound:
        __fallthrough;
    }

    return outputTensor;
}


void ModelValidator::FnsCandy16(
    String instance,
    LearningModelDeviceKind deviceKind,
    OutputBindingStrategy outputBindingStrategy,
    bool bindInputsAsIInspectable,
    float dataTolerance)
{
    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::PREP_TEST);
        // file name strings
        static wchar_t* modelFileName = L"winmlperf_coreml_FNS-Candy_prerelease_fp16.onnx";
        static wchar_t* inputDataImageFileName = L"fish_720.png";
        static wchar_t* outputDataFileName = L"output.png";
        static wchar_t* inputBindingName = L"inputImage";
        static const wchar_t* outputDataBindingName = L"outputImage";

        auto modulePath = FileHelpers::GetModulePath();
        auto fullModelPath = modulePath + modelFileName;
        auto outputFileName = modulePath + outputDataFileName;
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::PREP_TEST);

    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::LOAD_MODEL);
        // WinML model creation
        LearningModel model = nullptr;
        VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::LOAD_MODEL);

    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::RUN_TEST);
        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::CREATE_EVAL_CONTEXT);
            LearningModelSession modelSession = nullptr;
            VERIFY_NO_THROW(modelSession = LearningModelSession(model, LearningModelDevice(deviceKind)));
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::CREATE_EVAL_CONTEXT);

        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::BIND_VALUE);
            LearningModelBinding modelBinding(modelSession);
            auto fullImagePath = modulePath + inputDataImageFileName;
            BindImage(modelBinding, inputBindingName, fullImagePath.c_str(), bindInputsAsIInspectable);

            // create the tensor for the actual output
            auto output = model.OutputFeatures().First().Current();
            VERIFY_IS_TRUE(output.Kind() == LearningModelFeatureKind::Tensor);

            auto shape = winrt::single_threaded_vector(std::vector<int64_t> {1, 1});
            auto outputTensor = BindImageOutput(outputBindingStrategy, modelBinding, outputDataBindingName);
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::BIND_VALUE);

        // Evaluate the model
        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::EVAL_MODEL_FIRST_RUN);
            Log::Comment(String(L"Calling EvaluateSync on instance.") + instance);
            LearningModelEvaluationResult result = nullptr;
            VERIFY_NO_THROW(result = modelSession.Evaluate(modelBinding, {}), String(L"Instance: ") + instance);
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::EVAL_MODEL_FIRST_RUN);

        // Get results
        if (outputBindingStrategy == OutputBindingStrategy::Unbound)
        {
            // When output binding strategy is unbound, the output tensor was not set on bind.
            // Therefore, we need to retrieve it from the LearnignModelEvaluationResult
            auto outputTensor = result.Outputs().Lookup(outputDataBindingName).as<TensorFloat16Bit>();
        }
        else
        {
            VERIFY_ARE_EQUAL(result.Outputs().Lookup(outputDataBindingName), outputTensor);

            auto softwareBitmap = outputTensor.VideoFrame().SoftwareBitmap();

            auto folder = StorageFolder::GetFolderFromPathAsync(modulePath.c_str()).get();
            auto imagefile = folder.CreateFileAsync(outputDataFileName, CreationCollisionOption::ReplaceExisting).get();
            auto stream = imagefile.OpenAsync(FileAccessMode::ReadWrite).get();
            auto encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), stream).get();
            encoder.SetSoftwareBitmap(softwareBitmap);
            encoder.FlushAsync();

        }
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::RUN_TEST);
}

void ModelValidator::SqueezeNet(
    String instance,
    LearningModelDeviceKind deviceKind,
    float dataTolerance,
    bool bindAsImage,
    OutputBindingStrategy outputBindingStrategy,
    bool bindInputsAsIInspectable)
{
    g_RuntimeProfiler.Enable(ProfilerType::CPU);
    g_RuntimeProfiler.Enable(ProfilerType::GPU);
    g_RuntimeProfiler.Reset(ProfilerType::CPU);
    g_RuntimeProfiler.Reset(ProfilerType::GPU);

    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::PREP_TEST);
        // file name strings
        static wchar_t* modelFileName = L"model.onnx";
        static wchar_t* inputDataFileName = L"test_data_0_input.pb";
        static wchar_t* outputDataFileName = L"test_data_0_output.pb";
        static wchar_t* inputBindingName = L"data_0";
        static wchar_t* inputDataImageFileName = L"kitten_224.png";
        static const wchar_t* outputDataBindingName = L"softmaxout_1";

        auto modulePath = FileHelpers::GetModulePath();
        auto fullModelPath = modulePath + modelFileName;
        auto outputFileName = modulePath + outputDataFileName;
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::PREP_TEST);

    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::LOAD_MODEL);
        // WinML model creation
        LearningModel model = nullptr;
        VERIFY_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::LOAD_MODEL);

    WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::RUN_TEST);
        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::CREATE_EVAL_CONTEXT);
            LearningModelSession modelSession = nullptr;
            VERIFY_NO_THROW(modelSession = LearningModelSession(model, LearningModelDevice(deviceKind)));
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::CREATE_EVAL_CONTEXT);

        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::BIND_VALUE);
            LearningModelBinding modelBinding(modelSession);

            if (bindAsImage)
            {
                std::wstring fullImagePath = modulePath + inputDataImageFileName;
                BindImage(modelBinding, inputBindingName, fullImagePath.c_str(), bindInputsAsIInspectable);
            }
            else
            {
                auto inputDataPath = modulePath + inputDataFileName;
                auto inputTensor = ProtobufHelpers::LoadTensorFromProtobufFile(inputDataPath, false);
                BindTensor(modelBinding, inputBindingName, inputTensor, bindInputsAsIInspectable);
            }

            // load up the expected output
            auto expectedResultsTensor = ProtobufHelpers::LoadTensorFromProtobufFile(outputFileName, false);
            VERIFY_IS_NOT_NULL(expectedResultsTensor);

            // create the tensor for the actual output
            auto output = model.OutputFeatures().First().Current();
            VERIFY_IS_TRUE(output.Kind() == LearningModelFeatureKind::Tensor);
   
            auto outputTensor = BindOutput<TensorFloat>(
                outputBindingStrategy, modelBinding, outputDataBindingName, expectedResultsTensor.Shape());
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::BIND_VALUE);

        // Evaluate the model
        WINML_PROFILING_START(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::EVAL_MODEL_FIRST_RUN);
            Log::Comment(String(L"Calling EvaluateSync on instance.") + instance);
            LearningModelEvaluationResult result = nullptr;
            VERIFY_NO_THROW(result = modelSession.Evaluate(modelBinding, {}), String(L"Instance: ") + instance);
        WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::EVAL_MODEL_FIRST_RUN);

        // Get results
        if (outputBindingStrategy == OutputBindingStrategy::Unbound)
        {
            // When output binding strategy is unbound, the output tensor was not set on bind.
            // Therefore, we need to retrieve it from the LearnignModelEvaluationResult
            outputTensor = result.Outputs().Lookup(outputDataBindingName).as<ITensor>();
        }
        else
        {
            VERIFY_ARE_EQUAL(result.Outputs().Lookup(outputDataBindingName), outputTensor);
        }

        auto outDataExpected = expectedResultsTensor.as<TensorFloat>().GetAsVectorView();
        auto outDataActual = outputTensor.as<TensorFloat>().GetAsVectorView();

        VERIFY_IS_TRUE(outDataActual.Size() == outDataExpected.Size());
        for (uint32_t i = 0; i < outDataActual.Size(); i++)
        {
            float delta = std::abs(outDataActual.GetAt(i) - outDataExpected.GetAt(i));
            if (delta > dataTolerance)
            {
                Log::Comment(String().Format(L"EXPECTED:%f , ACTUAL: %f", outDataExpected.GetAt(i), outDataActual.GetAt(i)));
                Log::Error(String().Format(L"instance %ws, element %d", static_cast<const wchar_t*>(instance), i));
            }
        }
    WINML_PROFILING_STOP(g_RuntimeProfiler, WINML_RUNTIME_TEST_PERF::RUN_TEST);
    
    Log::Comment(WEX::Common::String().Format(L"Profiling data:"));
    for (int i = 0; i < WINML_RUNTIME_TEST_PERF::kCount; ++i)
    {
        WEX::Logging::Log::Comment(WEX::Common::String().Format(
            L"%s :  Time=%4.4f\tCPUUse(%%)=%4.4f\tAvgWorkingSetDelta(MB)=%4.4f\tMaxWorkingSetDelta(MB)=%4.4f\tGPUUse(%%)=%4.4f\tGPUDedicatedMem(MB)=%4.4f",
            WINML_RUNTIME_TEST_PERF_NAMES[i].c_str(),
            g_RuntimeProfiler[i].GetAverage(CounterType::TIMER),
            g_RuntimeProfiler[i].GetAverage(CounterType::CPU_USAGE),
            g_RuntimeProfiler[i].GetAverage(CounterType::WORKING_SET_USAGE),
            g_RuntimeProfiler[i].GetMax(CounterType::WORKING_SET_USAGE),
            g_RuntimeProfiler[i].GetAverage(CounterType::GPU_USAGE),
            g_RuntimeProfiler[i].GetAverage(CounterType::GPU_DEDICATED_MEM_USAGE)
        ));
    }
}
}
