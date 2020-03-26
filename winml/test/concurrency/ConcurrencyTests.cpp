#include "testPch.h"

#include "concurrencytests.h"
#include "model.h"
#include "SqueezeNetValidator.h"
#include "threadPool.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt;

namespace
{
void LoadBindEvalSqueezenetRealDataWithValidationConcurrently()
{
    WINML_SKIP_TEST("Skipping due to bug 21617097");

    constexpr auto load_test_model = [](const std::string& instance, LearningModelDeviceKind device)
    {
        WinML::Engine::Test::ModelValidator::SqueezeNet(instance, device, 0.00001f, false);
    };

    std::vector<std::thread> threads;
    for (const auto& instance : {"1", "2", "3", "4"})
    {
        threads.emplace_back(load_test_model, instance, LearningModelDeviceKind::Cpu);
    }
    if (GPUTEST_ENABLED) {
        for (const auto& instance : {"GPU_1", "GPU_2", "GPU_3", "GPU_4"}) {
            threads.emplace_back(load_test_model, instance, LearningModelDeviceKind::DirectX);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void ConcurrencyTestsApiSetup()
{
    init_apartment();
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

struct EvaluationUnit
{
    LearningModel model;
    LearningModelSession session;
    LearningModelBinding binding;
    winrt::Windows::Foundation::IAsyncOperation<LearningModelEvaluationResult> operation;
    LearningModelEvaluationResult result;

    EvaluationUnit() : model(nullptr), session(nullptr), binding(nullptr), result(nullptr) {}
};

// Run EvalAsync for each unit concurrently and get results
static void RunAsync(std::vector<EvaluationUnit> &evaluation_units)
{
    std::for_each(evaluation_units.begin(), evaluation_units.end(), [](EvaluationUnit &unit) {
        unit.operation = unit.session.EvaluateAsync(unit.binding, L"");
    });
    // get results
    std::for_each(evaluation_units.begin(), evaluation_units.end(), [](EvaluationUnit &unit) {
        unit.result = unit.operation.get();
    });
}

static void VerifyEvaluation(const std::vector<EvaluationUnit> &evaluation_units, std::vector<uint32_t> expected_indices)
{
    assert(evaluation_units.size() == expected_indices.size());
    for (size_t i = 0; i < evaluation_units.size(); ++i)
    {
        auto unit = evaluation_units[i];
        auto expectedIndex = expected_indices[i];
        auto result = unit.result.Outputs().Lookup(L"softmaxout_1").as<TensorFloat>().GetAsVectorView();
        int64_t maxIndex = 0;
        float maxValue = 0;
        for (uint32_t j = 0; j < result.Size(); ++j)
        {
            float val = result.GetAt(j);
            if (val > maxValue)
            {
                maxValue = val;
                maxIndex = j;
            }
        }
        WINML_EXPECT_TRUE(maxIndex == expectedIndex);
    }
}

static void copyFile(LPCWSTR from, LPCWSTR to)
{
    using namespace std;
    ifstream source(FileHelpers::GetModulePath() + from, ios::binary);
    ofstream dest(FileHelpers::GetModulePath() + to, ios::binary);

    dest << source.rdbuf();
}

// Run evaluations with different models
void EvalAsyncDifferentModels()
{
    copyFile(L"model.onnx", L"model2.onnx");

    std::vector<std::wstring> model_paths = { L"model.onnx", L"model2.onnx" };
    const unsigned int num_units = static_cast<unsigned int>(model_paths.size());
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());

    auto ifv = FileHelpers::LoadImageFeatureValue(L"kitten_224.png");

    for (unsigned int i = 0; i < num_units; ++i)
    {
        evaluation_units[i].model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + model_paths[i]);
        evaluation_units[i].session = LearningModelSession(evaluation_units[i].model);
        evaluation_units[i].binding = LearningModelBinding(evaluation_units[i].session);
        evaluation_units[i].binding.Bind(L"data_0", ifv);
    }

    RunAsync(evaluation_units);
    std::vector<uint32_t> indices(num_units, TABBY_CAT_INDEX);
    VerifyEvaluation(evaluation_units, indices);
}

// Run evaluations with same model, different sessions
void EvalAsyncDifferentSessions()
{
    unsigned int num_units = 3;
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());
    auto ifv = FileHelpers::LoadImageFeatureValue(L"kitten_224.png");

    // same model, different session
    auto model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"model.onnx");
    for (unsigned int i = 0; i < num_units; ++i)
    {
        evaluation_units[i].model = model;
        evaluation_units[i].session = LearningModelSession(evaluation_units[i].model);
        evaluation_units[i].binding = LearningModelBinding(evaluation_units[i].session);
        evaluation_units[i].binding.Bind(L"data_0", ifv);
    }

    RunAsync(evaluation_units);
    std::vector<uint32_t> indices(num_units, TABBY_CAT_INDEX);
    VerifyEvaluation(evaluation_units, indices);
}

// Run evaluations with same session (and model), with different bindings
void EvalAsyncDifferentBindings()
{
    unsigned int num_units = 2;
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());

    std::vector<ImageFeatureValue> ifvs = {FileHelpers::LoadImageFeatureValue(L"kitten_224.png"),
                                           FileHelpers::LoadImageFeatureValue(L"fish.png")};

    // same session, different binding
    auto model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"model.onnx");
    auto session = LearningModelSession(model);
    for (unsigned int i = 0; i < num_units; ++i)
    {
        evaluation_units[i].model = model;
        evaluation_units[i].session = session;
        evaluation_units[i].binding = LearningModelBinding(evaluation_units[i].session);
        evaluation_units[i].binding.Bind(L"data_0", ifvs[i]);
    }

    RunAsync(evaluation_units);
    VerifyEvaluation(evaluation_units, { TABBY_CAT_INDEX, TENCH_INDEX });
}

winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor UnusedCreateFeatureDescriptor(
    std::shared_ptr<onnxruntime::Model> model,
    const std::wstring& name,
    const std::wstring& description,
    bool is_required,
    const ::onnx::TypeProto *type_proto);

// Get random number in interval [1,max_number]
unsigned int getRandomNumber(unsigned int max_number) {
    return std::rand() % max_number + 1;
}

void MultiThreadLoadModel() {
    // load same model
    auto path = FileHelpers::GetModulePath() + L"model.onnx";
    unsigned int num_threads = getRandomNumber(MAX_THREADS);
    ThreadPool pool(num_threads);
    try {
        for (unsigned int i = 0; i < num_threads; ++i) {
            pool.SubmitWork([&path]() {
                auto model = LearningModel::LoadFromFilePath(path);
                std::wstring name(model.Name());
                WINML_EXPECT_EQUAL(name, L"squeezenet_old");
            });
        }
    }
    catch (...) {
        WINML_LOG_ERROR("Failed to load model concurrently.");
    }
}

// Create different sessions for each thread, and evaluate
void MultiThreadSession()
{
    auto path = FileHelpers::GetModulePath() + L"model.onnx";
    auto model = LearningModel::LoadFromFilePath(path);
    unsigned int num_threads = getRandomNumber(MAX_THREADS);
    std::vector<ImageFeatureValue> ivfs = {
        FileHelpers::LoadImageFeatureValue(L"kitten_224.png"),
        FileHelpers::LoadImageFeatureValue(L"fish.png")
    };
    std::vector<int> max_indices = {
        281, // tabby, tabby cat
        0 // tench, Tinca tinca
    };
    std::vector<float> max_values = {
        0.9314f,
        0.7385f
    };
    float tolerance = 0.001f;

    ThreadPool pool(num_threads);
    try
    {
        for (unsigned i = 0; i < num_threads; ++i)
        {
           pool.SubmitWork([&model,&ivfs,&max_indices,&max_values,tolerance,i]() {
                // TODO: add variations of CPU/GPU, etc
                auto j = i % ivfs.size();
                auto input = ivfs[j];
                auto expected_index = max_indices[j];
                auto expected_value = max_values[j];
                LearningModelSession session(model);
                std::wstring name(session.Model().Name());
                LearningModelBinding bind(session);
                bind.Bind(L"data_0", input);
                auto result = session.Evaluate(bind, L"").Outputs();
                auto softmax = result.Lookup(L"softmaxout_1");
                if (auto tensor = softmax.try_as<ITensorFloat>())
                {
                    auto view = tensor.GetAsVectorView();
                    float max_val = .0f;
                    int max_index = -1;
                    for (uint32_t k = 0; k < view.Size(); ++k)
                    {
                        auto val = view.GetAt(k);
                        if (val > max_val)
                        {
                            max_index = k;
                            max_val = val;
                        }
                    }
                    WINML_EXPECT_EQUAL(expected_index, max_index);
                    WINML_EXPECT_TRUE(std::abs(expected_value - max_val) < tolerance);
                }
           });
        }
    }
    catch (...) {
        WINML_LOG_ERROR("Failed to create session concurrently.");
    }
}
}

const ConcurrencyTestsApi& getapi() {
  static constexpr ConcurrencyTestsApi api = {
    ConcurrencyTestsApiSetup,
    LoadBindEvalSqueezenetRealDataWithValidationConcurrently,
    MultiThreadLoadModel,
    MultiThreadSession,
    EvalAsyncDifferentModels,
    EvalAsyncDifferentSessions,
    EvalAsyncDifferentBindings
  };
  return api;
}
