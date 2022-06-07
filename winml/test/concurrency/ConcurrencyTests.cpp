#include "testPch.h"

#include "concurrencytests.h"
#include "model.h"
#include "SqueezeNetValidator.h"
#include "threadPool.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace winml;
using namespace winrt;

namespace {
void LoadBindEvalSqueezenetRealDataWithValidationConcurrently() {
    WINML_SKIP_TEST("Skipping due to bug 21617097");

    constexpr auto load_test_model = [](const std::string& instance, LearningModelDeviceKind device) {
        WinML::Engine::Test::ModelValidator::SqueezeNet(instance, device, 0.00001f, false);
    };

    std::vector<std::thread> threads;
    for (const auto& instance : {"1", "2", "3", "4"}) {
        threads.emplace_back(load_test_model, instance, LearningModelDeviceKind::Cpu);
    }
    if (SkipGpuTests()) {} else {
      for (const auto& instance : {"GPU_1", "GPU_2", "GPU_3", "GPU_4"}) {
        threads.emplace_back(load_test_model, instance, LearningModelDeviceKind::DirectX);
      }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void ConcurrencyTestsClassSetup() {
    init_apartment();
    std::srand(static_cast<unsigned>(std::time(nullptr)));
#ifdef BUILD_INBOX
    winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

struct EvaluationUnit {
    LearningModel model;
    LearningModelSession session;
    LearningModelBinding binding;
    wf::IAsyncOperation<LearningModelEvaluationResult> operation;
    LearningModelEvaluationResult result;

    EvaluationUnit() : model(nullptr), session(nullptr), binding(nullptr), result(nullptr) {}
};

// Run EvalAsync for each unit concurrently and get results
void RunAsync(std::vector<EvaluationUnit> &evaluation_units) {
    std::for_each(evaluation_units.begin(), evaluation_units.end(), [](EvaluationUnit &unit) {
        unit.operation = unit.session.EvaluateAsync(unit.binding, L"");
    });
    // get results
    std::for_each(evaluation_units.begin(), evaluation_units.end(), [](EvaluationUnit &unit) {
        unit.result = unit.operation.get();
    });
}

void VerifyEvaluation(const std::vector<EvaluationUnit> &evaluation_units, std::vector<uint32_t> expected_indices) {
    assert(evaluation_units.size() == expected_indices.size());
    for (size_t i = 0; i < evaluation_units.size(); ++i) {
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

void CopyLocalFile(LPCWSTR from, LPCWSTR to) {
    using namespace std;
    ifstream source(FileHelpers::GetModulePath() + from, ios::binary);
    ofstream dest(FileHelpers::GetModulePath() + to, ios::binary);

    dest << source.rdbuf();
}

// Run evaluations with different models
void EvalAsyncDifferentModels() {
    CopyLocalFile(L"model.onnx", L"model2.onnx");

    std::vector<std::wstring> model_paths = { L"model.onnx", L"model2.onnx" };
    const unsigned int num_units = static_cast<unsigned int>(model_paths.size());
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());

    auto ifv = FileHelpers::LoadImageFeatureValue(L"kitten_224.png");

    for (unsigned int i = 0; i < num_units; ++i) {
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
void EvalAsyncDifferentSessions() {
    unsigned int num_units = 3;
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());
    auto ifv = FileHelpers::LoadImageFeatureValue(L"kitten_224.png");

    // same model, different session
    auto model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"model.onnx");
    for (unsigned int i = 0; i < num_units; ++i) {
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
void EvalAsyncDifferentBindings() {
    unsigned int num_units = 2;
    std::vector<EvaluationUnit> evaluation_units(num_units, EvaluationUnit());

    std::vector<ImageFeatureValue> ifvs = {FileHelpers::LoadImageFeatureValue(L"kitten_224.png"),
                                           FileHelpers::LoadImageFeatureValue(L"fish.png")};

    // same session, different binding
    auto model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"model.onnx");
    auto session = LearningModelSession(model);
    for (unsigned int i = 0; i < num_units; ++i) {
        evaluation_units[i].model = model;
        evaluation_units[i].session = session;
        evaluation_units[i].binding = LearningModelBinding(evaluation_units[i].session);
        evaluation_units[i].binding.Bind(L"data_0", ifvs[i]);
    }

    RunAsync(evaluation_units);
    VerifyEvaluation(evaluation_units, { TABBY_CAT_INDEX, TENCH_INDEX });
}

winml::ILearningModelFeatureDescriptor UnusedCreateFeatureDescriptor(
    std::shared_ptr<onnxruntime::Model> model,
    const std::wstring& name,
    const std::wstring& description,
    bool is_required,
    const ::onnx::TypeProto *type_proto);

// Get random number in interval [1,max_number]
unsigned int GetRandomNumber(unsigned int max_number) {
    return std::rand() % max_number + 1;
}

void MultiThreadLoadModel() {
    // load same model
    auto path = FileHelpers::GetModulePath() + L"model.onnx";
    ThreadPool pool(NUM_THREADS);
    try {
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
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

void MultiThreadMultiSessionOnDevice(const LearningModelDevice& device) {
    auto path = FileHelpers::GetModulePath() + L"model.onnx";
    auto model = LearningModel::LoadFromFilePath(path);
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
    std::vector<LearningModelSession> modelSessions(NUM_THREADS, nullptr);
    ThreadPool pool(NUM_THREADS);
    try {
        device.as<IMetacommandsController>()->SetMetacommandsEnabled(false);
        // create all the sessions
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            modelSessions[i] = LearningModelSession(model, device);
        }
        // start all the threads
        for (unsigned i_thread = 0; i_thread < NUM_THREADS; ++i_thread) {
            LearningModelSession &model_session = modelSessions[i_thread];
            pool.SubmitWork([&model_session,&ivfs,&max_indices,&max_values,tolerance,i_thread]() {
                ULONGLONG start_time = GetTickCount64();
                while (((GetTickCount64() - start_time) / 1000) < NUM_SECONDS) {
                    auto j = i_thread % ivfs.size();
                    auto input = ivfs[j];
                    auto expected_index = max_indices[j];
                    auto expected_value = max_values[j];
                    LearningModelBinding bind(model_session);
                    bind.Bind(L"data_0", input);
                    auto result = model_session.Evaluate(bind, L"").Outputs();
                    auto softmax = result.Lookup(L"softmaxout_1");
                    if (auto tensor = softmax.try_as<ITensorFloat>()) {
                        auto view = tensor.GetAsVectorView();
                        float max_val = .0f;
                        int max_index = -1;
                        for (uint32_t i = 0; i < view.Size(); ++i) {
                            auto val = view.GetAt(i);
                            if (val > max_val)
                            {
                                max_index = i;
                                max_val = val;
                            }
                        }
                        WINML_EXPECT_EQUAL(expected_index, max_index);
                        WINML_EXPECT_TRUE(std::abs(expected_value - max_val) < tolerance);
                    }
                }
           });
        }
    }
    catch (...) {
        WINML_LOG_ERROR("Failed to create session concurrently.");
    }
}

void MultiThreadMultiSession() {
    MultiThreadMultiSessionOnDevice(LearningModelDevice(LearningModelDeviceKind::Cpu));
}

void MultiThreadMultiSessionGpu() {
    MultiThreadMultiSessionOnDevice(LearningModelDevice(LearningModelDeviceKind::DirectX));
}

// Create different sessions for each thread, and evaluate
void MultiThreadSingleSessionOnDevice(const LearningModelDevice& device) {
    auto path = FileHelpers::GetModulePath() + L"model.onnx";
    auto model = LearningModel::LoadFromFilePath(path);
    LearningModelSession model_session = nullptr;
    WINML_EXPECT_NO_THROW(model_session = LearningModelSession(model, device));
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

    ThreadPool pool(NUM_THREADS);
    try {
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
           pool.SubmitWork([&model_session, &ivfs, &max_indices, &max_values, tolerance, i]() {
                ULONGLONG start_time = GetTickCount64();
                while (((GetTickCount64() - start_time) / 1000) < NUM_SECONDS) {
                    auto j = i % ivfs.size();
                    auto input = ivfs[j];
                    auto expected_index = max_indices[j];
                    auto expected_value = max_values[j];
                    std::wstring name(model_session.Model().Name());
                    LearningModelBinding bind(model_session);
                    bind.Bind(L"data_0", input);
                    auto result = model_session.Evaluate(bind, L"").Outputs();
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
                }
           });
        }
    }
    catch (...) {
        WINML_LOG_ERROR("Failed to create session concurrently.");
    }
}

void MultiThreadSingleSession() {
    MultiThreadSingleSessionOnDevice(LearningModelDevice(LearningModelDeviceKind::Cpu));
}

void MultiThreadSingleSessionGpu() {
    MultiThreadSingleSessionOnDevice(LearningModelDevice(LearningModelDeviceKind::DirectX));
}
}

const ConcurrencyTestsApi& getapi() {
  static ConcurrencyTestsApi api = {
    ConcurrencyTestsClassSetup,
    LoadBindEvalSqueezenetRealDataWithValidationConcurrently,
    MultiThreadLoadModel,
    MultiThreadMultiSession,
    MultiThreadMultiSessionGpu,
    MultiThreadSingleSession,
    MultiThreadSingleSessionGpu,
    EvalAsyncDifferentModels,
    EvalAsyncDifferentSessions,
    EvalAsyncDifferentBindings
  };

  if (SkipGpuTests()) {
    api.MultiThreadMultiSessionGpu = SkipTest;
    api.MultiThreadSingleSessionGpu = SkipTest;
  }
  return api;
}
