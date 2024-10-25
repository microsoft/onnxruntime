#pragma once

#ifdef _WIN32

#include <random>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <windows.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/constants.h"
#include "providers.h"
#include <gtest/gtest.h>
#include "../../shared_lib/test_fixture.h"
#include "test_allocator.h"
#include <iomanip>

// Common input structure
struct Input {
    const char* name = nullptr;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

extern std::unique_ptr<Ort::Env> ort_env;

// Base class for concurrent inference tests with ETW
class ConcurrentSessionETWTestBase {
protected:
    const std::chrono::steady_clock::time_point test_start_time;
    std::atomic<int> global_session_number{0};
    std::atomic<bool> should_stop{false};
    std::atomic<bool> etw_session_active{false};
    const size_t MAX_SESSIONS;
    const int MAX_ITERATIONS;
    const std::chrono::seconds TEST_DURATION;

    // Constructor with configurable parameters
    ConcurrentSessionETWTestBase(
        size_t max_sessions = 5,
        int max_iterations = 10,
        std::chrono::seconds test_duration = std::chrono::seconds(30)
    );

    virtual ~ConcurrentSessionETWTestBase() = default;

    // Virtual methods that must be implemented by derived classes
    virtual void GetInputsAndExpectedOutputs(
        std::vector<Input>& inputs,
        std::vector<int64_t>& expected_dims_y,
        std::vector<float>& expected_values_y,
        std::string& output_name) = 0;

    virtual Ort::Session CreateSession() = 0;

    // Protected utility methods
    std::string GetTimestamp() const;
    int GetRandomNumber(int min, int max) const;
    bool ExecuteWindowsProcess(const wchar_t* command, bool wait_for_completion = true);
    void ETWRecordingThread();
    void InferenceWorker();

    template <typename OutT>
    void RunSession(
        OrtAllocator& allocator,
        Ort::Session& session_object,
        std::vector<Input>& inputs,
        const char* output_name,
        const std::vector<int64_t>& dims_y,
        const std::vector<OutT>& values_y,
        Ort::Value* output_tensor) {

        std::vector<Ort::Value> ort_inputs;
        std::vector<const char*> input_names;

        for (size_t i = 0; i < inputs.size(); i++) {
            input_names.emplace_back(inputs[i].name);
            ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                allocator.Info(&allocator),
                inputs[i].values.data(),
                inputs[i].values.size(),
                inputs[i].dims.data(),
                inputs[i].dims.size()
            ));
        }

        if (output_tensor) {
            session_object.Run(Ort::RunOptions{nullptr}, input_names.data(),
                ort_inputs.data(), ort_inputs.size(), &output_name, output_tensor, 1);
        } else {
            auto ort_outputs = session_object.Run(Ort::RunOptions{nullptr},
                input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
            ASSERT_EQ(ort_outputs.size(), 1u);
            output_tensor = &ort_outputs[0];
        }

        auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
        ASSERT_EQ(type_info.GetShape(), dims_y);

        const float tolerance =
    #ifdef USE_CUDA
            1e-5f;
    #else
            1e-6f;
    #endif

        OutT* f = output_tensor->GetTensorMutableData<OutT>();
        for (size_t i = 0; i != static_cast<size_t>(5); ++i) {
            ASSERT_NEAR(values_y[i], f[i], tolerance);
        }
    }

    template <typename OutT>
    void TestInference(
        Ort::Session& session,
        std::vector<Input>& inputs,
        const char* output_name,
        const std::vector<int64_t>& expected_dims_y,
        const std::vector<OutT>& expected_values_y) {

        auto default_allocator = std::make_unique<MockedOrtAllocator>();
        Ort::Value value_y = Ort::Value::CreateTensor<float>(
            default_allocator.get(),
            expected_dims_y.data(),
            expected_dims_y.size()
        );

        RunSession<OutT>(
            *default_allocator,
            session,
            inputs,
            output_name,
            expected_dims_y,
            expected_values_y,
            &value_y
        );
    }

public:
    void RunConcurrentTest();
};

#endif // _WIN32
