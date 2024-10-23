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

// Shared test structures and functions
struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

extern std::unique_ptr<Ort::Env> ort_env;
static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/squeezenet/model.onnx");

// Required shared functions from original test
static void GetInputsAndExpectedOutputs(std::vector<Input>& inputs,
                                      std::vector<int64_t>& expected_dims_y,
                                      std::vector<float>& expected_values_y,
                                      std::string& output_name) {
    inputs.resize(1);
    Input& input = inputs.back();
    input.name = "data_0";
    input.dims = {1, 3, 224, 224};
    size_t input_tensor_size = 224 * 224 * 3;
    input.values.resize(input_tensor_size);
    auto& input_tensor_values = input.values;
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    expected_dims_y = {1, 1000, 1, 1};
    expected_values_y = {0.000045f, 0.003846f, 0.000125f, 0.001180f, 0.001317f};
    output_name = "softmaxout_1";
}

template <typename OutT>
static void RunSession(OrtAllocator& allocator, Ort::Session& session_object,
                      std::vector<Input>& inputs,
                      const char* output_name,
                      const std::vector<int64_t>& dims_y,
                      const std::vector<OutT>& values_y,
                      Ort::Value* output_tensor) {
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < inputs.size(); i++) {
        input_names.emplace_back(inputs[i].name);
        ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(allocator.Info(&allocator),
            inputs[i].values.data(), inputs[i].values.size(),
            inputs[i].dims.data(), inputs[i].dims.size()));
    }

    if (output_tensor)
        session_object.Run(Ort::RunOptions{nullptr}, input_names.data(),
            ort_inputs.data(), ort_inputs.size(), &output_name, output_tensor, 1);
    else {
        auto ort_outputs = session_object.Run(Ort::RunOptions{nullptr},
            input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
        ASSERT_EQ(ort_outputs.size(), 1u);
        output_tensor = &ort_outputs[0];
    }

    auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), dims_y);

#ifdef USE_CUDA
    const float tolerance = 1e-5f;
#else
    const float tolerance = 1e-6f;
#endif

    OutT* f = output_tensor->GetTensorMutableData<OutT>();
    for (size_t i = 0; i != static_cast<size_t>(5); ++i) {
        ASSERT_NEAR(values_y[i], f[i], tolerance);
    }
}

template <typename T, typename OutT>
static void TestInference(Ort::Session& session,
                         std::vector<Input>& inputs,
                         const char* output_name,
                         const std::vector<int64_t>& expected_dims_y,
                         const std::vector<OutT>& expected_values_y) {
    auto default_allocator = std::make_unique<MockedOrtAllocator>();
    Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(),
        expected_dims_y.data(), expected_dims_y.size());

    RunSession<OutT>(*default_allocator,
                     session,
                     inputs,
                     output_name,
                     expected_dims_y,
                     expected_values_y,
                     &value_y);
}

template <typename T, typename OutT>
static Ort::Session GetSessionObj(Ort::Env& env, T model_uri, int provider_type) {
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();

    if (provider_type == 1) {
#ifdef USE_CUDA
        OrtCUDAProviderOptionsV2* options;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&options));
        session_options.AppendExecutionProvider_CUDA_V2(*options);
        std::cout << "Running simple inference with cuda provider" << std::endl;
#else
        return Ort::Session(nullptr);
#endif
    } else {
        std::cout << "Running simple inference with default provider" << std::endl;
    }

    return Ort::Session(env, model_uri, session_options);
}

class WindowsConcurrentInferenceWithETWTest {
protected:
    const std::chrono::steady_clock::time_point test_start_time = std::chrono::steady_clock::now();

    std::string GetTimestamp() {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - test_start_time).count();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << (ms / 1000.0);
        return "[" + oss.str() + "s] ";
    }

    struct SessionWrapper {
        Ort::Session session;
        bool in_use;
        SessionWrapper(Ort::Session&& s) : session(std::move(s)), in_use(false) {}
    };

    std::vector<std::unique_ptr<SessionWrapper>> session_pool;
    std::mutex session_mutex;
    std::atomic<int> global_session_number{0};
    std::atomic<bool> should_stop{false};
    std::atomic<bool> etw_session_active{false};
    const size_t MAX_SESSIONS = 5;
    const int MAX_ITERATIONS = 10;
    const std::chrono::seconds TEST_DURATION{30};

    // Windows-specific process execution for ETW recording
    bool ExecuteWindowsProcess(const wchar_t* command, bool wait_for_completion = true) {
        STARTUPINFOW si = {sizeof(si)};
        PROCESS_INFORMATION pi;

        wchar_t cmd_buffer[MAX_PATH];
        wcscpy_s(cmd_buffer, command);

        BOOL success = CreateProcessW(
            nullptr,        // No module name (use command line)
            cmd_buffer,     // Command line
            nullptr,        // Process handle not inheritable
            nullptr,        // Thread handle not inheritable
            FALSE,         // Set handle inheritance to FALSE
            0,            // No creation flags
            nullptr,        // Use parent's environment block
            nullptr,        // Use parent's starting directory
            &si,          // Pointer to STARTUPINFO structure
            &pi           // Pointer to PROCESS_INFORMATION structure
        );

        if (!success) {
            std::wcerr << L"CreateProcess failed for ETW command: " << command
                      << L". Error: " << GetLastError() << std::endl;
            return false;
        }

        if (wait_for_completion) {
            WaitForSingleObject(pi.hProcess, INFINITE);
            DWORD exit_code;
            GetExitCodeProcess(pi.hProcess, &exit_code);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            return exit_code == 0;
        } else {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            return true;
        }
    }

    int GetRandomNumber(int min, int max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

    void Initialize() {
        std::cout << GetTimestamp() << "Test initialized. Will create sessions per inference thread." << std::endl;
    }

void ETWRecordingThread() {
        while (!should_stop) {
            std::cout << GetTimestamp() << "Starting new ETW recording session..." << std::endl;

            // Start ETW recording
            if (!ExecuteWindowsProcess(L"wpr.exe -start ..\\..\\..\\..\\ort.wprp -start ..\\..\\..\\..\\onnxruntime\\test\\platform\\windows\\logging\\etw_provider.wprp", true)) {
                std::cerr << GetTimestamp() << "Failed to start ETW recording" << std::endl;
                should_stop = true;
                continue;
            }

            etw_session_active = true;
            std::cout << GetTimestamp() << "ETW recording started successfully" << std::endl;

            // Random pause between 1-5 seconds while recording
            Sleep(GetRandomNumber(1000, 5000));

            std::cout << GetTimestamp() << "Stopping ETW recording session..." << std::endl;
            // Stop ETW recording
            if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
                std::cerr << GetTimestamp() << "Failed to stop ETW recording" << std::endl;
                should_stop = true;
                etw_session_active = false;
                continue;
            }

            etw_session_active = false;
            std::cout << GetTimestamp() << "ETW recording stopped successfully" << std::endl;

            // Random pause before next recording
            Sleep(GetRandomNumber(1000, 3000));
        }

        // Ensure ETW session is stopped if we're exiting the thread
        if (etw_session_active) {
            std::cout << GetTimestamp() << "Stopping remaining ETW recording session before exit..." << std::endl;
            if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
                std::cerr << GetTimestamp() << "Failed to stop final ETW recording" << std::endl;
            } else {
                std::cout << GetTimestamp() << "Final ETW recording stopped successfully" << std::endl;
            }
            etw_session_active = false;
        }
    }

    void InferenceWorker() {
        std::vector<Input> inputs;
        std::vector<int64_t> expected_dims_y;
        std::vector<float> expected_values_y;
        std::string output_name;
        GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

        int total_iterations = 0;
        auto thread_start_time = std::chrono::steady_clock::now();
        auto thread_id = std::this_thread::get_id();

        while (!should_stop) {
            // Get next global session number
            int current_session = ++global_session_number;

            // Create a new session
            std::cout << GetTimestamp() << "Thread " << thread_id
                     << ": Creating new session #" << current_session << "..." << std::endl;

            std::unique_ptr<Ort::Session> session;
            try {
                session = std::make_unique<Ort::Session>(GetSessionObj<PATH_TYPE, float>(*ort_env, MODEL_URI, 0));
                if (!session) {
                    std::cerr << GetTimestamp() << "Thread " << thread_id
                             << ": Failed to create session #" << current_session
                             << " (null session)" << std::endl;
                    continue;
                }
            } catch (const Ort::Exception& e) {
                std::cerr << GetTimestamp() << "Thread " << thread_id
                         << ": Exception creating session #" << current_session
                         << ": " << e.what() << std::endl;
                continue;
            }

            std::cout << GetTimestamp() << "Thread " << thread_id
                     << ": Session #" << current_session << " created successfully" << std::endl;

            // Run inferences with this session
            int session_iterations = 0;
            auto session_start_time = std::chrono::steady_clock::now();
            bool session_error = false;

            while (!should_stop && session_iterations < MAX_ITERATIONS && !session_error) {
                try {
                    TestInference<PATH_TYPE, float>(
                        *session,
                        inputs,
                        output_name.c_str(),
                        expected_dims_y,
                        expected_values_y
                    );
                    session_iterations++;
                    total_iterations++;

                    // Add periodic progress logging
                    // if (session_iterations % 5 == 0) {
                    //     std::cout << GetTimestamp() << "Thread " << thread_id
                    //              << ": Session #" << current_session
                    //              << " completed " << session_iterations
                    //              << " inferences so far" << std::endl;
                    // }
                }
                catch (const Ort::Exception& e) {
                    std::cerr << GetTimestamp() << "Thread " << thread_id
                             << ": Inference failed in session #" << current_session
                             << ": " << e.what() << std::endl;
                    session_error = true;  // Break out of session on error
                }

                // Random pause between inferences with timeout check
                for (int i = 0; i < 5 && !should_stop; i++) {  // Split sleep into smaller chunks
                    Sleep(GetRandomNumber(20, 100));
                }
            }

            auto session_end_time = std::chrono::steady_clock::now();
            auto session_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                session_end_time - session_start_time).count();

            std::cout << GetTimestamp() << "Thread " << thread_id
                     << ": Session #" << current_session
                     << " completed " << session_iterations << " inferences in "
                     << session_duration / 1000.0 << " seconds" << std::endl;

            // Explicitly destroy the session
            std::cout << GetTimestamp() << "Thread " << thread_id
                     << ": Destroying session #" << current_session << "..." << std::endl;

            session.reset(); // Properly destroy the session using smart pointer

            std::cout << GetTimestamp() << "Thread " << thread_id
                     << ": Session #" << current_session << " destroyed" << std::endl;

            // Check if we should stop
            if (std::chrono::steady_clock::now() - test_start_time >= TEST_DURATION) {
                should_stop = true;
            }

            // Small pause between sessions with timeout check
            for (int i = 0; i < 5 && !should_stop; i++) {
                Sleep(20);
            }
        }

        auto thread_end_time = std::chrono::steady_clock::now();
        auto thread_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            thread_end_time - thread_start_time).count();

        std::cout << GetTimestamp() << "\nThread " << thread_id << " Summary:" << std::endl;
        std::cout << GetTimestamp() << "Total inferences completed: " << total_iterations << std::endl;
        std::cout << GetTimestamp() << "Total running time: " << thread_duration / 1000.0 << " seconds" << std::endl;
    }
public:
    void RunConcurrentTest() {
        std::cout << GetTimestamp() << "Starting Windows Concurrent Inference Test with ETW Recording..." << std::endl;
        std::cout << GetTimestamp() << "Configuration: MAX_SESSIONS=" << MAX_SESSIONS
                 << ", MAX_ITERATIONS=" << MAX_ITERATIONS
                 << ", TEST_DURATION=" << TEST_DURATION.count() << "s" << std::endl;

        Initialize();
        auto start_time = std::chrono::steady_clock::now();

        // Start ETW recording thread
        std::thread etw_thread(&WindowsConcurrentInferenceWithETWTest::ETWRecordingThread, this);

        // Start inference worker threads
        std::vector<std::thread> inference_threads;
        for (size_t i = 0; i < MAX_SESSIONS; ++i) {
            inference_threads.emplace_back(&WindowsConcurrentInferenceWithETWTest::InferenceWorker, this);
        }

        // Wait for test duration
        Sleep(TEST_DURATION.count() * 1000);

        // Signal threads to stop
        should_stop = true;

        // Ensure ETW thread stops first and cleans up
        if (etw_thread.joinable()) {
            etw_thread.join();
        }

        // Additional safety check for ETW session
        if (etw_session_active) {
            std::cout << GetTimestamp() << "Cleaning up lingering ETW session..." << std::endl;
            if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
                std::cerr << GetTimestamp() << "Failed to stop lingering ETW session" << std::endl;
            } else {
                std::cout << GetTimestamp() << "Lingering ETW session stopped successfully" << std::endl;
            }
            etw_session_active = false;
        }

        // Wait for inference threads
        for (auto& thread : inference_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();

        std::cout << GetTimestamp() << "\nTest Summary:" << std::endl;
        std::cout << GetTimestamp() << "-------------" << std::endl;
        std::cout << GetTimestamp() << "Total test duration: " << duration << " seconds" << std::endl;
        std::cout << GetTimestamp() << "Total sessions created: " << global_session_number << std::endl;
        std::cout << GetTimestamp() << "ETW session final state: "
                 << (etw_session_active ? "Active (Cleaned up)" : "Inactive") << std::endl;
    }
};

TEST(LoggingTests, WindowsConcurrentInferenceWithETWTest) {
    WindowsConcurrentInferenceWithETWTest test;
    test.RunConcurrentTest();
}

#endif // _WIN32
