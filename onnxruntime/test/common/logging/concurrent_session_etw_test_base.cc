#ifdef _WIN32

#include "concurrent_session_etw_test_base.h"

ConcurrentSessionETWTestBase::ConcurrentSessionETWTestBase(
    size_t max_sessions,
    int max_iterations,
    std::chrono::seconds test_duration
) : test_start_time(std::chrono::steady_clock::now()),
    MAX_SESSIONS(max_sessions),
    MAX_ITERATIONS(max_iterations),
    TEST_DURATION(test_duration) {}

std::string ConcurrentSessionETWTestBase::GetTimestamp() const {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - test_start_time).count();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << (ms / 1000.0);
    return "[" + oss.str() + "s] ";
}

int ConcurrentSessionETWTestBase::GetRandomNumber(int min, int max) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

bool ConcurrentSessionETWTestBase::ExecuteWindowsProcess(const wchar_t* command, bool wait_for_completion) {
    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;

    wchar_t cmd_buffer[MAX_PATH];
    wcscpy_s(cmd_buffer, command);

    BOOL success = CreateProcessW(nullptr, cmd_buffer, nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi);
    if (!success) {
        std::wcerr << L"CreateProcess failed for ETW command: " << command << L". Error: " << GetLastError() << std::endl;
        return false;
    }

    if (wait_for_completion) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return exit_code == 0;
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return true;
}

void ConcurrentSessionETWTestBase::ETWRecordingThread() {
    while (!should_stop) {
        std::cout << GetTimestamp() << "Starting new ETW recording session..." << std::endl;

        if (!ExecuteWindowsProcess(L"wpr.exe -start ..\\..\\..\\..\\ort.wprp -start ..\\..\\..\\..\\onnxruntime\\test\\platform\\windows\\logging\\etw_provider.wprp", true)) {
            std::cerr << GetTimestamp() << "Failed to start ETW recording" << std::endl;
            should_stop = true;
            continue;
        }

        etw_session_active = true;
        Sleep(GetRandomNumber(1000, 5000));

        std::cout << GetTimestamp() << "Stopping ETW recording session..." << std::endl;
        if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
            std::cerr << GetTimestamp() << "Failed to stop ETW recording" << std::endl;
            should_stop = true;
            etw_session_active = false;
            continue;
        }

        etw_session_active = false;
        Sleep(GetRandomNumber(1000, 3000));
    }

    if (etw_session_active) {
        if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
            std::cerr << GetTimestamp() << "Failed to stop final ETW recording" << std::endl;
        }
        etw_session_active = false;
    }
}

void ConcurrentSessionETWTestBase::InferenceWorker() {
    std::vector<Input> inputs;
    std::vector<int64_t> expected_dims_y;
    std::vector<float> expected_values_y;
    std::string output_name;
    GetInputsAndExpectedOutputs(inputs, expected_dims_y, expected_values_y, output_name);

    int total_iterations = 0;
    auto thread_start_time = std::chrono::steady_clock::now();
    auto thread_id = std::this_thread::get_id();

    while (!should_stop) {
        int current_session = ++global_session_number;

        std::cout << GetTimestamp() << "Thread " << thread_id
                 << ": Creating new session #" << current_session << "..." << std::endl;

        std::unique_ptr<Ort::Session> session;
        try {
            session = std::make_unique<Ort::Session>(CreateSession());
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

        int session_iterations = 0;
        auto session_start_time = std::chrono::steady_clock::now();
        bool session_error = false;

        while (!should_stop && session_iterations < MAX_ITERATIONS && !session_error) {
            try {
                TestInference<float>(
                    *session,
                    inputs,
                    output_name.c_str(),
                    expected_dims_y,
                    expected_values_y
                );
                session_iterations++;
                total_iterations++;
            }
            catch (const Ort::Exception& e) {
                std::cerr << GetTimestamp() << "Thread " << thread_id
                         << ": Inference failed in session #" << current_session
                         << ": " << e.what() << std::endl;
                session_error = true;
            }

            for (int i = 0; i < 5 && !should_stop; i++) {
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

        session.reset();

        if (std::chrono::steady_clock::now() - test_start_time >= TEST_DURATION) {
            should_stop = true;
        }

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

void ConcurrentSessionETWTestBase::RunConcurrentTest() {
    std::cout << GetTimestamp() << "Starting Concurrent Inference Test with ETW Recording..." << std::endl;
    std::cout << GetTimestamp() << "Configuration: MAX_SESSIONS=" << MAX_SESSIONS
              << ", MAX_ITERATIONS=" << MAX_ITERATIONS
              << ", TEST_DURATION=" << TEST_DURATION.count() << "s" << std::endl;

    auto start_time = std::chrono::steady_clock::now();

    // Start ETW recording thread
    std::thread etw_thread(&ConcurrentSessionETWTestBase::ETWRecordingThread, this);

    // Start inference threads
    std::vector<std::thread> inference_threads;
    for (size_t i = 0; i < MAX_SESSIONS; ++i) {
        inference_threads.emplace_back(&ConcurrentSessionETWTestBase::InferenceWorker, this);
    }

    // Wait for test duration
    Sleep(TEST_DURATION.count() * 1000);
    should_stop = true;

    // Wait for ETW thread
    if (etw_thread.joinable()) {
        etw_thread.join();
    }

    // Final ETW cleanup if needed
    if (etw_session_active) {
        std::cout << GetTimestamp() << "Cleaning up lingering ETW session..." << std::endl;
        if (!ExecuteWindowsProcess(L"wpr.exe -stop ort.etl -skipPdbGen", true)) {
            std::cerr << GetTimestamp() << "Failed to stop lingering ETW session" << std::endl;
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
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << GetTimestamp() << "\nTest Summary:" << std::endl;
    std::cout << GetTimestamp() << "-------------" << std::endl;
    std::cout << GetTimestamp() << "Total test duration: " << duration << " seconds" << std::endl;
    std::cout << GetTimestamp() << "Total sessions created: " << global_session_number << std::endl;
    std::cout << GetTimestamp() << "ETW session final state: "
              << (etw_session_active ? "Active (Cleaned up)" : "Inactive") << std::endl;
}

#endif // _WIN32
