#ifdef _WIN32

#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/logging/concurrent_session_etw_test_base.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/qnn/qnn_test_utils.h"

class QNNConcurrentSessionETWTest : public ConcurrentSessionETWTestBase {
private:
    static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/constant_floats.onnx");
    const bool use_htp_;

protected:
    void GetInputsAndExpectedOutputs(
        std::vector<Input>& inputs,
        std::vector<int64_t>& expected_dims_y,
        std::vector<float>& expected_values_y,
        std::string& output_name) override {

        // Setup input data
        inputs.resize(1);
        Input& input = inputs.back();
        input.name = "data_0";
        input.dims = {1, 2, 2, 2};  // NCHW format
        size_t input_tensor_size = 1 * 2 * 2 * 2;
        input.values.resize(input_tensor_size);

        // Fill input with sequential values
        for (unsigned int i = 0; i < input_tensor_size; i++) {
            input.values[i] = static_cast<float>(i) / static_cast<float>(input_tensor_size);
        }

        // Setup expected output - matches basic QNN test expected values
        expected_dims_y = {1, 2, 2, 2};  // Same shape as input for this test
        expected_values_y = {0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f};
        output_name = "output";
    }

    Ort::Session CreateSession() override {
        Ort::SessionOptions session_options;
        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
        session_options.DisablePerSessionThreads();

        // Setup QNN provider options
        onnxruntime::ProviderOptions qnn_options;
        qnn_options["backend_path"] = use_htp_ ? "QnnHtp.dll" : "QnnCpu.dll";
        if (use_htp_) {
            session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }

        if (use_htp_) {
            qnn_options["enable_htp_fp16_precision"] = "1";
        }

        session_options.AppendExecutionProvider("QNN", qnn_options);

        return Ort::Session(*ort_env, MODEL_URI, session_options);
    }

public:
    QNNConcurrentSessionETWTest(
        bool use_htp = false,
        size_t max_sessions = 5,
        int max_iterations = 10,
        std::chrono::seconds test_duration = std::chrono::seconds(30)
    ) : ConcurrentSessionETWTestBase(max_sessions, max_iterations, test_duration),
        use_htp_(use_htp) {}
};

// Test for QNN CPU backend
TEST(LoggingTests, WindowsConcurrentSessionsWithETWTestQnnCpu) {
    QNNConcurrentSessionETWTest test(false);  // false = use CPU backend
    test.RunConcurrentTest();
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
// Test for QNN HTP backend
TEST(LoggingTests, WindowsConcurrentSessionsWithETWTestQnnHtp) {
    QNNConcurrentSessionETWTest test(true);  // true = use HTP backend
    test.RunConcurrentTest();
}
#endif

#endif // _WIN32
