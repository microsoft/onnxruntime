#ifdef _WIN32

#include "concurrent_session_etw_test_base.h"

class CPUConcurrentSessionETWTest : public ConcurrentSessionETWTestBase {
private:
    static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/squeezenet/model.onnx");

protected:
    void GetInputsAndExpectedOutputs(
        std::vector<Input>& inputs,
        std::vector<int64_t>& expected_dims_y,
        std::vector<float>& expected_values_y,
        std::string& output_name) override {

        // Setup input
        inputs.resize(1);
        Input& input = inputs.back();
        input.name = "data_0";
        input.dims = {1, 3, 224, 224};
        size_t input_tensor_size = 224 * 224 * 3;
        input.values.resize(input_tensor_size);

        // Fill input values
        for (unsigned int i = 0; i < input_tensor_size; i++) {
            input.values[i] = (float)i / (input_tensor_size + 1);
        }

        // Setup expected output
        expected_dims_y = {1, 1000, 1, 1};
        expected_values_y = {0.000045f, 0.003846f, 0.000125f, 0.001180f, 0.001317f};
        output_name = "softmaxout_1";
    }

    Ort::Session CreateSession() override {
        Ort::SessionOptions session_options;
        session_options.DisablePerSessionThreads();
        return Ort::Session(*ort_env, MODEL_URI, session_options);
    }

public:
    // Use base class constructor with default or custom parameters
    CPUConcurrentSessionETWTest(
        size_t max_sessions = 5,
        int max_iterations = 10,
        std::chrono::seconds test_duration = std::chrono::seconds(30)
    ) : ConcurrentSessionETWTestBase(max_sessions, max_iterations, test_duration) {}
};

TEST(LoggingTests, WindowsConcurrentInferenceWithETWTestCPU) {
    CPUConcurrentSessionETWTest test;
    test.RunConcurrentTest();
}

#endif // _WIN32
