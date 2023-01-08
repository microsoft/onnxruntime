/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "gtest/gtest.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/constants.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/framework/test_utils.h"
#include <thread>

namespace onnxruntime {
namespace test {

#define ONNX_SHORTCUT_FILE_PATH ("testdata/hailo/shortcut_example.onnx")
#define ONNX_MULTIPULE_OUTPUTS_FILE_PATH ("testdata/hailo/multiple_outputs.onnx")

#define DEFAULT_DEVICE_ID (0)
#define DEFAULT_CONST_VALUE (0xAB)
#define DEFAULT_TEST_REPETITIONS (100)
typedef float float32_t;

template <class T>
using ShapeValuesPairsVector = std::vector<std::pair<std::vector<int64_t>, std::vector<T>>>;

template <typename T>
std::vector<T> create_const_dataset(size_t size, const T const_val = DEFAULT_CONST_VALUE)
{
    return std::vector<T>(size, const_val);
}

class TestDataShortcut
{
public:
    TestDataShortcut(const float32_t const_val = DEFAULT_CONST_VALUE) :
        m_model_path(ONNX_SHORTCUT_FILE_PATH),
        m_input_shape({1, 3, 24, 16}),
        m_input_name("x.1"),
        m_output_names({"28"})
    {
        size_t input_size = m_input_shape[0] * m_input_shape[1] * m_input_shape[2] * m_input_shape[3];
        m_input_dataset = create_const_dataset<float32_t>(input_size, const_val);

        const size_t outputs_count = 1;
        m_expected_results.reserve(outputs_count);
        m_expected_results.emplace_back(m_input_shape, m_input_dataset); 
    }
    
    std::string m_model_path;
    std::vector<int64_t> m_input_shape;
    std::string m_input_name;
    std::vector<std::string> m_output_names;
    std::vector<float32_t> m_input_dataset;
    ShapeValuesPairsVector<float32_t> m_expected_results;
};

class TestDataMultipleOutputs
{
public:
    TestDataMultipleOutputs(uint32_t frames_count = 1) :
        m_model_path(ONNX_MULTIPULE_OUTPUTS_FILE_PATH),
        m_input_shape({frames_count, 3, 4, 4}),
        m_input_name("input0"),
        m_output_names({"output0", "output1"})
    {
        size_t input_size = m_input_shape[0] * m_input_shape[1] * m_input_shape[2] * m_input_shape[3];
        m_input_dataset = create_const_dataset<uint8_t>(input_size);

        const size_t outputs_count = 2;
        m_expected_results.reserve(outputs_count);

        std::vector<int64_t> output0_shape = {frames_count, 8, 4, 4};
        size_t output0_size = output0_shape[0] * output0_shape[1] * output0_shape[2] * output0_shape[3];
        const uint8_t output0_expected_byte = 0x8b;
        m_expected_results.emplace_back(std::move(output0_shape), create_const_dataset(output0_size, output0_expected_byte));

        std::vector<int64_t> output1_shape = {frames_count, 3, 4, 4};
        size_t output1_size = output1_shape[0] * output1_shape[1] * output1_shape[2] * output1_shape[3];
        const uint8_t output1_expected_byte = 0x2f;
        m_expected_results.emplace_back(std::move(output1_shape), create_const_dataset(output1_size, output1_expected_byte));
    }
    
    std::string m_model_path;
    std::vector<int64_t> m_input_shape;
    std::string m_input_name;
    std::vector<std::string> m_output_names;
    std::vector<uint8_t> m_input_dataset;
    ShapeValuesPairsVector<uint8_t> m_expected_results;
};

template <typename T>
void verify_single_result(const OrtValue& fetched_val, const std::vector<int64_t> &expected_dims, const std::vector<T> &expected_values)
{
    auto& tensor = fetched_val.Get<Tensor>();
    TensorShape expected_shape(expected_dims);
    ASSERT_EQ(expected_shape, tensor.Shape());

    const std::vector<T> output(tensor.template Data<T>(), tensor.template Data<T>() + expected_values.size());
    ASSERT_EQ(output.size(), expected_values.size());
    ASSERT_EQ(0, std::memcmp(expected_values.data(), output.data(), expected_values.size()));
}

template <typename T>
void verify_results(const std::vector<OrtValue>& fetches, const ShapeValuesPairsVector<T> &expected_results)
{
    ASSERT_EQ(fetches.size(), expected_results.size());
    for (size_t i = 0; i < fetches.size(); i++) {
        verify_single_result(fetches[i], expected_results[i].first, expected_results[i].second);
    }
}

template <typename T>
void run_session_and_verify_results(InferenceSessionWrapper &session_object, AllocatorPtr &cpu_allocator,
    std::vector<int64_t> &input_shape, std::vector<T> &input_dataset, const std::string &input_name,
    const std::vector<std::string> &output_names, const ShapeValuesPairsVector<T> &expected_results)
{
    OrtValue in_ort_val;
    CreateMLValue<T>(cpu_allocator, input_shape, input_dataset, &in_ort_val);

    NameMLValMap feeds;
    feeds.insert(std::make_pair(input_name, in_ort_val));
    
    std::vector<OrtValue> fetches;
    auto status = session_object.Run(feeds, output_names, &fetches);
    ASSERT_TRUE(status.IsOK());
    
    verify_results(fetches, expected_results);
}

template <typename T>
void run_hailo_test(const std::string &model_path, const std::string &session_logid,
    std::vector<int64_t> &input_shape, std::vector<T> &input_dataset, const std::string &input_name,
    const std::vector<std::string> &output_names, const ShapeValuesPairsVector<T> &expected_results)
{
    SessionOptions session_options;
    session_options.session_logid = session_logid;
    InferenceSessionWrapper session_object{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session_object.Load(model_path));

    auto hailo_provider = DefaultHailoExecutionProvider();
    auto cpu_allocator = hailo_provider->GetAllocator(DEFAULT_DEVICE_ID, OrtMemTypeCPU);
    ASSERT_STREQ(cpu_allocator->Info().name, "HailoCpu");
    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(hailo_provider)));

    ASSERT_STATUS_OK(session_object.Initialize());
    run_session_and_verify_results(session_object, cpu_allocator, input_shape, input_dataset, input_name,
        output_names, expected_results);
}

/*
    This is a sanity test, containing pre-process op, HailoOp and post-process op.
    
    Model structure:
    Mul (multiply by 1) -> HailoOp (shortcut) -> Mul (multiply by 1)
    ORT will insert MemcpyToHost op before HailoOp and MemcpyFromHost op after it.

    Single input ("x.1").
    Single output ("28").
*/
TEST(HailoCustomOpTest, hailo_custom_op_sanity)
{
    auto shortcut_model_data = TestDataShortcut();
    const std::string session_logid = "HailoCustomOpTest.hailo_custom_op_sanity";
    run_hailo_test(shortcut_model_data.m_model_path, session_logid, shortcut_model_data.m_input_shape, shortcut_model_data.m_input_dataset,
        shortcut_model_data.m_input_name, shortcut_model_data.m_output_names, shortcut_model_data.m_expected_results);
}

/*
    This is a sanity test for multiple outputs, the tested model contains only HailoOp.

    Single input ("input0").
    Two outputs ("output0", "output1").
*/
TEST(HailoCustomOpTest, multipule_outputs)
{
    auto multple_outputs_model_data = TestDataMultipleOutputs();
    const std::string session_logid = "HailoCustomOpTest.multipule_outputs";
    run_hailo_test<uint8_t>(multple_outputs_model_data.m_model_path, session_logid, multple_outputs_model_data.m_input_shape,
        multple_outputs_model_data.m_input_dataset, multple_outputs_model_data.m_input_name,
        multple_outputs_model_data.m_output_names, multple_outputs_model_data.m_expected_results);
}

TEST(HailoCustomOpTest, multipule_outputs_dynamic_frames_count)
{
    const uint32_t frames_count = 5;
    auto multple_outputs_model_data = TestDataMultipleOutputs(frames_count);
    const std::string session_logid = "HailoCustomOpTest.multipule_outputs_dynamic_frames_count";
    run_hailo_test<uint8_t>(multple_outputs_model_data.m_model_path, session_logid, multple_outputs_model_data.m_input_shape,
        multple_outputs_model_data.m_input_dataset, multple_outputs_model_data.m_input_name,
        multple_outputs_model_data.m_output_names, multple_outputs_model_data.m_expected_results);
}

TEST(HailoCustomOpTest, multipule_sessions)
{
    auto shortcut_model_data = TestDataShortcut();
    SessionOptions so1;
    InferenceSessionWrapper session_object1{so1, GetEnvironment()};
    auto hailo_provider1 = DefaultHailoExecutionProvider();
    auto cpu_allocator1 = hailo_provider1->GetAllocator(DEFAULT_DEVICE_ID, OrtMemTypeCPU);
    ASSERT_STATUS_OK(session_object1.RegisterExecutionProvider(std::move(hailo_provider1)));
    ASSERT_STATUS_OK(session_object1.Load(shortcut_model_data.m_model_path));
    ASSERT_STATUS_OK(session_object1.Initialize());

    run_session_and_verify_results(session_object1, cpu_allocator1, shortcut_model_data.m_input_shape,
        shortcut_model_data.m_input_dataset, shortcut_model_data.m_input_name, shortcut_model_data.m_output_names,
        shortcut_model_data.m_expected_results);

    auto multple_outputs_model_data = TestDataMultipleOutputs();
    SessionOptions so2;
    InferenceSessionWrapper session_object2{so2, GetEnvironment()};
    auto hailo_provider2 = DefaultHailoExecutionProvider();
    auto cpu_allocator2 = hailo_provider2->GetAllocator(DEFAULT_DEVICE_ID, OrtMemTypeCPU);
    ASSERT_STATUS_OK(session_object2.RegisterExecutionProvider(std::move(hailo_provider2)));
    ASSERT_STATUS_OK(session_object2.Load(multple_outputs_model_data.m_model_path));
    ASSERT_STATUS_OK(session_object2.Initialize());

    run_session_and_verify_results(session_object2, cpu_allocator2, multple_outputs_model_data.m_input_shape,
        multple_outputs_model_data.m_input_dataset, multple_outputs_model_data.m_input_name, multple_outputs_model_data.m_output_names,
        multple_outputs_model_data.m_expected_results);

    run_session_and_verify_results(session_object1, cpu_allocator1, shortcut_model_data.m_input_shape,
        shortcut_model_data.m_input_dataset, shortcut_model_data.m_input_name, shortcut_model_data.m_output_names,
        shortcut_model_data.m_expected_results);
}

/*
    Test support for working with one session and running inference from multiple threads with different inputs.
*/
TEST(HailoCustomOpTest, one_session_multiple_threads)
{
    SessionOptions so;
    InferenceSessionWrapper session_object{so, GetEnvironment()};
    auto hailo_provider = DefaultHailoExecutionProvider();
    auto cpu_allocator = hailo_provider->GetAllocator(DEFAULT_DEVICE_ID, OrtMemTypeCPU);
    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(hailo_provider)));
    ASSERT_STATUS_OK(session_object.Load(ONNX_SHORTCUT_FILE_PATH));
    ASSERT_STATUS_OK(session_object.Initialize());

    const size_t threads_count = 20;
    std::vector<TestDataShortcut> datasets;
    datasets.reserve(threads_count);
    for (size_t i = 0; i < threads_count; i++) {
        datasets.emplace_back(TestDataShortcut(i));
    }
    std::vector<std::thread> threads;
    threads.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i) {
        threads.emplace_back(std::thread(run_session_and_verify_results<float32_t>, std::ref(session_object), std::ref(cpu_allocator),
            std::ref(datasets[i].m_input_shape), std::ref(datasets[i].m_input_dataset), std::ref(datasets[i].m_input_name),
            std::ref(datasets[i].m_output_names), std::ref(datasets[i].m_expected_results)));
    }
    for (auto& th : threads) {
        th.join();
    }
}

TEST(HailoCustomOpTest, one_session_multiple_threads_stress)
{
    for (size_t i = 0; i < DEFAULT_TEST_REPETITIONS; i++) {
        SessionOptions so;
        InferenceSessionWrapper session_object{so, GetEnvironment()};
        auto hailo_provider = DefaultHailoExecutionProvider();
        auto cpu_allocator = hailo_provider->GetAllocator(DEFAULT_DEVICE_ID, OrtMemTypeCPU);
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(hailo_provider)));
        ASSERT_STATUS_OK(session_object.Load(ONNX_SHORTCUT_FILE_PATH));
        ASSERT_STATUS_OK(session_object.Initialize());

        const size_t threads_count = 20;
        std::vector<TestDataShortcut> datasets;
        datasets.reserve(threads_count);
        for (size_t i = 0; i < threads_count; i++) {
            datasets.emplace_back(TestDataShortcut(i));
        }
        std::vector<std::thread> threads;
        threads.reserve(threads_count);
        for (size_t i = 0; i < threads_count; ++i) {
            threads.emplace_back(std::thread(run_session_and_verify_results<float32_t>, std::ref(session_object), std::ref(cpu_allocator),
                std::ref(datasets[i].m_input_shape), std::ref(datasets[i].m_input_dataset), std::ref(datasets[i].m_input_name),
                std::ref(datasets[i].m_output_names), std::ref(datasets[i].m_expected_results)));
        }
        for (auto& th : threads) {
            th.join();
        }
    }
}

TEST(HailoCustomOpTest, hailo_provider_in_ort_api)
{
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(11);
    OrtSessionOptions* soptions;
    auto so_status = ort_api->CreateSessionOptions(&soptions);
    ASSERT_TRUE(so_status == nullptr); // nullptr for Status* indicates success (As written in onnxruntime_c_api.h)

    auto provider_status = ort_api->SessionOptionsAppendExecutionProvider_Hailo(soptions, 1);
    ASSERT_TRUE(provider_status == nullptr); // nullptr for Status* indicates success (As written in onnxruntime_c_api.h)
}

}  // namespace test
}  // namespace onnxruntime

