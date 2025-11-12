#pragma once

#include "compilation_result.h"
#include "config.h"
#include "dataset_reader.h"
#include "io_binding.h"
#include "tensors_reader_writer.h"
#include "performance_result.h"
#include "profiling_utils.h"
#include "onnxruntime_cxx_api.h"

#include <utility>

constexpr int DEFAULT_GRAPH_OPTIM_LEVEL = -1;
#ifdef USE_WINML_FEATURES
const OrtExecutionProviderDevicePolicy DEFAULT_EP_POLICY = OrtExecutionProviderDevicePolicy_DEFAULT;
#endif

/**
 * @class PerformanceRunner
 * @brief Class for running performance tests using ONNX Runtime.
 *
 * This class was inspired by the implementation in the following repository:
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/performance_runner.cc
 */
class PerformanceRunner {
public:
    bool IsPerformant(float perf_threshold, const PerformanceResult& performance_result);

#ifdef USE_WINML_FEATURES

    void ConfigureExecutionProviders(
        Ort::SessionOptions& session_options,
        std::unordered_map<std::string, std::string>& ep_options,
        Ort::Env& env,
        std::optional<EpInfo>& ep_info);
    OrtStatus* CompileModel(
        const OrtApi& ortApi,
        OrtSessionOptions* sessionOptions,
        const std::filesystem::path& modelPath,
        const std::filesystem::path& compiledModelPath);
    std::string ToString(OrtExecutionProviderDevicePolicy policy);
    std::string ToString(OrtHardwareDeviceType policy);
    void ConvertToOrtKeyValuePairs(
        const std::unordered_map<std::string, std::string>& ep_options_map, Ort::KeyValuePairs& ep_options);

#endif

    PerformanceRunner(
        std::wstring model_path,
        std::string model_key,
        std::unique_ptr<IDatasetReader>& dataset_reader
#ifdef USE_WINML_FEATURES
        ,
        std::wstring compiled_model_path,
        bool should_compile_model
#endif
    );

    std::pair<bool, CompilationResult> InitializeSession(
#ifndef USE_WINML_FEATURES
        std::string execution_provider,
#endif
        std::unordered_map<std::string, std::string>& ep_options,
        std::unordered_map<std::string, std::string>& session_options,
        int graph_opt_level = DEFAULT_GRAPH_OPTIM_LEVEL
#ifdef USE_WINML_FEATURES
        ,
        std::optional<OrtExecutionProviderDevicePolicy> ep_policy = DEFAULT_EP_POLICY,
        std::optional<EpInfo> ep_info = std::nullopt
#endif
    );

    std::pair<bool, PerformanceResult> RunInferences(ITensorsWriter& outputs_writer);

    std::pair<bool, PerformanceResult> RunInferencesStreamOnly(ITensorsWriter& outputs_writer);

    void ClearOutputs();

    const std::unordered_map<std::string, std::vector<Ort::Value>>& GetOutputs() const;

    const std::vector<std::string>& GetOutputNames() const;

private:
    void CreateIoBinding();
    void GenerateRandomSample(unsigned int seed, Ort::Value& tensor);
    void SaveSampleOutputs(const std::vector<Ort::Value>& outputs);
    void ClearOutputContents();

    std::wstring m_model_path;
    std::string m_model_key;
#ifdef USE_WINML_FEATURES
    std::wstring m_compiled_model_path;
    bool m_should_compile_model;
#endif

    std::unique_ptr<IDatasetReader>& m_dataset_reader;
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::RunOptions m_run_options;
    Ort::AllocatorWithDefaultOptions m_allocator;

    std::vector<std::string> m_input_names;
    std::vector<const char*> m_input_name_ptrs;
    std::vector<Ort::Value> m_input_tensors;
    std::vector<std::string> m_output_names;
    std::vector<const char*> m_output_name_ptrs;
    std::vector<Ort::Value> m_output_tensors;

    std::unordered_map<std::string, std::vector<Ort::Value>> m_dataset_outputs;
};
