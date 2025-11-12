// EpValidationTool.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#ifdef USE_WINML_FEATURES
#include <windows.h>
#include <MddBootstrap.h>
#include <WindowsAppSDK-VersionInfo.h>
#include <windows.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/base.h>

#endif
#include "accuracy_validation.h"
#include "config.h"
#include "dataset_reader.h"
#include "numpy_store.h"
#include "performance_result.h"
#include "performance_runner.h"
#include "result_reporter.h"
#include <filesystem>
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
int wmain(int argc, const wchar_t* argv[])
{
    Config config;

    if (!config.FromCommandLine(argc, argv))
    {
        std::cerr << "ERROR: Failed to parse command line arguments." << std::endl;
        return -1;
    }

    std::unique_ptr<IDatasetReader> dataset_reader;
    if (!config.dataset_dir.empty())
    {
        dataset_reader = CreateDatasetReader(config.dataset_dir);
        if (!dataset_reader)
        {
            std::cerr << "ERROR: Failed to create dataset reader." << std::endl;
            return -1;
        }
    }
#ifdef USE_WINML_FEATURES
    if (config.runtimeInfo && config.runtimeInfo.has_value())
    {
        std::cout << "Runtime version is passed from cmd." << std::endl;
        std::cout << "UINT32 representation of Major Minor Version : " << config.runtimeInfo->majorMinorVersion
                  << std::endl;
        std::cout << "UINT64 representation of Package Version : " << config.runtimeInfo->packageVersion.Version
                  << std::endl;
        std::wcout << "Version Tag : " << config.runtimeInfo->versionTag << std::endl;
        MddBootstrapInitialize2(
            config.runtimeInfo->majorMinorVersion,
            PCWSTR{config.runtimeInfo->versionTag.c_str()},
            config.runtimeInfo->packageVersion,
            MddBootstrapInitializeOptions_OnNoMatch_ShowUI);
    }
    else
    {
        std::cout << "Runtime version not specified . Using the one with which exe is build." << std::endl;
        PACKAGE_VERSION version{};
        version.Version = WINDOWSAPPSDK_RUNTIME_VERSION_UINT64;
        MddBootstrapInitialize2(
            WINDOWSAPPSDK_RELEASE_MAJORMINOR,    // points to 0x00010008 (1.8) right now
            WINDOWSAPPSDK_RELEASE_VERSION_TAG_W, // e.g., L"",L"stable", L"preview", or L"experimental") that indicates
                                                 // the release channel. By default it is empty string.
            version,
            MddBootstrapInitializeOptions_OnNoMatch_ShowUI);
    }
    winrt::init_apartment();
    Ort::InitApi();
    if (config.shouldCompile && config.compiledModelPath.empty())
    {
        auto model_path = config.model_path;
        auto pos = model_path.rfind(L".");
        if (pos != std::wstring::npos)
        {
            config.compiledModelPath = std::wstring(model_path.substr(0, pos)) + L"_ctx.onnx";
        }
        else
        {
            config.compiledModelPath = model_path + L"_ctx.onnx";
        }
    }
#endif
    // Memory optimization: Load FP32 data more efficiently
    std::unordered_map<std::string, std::vector<Ort::Value>> fp32_cpu_outputs;
    std::unordered_map<std::string, std::string> output_map;
    if (dataset_reader)
    {
        output_map = dataset_reader->GetOutputNames();
    }
    size_t num_samples = dataset_reader ? dataset_reader->GetNumSamples() : 100;
    size_t output_loaded = 0;

    // Memory optimization: Only load FP32 outputs when needed for specific stages
    if (config.stage == ExecutionStage::DATAGEN || config.stage >= ExecutionStage::ACCURACY)
    {
        if (!output_map.empty())
        {
            for (const auto& [name, value] : output_map)
            {
                fp32_cpu_outputs[name].clear();
                fp32_cpu_outputs[name].reserve(num_samples);
                for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx)
                {
                    Ort::Value fp32_tensor(nullptr);
                    if (!dataset_reader->LoadOutput(sample_idx, name, fp32_tensor))
                    {
                        std::cout << "WARNING: Failed to load fp32 CPU output tensors for output: " << name << std::endl;
                        break;
                    }
                    fp32_cpu_outputs[name].emplace_back(std::move(fp32_tensor));
                    output_loaded++;
                }
            }
        }
        else
        {
            std::cout << "No output map available from dataset reader." << std::endl;
        }
    }

    if (config.stage == ExecutionStage::DATAGEN)
    {
        std::cout << "----- Generating dataset" << std::endl;
        std::unordered_map<std::string, std::string> empty_options;

        // Memory optimization: Use scope-based memory management
        {
            std::wstring ref_model_wpath(config.ref_model_path.begin(), config.ref_model_path.end());
            PerformanceRunner ref_performance_runner(
                ref_model_wpath,
                "",
                dataset_reader
#ifdef USE_WINML_FEATURES
                ,
                L"",
                false
#endif
            );
            if (!output_loaded)
            {
                std::cout << "No FP32 CPU outputs found, using reference model to generate dataset." << std::endl;
                if (!config.ref_model_path.empty())
                {
                    std::cout << "\n----- Initializing CPU session with reference model" << std::endl;
                    const auto& [ref_init_success, ref_init_result] = ref_performance_runner.InitializeSession(
#ifndef USE_WINML_FEATURES
                        config.execution_provider,
#endif
                        empty_options,
                        empty_options);
                    if (!ref_init_success)
                    {
                        std::cerr << "ERROR: CPU (reference) session initialization failed." << std::endl;
                        return -1;
                    }

                    std::cout << "----- Running inferences on CPU with reference model" << std::endl;
                    NumpyTensorsStore fp32_outputs_store(config.out_dir, output_map, "fp32_cpu");
                    const auto& [ref_run_success, ref_perf] = ref_performance_runner.RunInferences(fp32_outputs_store);
                    if (!ref_run_success)
                    {
                        std::cerr << "ERROR: CPU (reference) inference failed." << std::endl;
                        return -1;
                    }
                }
                else
                {
                    std::cerr << "ERROR: No reference model provided for dataset generation." << std::endl;
                    return -1;
                }
            }
            else
            {
                std::cout << "Using existing FP32 CPU outputs for dataset generation." << std::endl;
            }
        } // ref_performance_runner destroyed here (memory optimization)

        // Memory optimization: Second scope for QDQ model processing  
        {
            std::cout << "\n----- Initializing CPU session with QDQ model" << std::endl;
            std::wstring model_wpath(config.model_path.begin(), config.model_path.end());
            PerformanceRunner cpu_performance_runner(
                model_wpath,
                config.model_key,
                dataset_reader
#ifdef USE_WINML_FEATURES
                ,
                L"",
                false
#endif
            );
            const auto& [cpu_init_success, cpu_init_result] = cpu_performance_runner.InitializeSession(
#ifndef USE_WINML_FEATURES
                config.execution_provider,
#endif
                empty_options,
                empty_options);
            if (!cpu_init_success)
            {
                std::cerr << "ERROR: CPU (QDQ) session initialization failed." << std::endl;
                return -1;
            }

            std::cout << "----- Running inferences on CPU with QDQ model" << std::endl;
            NumpyTensorsStore cpu_outputs_store(config.out_dir, output_map, "qdq_cpu");
            const auto& [cpu_run_success, cpu_perf] = cpu_performance_runner.RunInferences(cpu_outputs_store);
            if (!cpu_run_success)
            {
                std::cerr << "ERROR: CPU (QDQ) inference failed." << std::endl;
                return -1;
            }
            const auto& qdq_cpu_outputs = cpu_performance_runner.GetOutputs();

            std::unordered_map<std::string, std::vector<float>> l2norm_map;
            if (output_loaded)
            {
                l2norm_map = CalculateCPUL2Norm(qdq_cpu_outputs, fp32_cpu_outputs);
            }
            else 
            {
                // Note: ref_performance_runner is out of scope, so we use the loaded fp32_cpu_outputs
                // This is a limitation but maintains memory efficiency
                std::cerr << "ERROR: Reference model outputs not available in this scope for L2 norm calculation." << std::endl;
                return -1;
            }
            if (l2norm_map.empty())
            {
                std::cerr << "ERROR: Failed to calculate the L2 norm values." << std::endl;
                return -1;
            }

            if (!dataset_reader->StoreL2Norm(l2norm_map))
            {
                std::cerr << "ERROR: Failed to store L2 norm values." << std::endl;
                return -1;
            }
        } // cpu_performance_runner destroyed here (memory optimization)

        std::cout << "----- Dataset generation completed successfully." << std::endl;
    }
    if (config.stage >= ExecutionStage::COMPILATION)
    {
        // Memory optimization: Scope-based memory management for NPU processing
        {
            std::cout << "----- Initializing NPU session" << std::endl;
            std::wstring model_wpath(config.model_path.begin(), config.model_path.end());
            PerformanceRunner performance_runner(
                model_wpath,
                config.model_key,
                dataset_reader
#ifdef USE_WINML_FEATURES
                ,
                config.compiledModelPath,
                config.shouldCompile
#endif
            );

            auto [is_compiling, compilation_result] = performance_runner.InitializeSession(
#ifndef USE_WINML_FEATURES
                config.execution_provider,
#endif
                config.ep_options,
                config.session_options,
                config.graph_opt_level
#ifdef USE_WINML_FEATURES
                ,
                config.ep_policy,
                config.ep_info
#endif
            );
            if (!is_compiling)
            {
                std::cerr << "ERROR: NPU session initialization failed." << std::endl;
                return -1;
            }

            PerformanceResult performance_result;
            std::unordered_map<std::string, std::vector<float>> accuracy_result;

            if (config.stage >= ExecutionStage::INFERENCE)
            {
                NumpyTensorsStore npu_outputs_store(config.out_dir, "npu");
                if (!output_map.empty())
                {
                    npu_outputs_store = NumpyTensorsStore(config.out_dir, output_map, "npu");
                }
                std::cout << "----- Running inferences on NPU" << std::endl;

                // Memory optimization: Only store outputs in memory if ACCURACY stage will run
                bool need_outputs_in_memory = (config.stage >= ExecutionStage::ACCURACY);

                const auto& [success, perf_result] = need_outputs_in_memory ? 
                    performance_runner.RunInferences(npu_outputs_store) :
                    performance_runner.RunInferencesStreamOnly(npu_outputs_store);
                performance_result = perf_result;
                if (!success)
                {
                    std::cerr << "ERROR: NPU inference failed." << std::endl;
                    return -1;
                }

                // Only get outputs if needed for accuracy stage
                const auto* npu_outputs_ptr = need_outputs_in_memory ? 
                    &performance_runner.GetOutputs() : nullptr;
                if (config.perf_threshold != 0.0f &&
                    !performance_runner.IsPerformant(config.perf_threshold, performance_result))
                {
                    std::cerr << "ERROR: Performance threshold not met." << std::endl;
                    return -1;
                }

                // Memory optimization: Clean up if not proceeding to accuracy stage
                if (config.stage < ExecutionStage::ACCURACY && !need_outputs_in_memory)
                {
                    performance_runner.ClearOutputs();
                    std::cout << "----- NPU inference completed (streaming mode)." << std::endl;
                }

                if (config.stage >= ExecutionStage::ACCURACY)
                {
                    if (!npu_outputs_ptr)
                    {
                        std::cerr << "ERROR: NPU outputs not available for accuracy checking." << std::endl;
                        return -1;
                    }
                    const auto& npu_outputs = *npu_outputs_ptr;
                    std::cout << "----- Checking accuracy" << std::endl;
                    std::cout << "Loading L2 norm values and fp32 model outputs." << std::endl;

                    std::unordered_map<std::string, std::vector<float>> l2_norm_outputs;
                    if (!output_map.empty())
                    {
                        for (const auto& [name, value] : output_map)
                        {
                            // Load l2_norm values
                            l2_norm_outputs[name].clear();
                            if (!dataset_reader->LoadL2Norm(name, l2_norm_outputs[name]))
                            {
                                std::cerr << "ERROR: Failed to load l2_norm values for output: " << name << std::endl;
                                return -1;
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "ERROR: No output map available from dataset reader, skipping accuracy validation." << std::endl;
                        return -1;
                    }

                    if (!output_loaded)
                    {
                        std::cerr << "ERROR: No FP32 CPU outputs found for accuracy validation." << std::endl;
                        return -1;
                    }

                    auto [accuracy_result_map, is_accurate] =
                        CheckAccuracy(config.acc_threshold, npu_outputs, fp32_cpu_outputs, l2_norm_outputs, output_map);
                    accuracy_result = accuracy_result_map;
                    if (!is_accurate)
                    {
                        std::cerr << "ERROR: Accuracy check failed." << std::endl;
                        return -1;
                    }
                }
            }

            ResultReporter result_reporter(config, compilation_result, performance_result, accuracy_result);

            result_reporter.PrintToConsole();
            if (!config.result_path.empty())
            {
                std::cout << "Going to generate dump" << std::endl;
                result_reporter.DumpToJson(config.result_path);
            }
        } // performance_runner destroyed here (memory optimization)

    }
    // Release the DDLM and clean up.
#ifdef USE_WINML_FEATURES
    MddBootstrapShutdown();
#endif
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files
//   to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
