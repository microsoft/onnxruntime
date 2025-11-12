#include "performance_runner.h"
#include "crypto.h"
#include "encryption_key_manager.h"
#include "model_utils.h"
#include "profiling_utils.h"

#include <chrono>
#include <random>
#ifdef USE_WINML_FEATURES
#include "config.h"
#include <windows.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/base.h>

using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Foundation;
#endif
PerformanceRunner::PerformanceRunner(
    std::wstring model_path,
    std::string model_key,
    std::unique_ptr<IDatasetReader>& dataset_reader
#ifdef USE_WINML_FEATURES
    ,
    std::wstring compiled_model_path,
    bool should_compile_model
#endif
    )
    : m_model_path(model_path),
      m_model_key(model_key),
      m_dataset_reader(dataset_reader),
      m_env(ORT_LOGGING_LEVEL_WARNING, "ep_validtion_tool_winml"),
      m_session(nullptr),
      m_run_options()
#ifdef USE_WINML_FEATURES
      ,
      m_compiled_model_path(compiled_model_path),
      m_should_compile_model(should_compile_model)
#endif
{
}
#ifdef USE_WINML_FEATURES
std::string PerformanceRunner::ToString(OrtExecutionProviderDevicePolicy policy)
{
    switch (policy)
    {
    case OrtExecutionProviderDevicePolicy_PREFER_NPU:
        return "PREFER_NPU";
    case OrtExecutionProviderDevicePolicy_PREFER_CPU:
        return "PREFER_CPU";
    case OrtExecutionProviderDevicePolicy_PREFER_GPU:
        return "PREFER_GPU";
    case OrtExecutionProviderDevicePolicy_DEFAULT:
        return "DEFAULT";
    default:
        return "UNKNOWN";
    }
}

std::string PerformanceRunner::ToString(OrtHardwareDeviceType policy)
{
    switch (policy)
    {
    case OrtHardwareDeviceType_NPU:
        return "NPU";
    case OrtHardwareDeviceType_CPU:
        return "CPU";
    case OrtHardwareDeviceType_GPU:
        return "GPU";
    default:
        return "";
    }
}

void PerformanceRunner::ConvertToOrtKeyValuePairs(
    const std::unordered_map<std::string, std::string>& ep_options_map, Ort::KeyValuePairs& ep_options)
{
    for (const auto& [key, value] : ep_options_map)
    {
        ep_options.Add(key.c_str(), value.c_str());
    }
}

/**
 * @brief Compile an ONNX model using the OrtCompileApi
 *
 * This function demonstrates how to:
 * 1. Get the compile API
 * 2. Create compilation options from session options
 * 3. Configure input and output paths
 * 4. Compile the model
 *
 * @param ortApi ORT API instance
 * @param env ORT environment
 * @param sessionOptions Session options to use for compilation
 * @param modelPath Path to the input model
 * @param compiledModelPath Path where the compiled model will be saved
 * @return OrtStatus* Status of the compilation, nullptr if successful
 */
OrtStatus* PerformanceRunner::CompileModel(
    const OrtApi& ortApi,
    OrtSessionOptions* sessionOptions,
    const std::filesystem::path& modelPath,
    const std::filesystem::path& compiledModelPath)
{
    std::cout << "Compiling model from " << modelPath << std::endl;
    std::cout << "Output path: " << compiledModelPath << std::endl;

    // Get compile API
    const OrtCompileApi* compileApi = ortApi.GetCompileApi();
    if (!compileApi)
    {
        std::cerr << "Failed to get compile API" << std::endl;
        return nullptr;
    }

    // Create compilation options from session options
    OrtModelCompilationOptions* compileOptions = nullptr;
    OrtStatus* status =
        compileApi->CreateModelCompilationOptionsFromSessionOptions(m_env, sessionOptions, &compileOptions);
    if (status != nullptr)
    {
        std::cerr << "Failed to create compilation options: " << ortApi.GetErrorMessage(status) << std::endl;
        return status;
    }

    // Set input and output model paths
    status = compileApi->ModelCompilationOptions_SetInputModelPath(compileOptions, modelPath.c_str());
    if (status != nullptr)
    {
        std::cerr << "Failed to set input model path: " << ortApi.GetErrorMessage(status) << std::endl;
        compileApi->ReleaseModelCompilationOptions(compileOptions);
        return status;
    }

    status = compileApi->ModelCompilationOptions_SetOutputModelPath(compileOptions, compiledModelPath.c_str());
    if (status != nullptr)
    {
        std::cerr << "Failed to set output model path: " << ortApi.GetErrorMessage(status) << std::endl;
        compileApi->ReleaseModelCompilationOptions(compileOptions);
        return status;
    }

    // Measure compile time
    std::cout << "Starting compile, this may take a few moments..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Compile the model
    status = compileApi->CompileModel(m_env, compileOptions);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    if (status == nullptr)
    {
        std::cout << "Model compiled successfully in " << duration.count() << " ms" << std::endl;
    }
    else
    {
        std::cerr << "Failed to compile model: " << ortApi.GetErrorMessage(status) << std::endl;
    }

    compileApi->ReleaseModelCompilationOptions(compileOptions);
    return status;
}
void PerformanceRunner::ConfigureExecutionProviders(
    Ort::SessionOptions& session_options,
    std::unordered_map<std::string, std::string>& ep_options,
    Ort::Env& m_env,
    std::optional<EpInfo>& ep_info)
{
    try
    {
        std::cout << "ONNX Version string: " << Ort::GetVersionString() << std::endl;
        std::cout << "Getting available providers..." << std::endl;
        auto catalog = winrt::Microsoft::Windows::AI::MachineLearning::ExecutionProviderCatalog::GetDefault();
        auto providers = catalog.FindAllProviders();

        // 3 Ep's are allowlisted -OpenVino, Qnn, Vitis . If any other EP is provided while running exe through command
        // params, add that to the existing set .
        if (ep_info.has_value() && !ep_info.value().name.empty())
        {
            std::cout << "Adding " << ep_info.value().name << " to default EP list ." << std::endl;
            g_executionProviders.insert(ep_info.value().name);
        }
        for (const auto& provider : providers)
        {
            std::string providerName = winrt::to_string(provider.Name());
            // Add a check to install only whitelisted EP's list
            std::cout << "Provider: " << providerName << std::endl;
            if (g_executionProviders.find(providerName) != g_executionProviders.end())
            {
                std::cout << providerName << " is a valid supported execution provider in EP Tool." << std::endl;
                // Download required components for this provider (if not already present)
                auto result = provider.EnsureReadyAsync().get();
                if (result &&
                    result.Status() ==
                        winrt::Microsoft::Windows::AI::MachineLearning::ExecutionProviderReadyResultState::Success)
                {
                    // Register the provider with ONNX Runtime
                    bool registered = provider.TryRegister();
                    std::wcout << "  Registration status: " << (registered ? "SUCCEESS" : "FAILED") << std::endl;
                }
            }
        }
        // Get all available EP devices from the environment
        std::vector<Ort::ConstEpDevice> ep_devices = m_env.GetEpDevices();

        std::cout << "Discovered " << ep_devices.size() << " execution provider device(s)" << std::endl;

        // Accumulate devices by ep_name
        // Passing all devices for a given EP in a single call allows the execution provider
        // to select the best configuration or combination of devices, rather than being limited
        // to a single device. This enables optimal use of available hardware if supported by the EP.
        std::unordered_map<std::string, std::vector<Ort::ConstEpDevice>> ep_device_map;
        for (const auto& device : ep_devices)
        {
            ep_device_map[device.EpName()].push_back(device);
        }

        std::vector<Ort::ConstEpDevice> filteredDevices;
        if (ep_info.has_value())
        {
            const auto& info = ep_info.value();
            for (const auto& device : ep_devices)
            {
                bool match = true;

                // Match EP name
                if (info.name.empty() || device.EpName() != info.name)
                {
                    continue;
                }

                // Match Device Type
                if (info.deviceType.has_value())
                {
                    if (ToString(device.Device().Type()) != info.deviceType.value())
                    {
                        continue;
                    }
                }

                // Match Device Vendor ID
                if (info.deviceVendorId.has_value())
                {
                    if (device.Device().VendorId() != info.deviceVendorId.value())
                    {
                        continue;
                    }
                }

                // Match Device ID
                if (info.deviceId.has_value())
                {
                    if (device.Device().DeviceId() != info.deviceId.value())
                    {
                        continue;
                    }
                }

                filteredDevices.push_back(device);
            }
        }
        else
        {
            for (const auto& [ep_name, devices] : ep_device_map)
            {
                if (g_executionProviders.find(ep_name) != g_executionProviders.end())
                {
                    for (const auto& device : devices)
                    {
                        std::string deviceType = ToString(device.Device().Type());
                        if (deviceType == "NPU")
                        {
                            filteredDevices.push_back(device);
                        }
                    }
                    std::cout << "Successfully added " << ep_name << " EP" << std::endl;
                }

                else
                {
                    std::cout << "Skipping EP: " << ep_name << std::endl;
                }
            }
        }
        std::wcout << L" Filtered Execution Provider Information:\n";
        std::wcout
            << L"---------------------------------------------------------------------------------------------\n";
        std::wcout << std::left << std::setw(30) << L"Provider" << std::setw(20) << L"EP Vendor" << std::setw(20)
                   << L"EP Metadata" << std::setw(20) << L"Device Type" << std::setw(20) << L"Device ID"
                   << std::setw(20) << L"Device Vendor Id" << std::setw(20) << L"Device Vendor" << std::setw(20)
                   << L"Device Metadata" << std::endl;
        std::wcout
            << L"---------------------------------------------------------------------------------------------\n";

        for (const auto& device : filteredDevices)
        {
            std::cout << std::left << std::setw(30) << device.EpName();

            std::cout << std::setw(20) << device.EpVendor() << std::setw(20) << device.EpMetadata() << std::setw(20)
                      << ToString(device.Device().Type()) << std::setw(20) << device.Device().DeviceId()
                      << std::setw(20) << device.Device().VendorId() << std::setw(20) << device.Device().Vendor()
                      << std::setw(20) << device.Device().Metadata() << std::endl;
        }

        std::wcout << L"-------------------------------------------------------------\n";
        Ort::KeyValuePairs epOptionsPairs;
        ConvertToOrtKeyValuePairs(ep_options, epOptionsPairs);
        session_options.AppendExecutionProvider_V2(m_env, filteredDevices, epOptionsPairs);
    }
    catch (const winrt::hresult_error& ex)
    {
        std::cerr << "ERROR: WinRT exception in ConfigureExecutionProviders: " << winrt::to_string(ex.message()).c_str()
                  << " (HRESULT: 0x" << std::hex << ex.code() << ")" << std::endl;
        throw;
    }
    catch (const Ort::Exception& ex)
    {
        std::cerr << "ERROR: ORT exception in ConfigureExecutionProviders: " << ex.what()
                  << " (error code: " << ex.GetOrtErrorCode() << ")" << std::endl;
        throw;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: Standard exception in ConfigureExecutionProviders: " << ex.what() << std::endl;
        throw;
    }
}
#endif

std::pair<bool, CompilationResult> PerformanceRunner::InitializeSession(
#ifndef USE_WINML_FEATURES
    std::string execution_provider,
#endif
    std::unordered_map<std::string, std::string>& ep_options,
    std::unordered_map<std::string, std::string>& session_options,
    int graph_opt_level
#ifdef USE_WINML_FEATURES
    ,
    std::optional<OrtExecutionProviderDevicePolicy> ep_policy,
    std::optional<EpInfo> ep_info
#endif
)
{
    CompilationResult compilation_result;

    if (m_session)
    {
        m_session.release();
    }

    Ort::SessionOptions sess_options;

    // Set onnxruntime_perf_test defaults.
    sess_options.EnableCpuMemArena();
    sess_options.EnableMemPattern();
    sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (graph_opt_level != DEFAULT_GRAPH_OPTIM_LEVEL)
    {
        sess_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(graph_opt_level));
    }

    for (const auto& option : session_options)
    {
        sess_options.AddConfigEntry(option.first.c_str(), option.second.c_str());
    }

    try
    {
        std::filesystem::path actualModelPath;
        actualModelPath = m_model_path;
#ifdef USE_WINML_FEATURES
        if (ep_policy.has_value())
        {
            // Set the EP selection policy based on command line
            std::cout << "Using EP Selection Policy: " << ToString(ep_policy.value()) << std::endl;
            sess_options.SetEpSelectionPolicy(ep_policy.value());
        }
        else
        {
            // Use explicit configuration
            std::cout << "Using explicit EP configuration" << std::endl;
            ConfigureExecutionProviders(sess_options, ep_options, m_env, ep_info);
        }
        // Handle model compilation if needed

        bool isCompiledModelAvailable = std::filesystem::exists(m_compiled_model_path);
        if (isCompiledModelAvailable)
        {
            std::wcout << "Using existing compiled model: " << m_compiled_model_path << std::endl;
            actualModelPath = m_compiled_model_path;
        }
        else if (m_should_compile_model)
        {
            std::cout << "No compiled model found, attempting to create compiled model" << std::endl;

            const OrtApi& ortApi = Ort::GetApi();
            OrtStatus* status = CompileModel(ortApi, sess_options, m_model_path, m_compiled_model_path);

            if (status == nullptr && std::filesystem::exists(m_compiled_model_path))
            {
                std::wcout << "Compiled model created successfully at " << m_compiled_model_path << std::endl;
                actualModelPath = m_compiled_model_path;
            }
            else
            {
                std::wcout << "Falling back to original model: " << m_model_path << std::endl;
                actualModelPath = m_model_path;
                if (status != nullptr)
                {
                    ortApi.ReleaseStatus(status);
                }
            }
        }
        else
        {
            std::wcout << "Using original model: " << m_model_path << std::endl;
            actualModelPath = m_model_path;
        }
#endif
#ifndef USE_WINML_FEATURES
        if (execution_provider == "qnn")
        {
            sess_options.AppendExecutionProvider("QNN", ep_options);
        }
        else if (execution_provider == "ov")
        {
            sess_options.AppendExecutionProvider_OpenVINO_V2(ep_options);
        }
        else if (execution_provider == "vitisai")
        {
            sess_options.AppendExecutionProvider_VitisAI(ep_options);
        }
        else if (execution_provider == "cpu")
        {
            // Session is created with CPU EP by default
        }
        else
        {
            std::cerr << "ERROR: Unsupported execution provider: " << execution_provider << std::endl;
            return {false, compilation_result};
        }
#endif
        auto compilation_start = std::chrono::high_resolution_clock::now();

        if (m_model_key.empty())
        {
            // Unencrypted model loading
            m_session = Ort::Session(m_env, actualModelPath.c_str(), sess_options);
        }
#ifndef USE_WINML_FEATURES
        else
        {
            std::vector<unsigned char> model_data; // Encrypted model data
            auto model_key = EncryptionKeyManager::GetEncryptionKeyFromKeyId(m_model_key);
            if (model_key == nullptr)
            {
                // Key ID is not found, use the key as is
                model_key = m_model_key.c_str();
            }
            LoadEncryptedModelData(wstring_to_string(m_model_path), model_data);
            Crypto::decrypt(model_key, model_data);
            m_session = Ort::Session(m_env, model_data.data(), model_data.size(), sess_options);
        }
#endif
        auto compilation_end = std::chrono::high_resolution_clock::now();

        MemoryUsage memory_usage = GetMemoryUsage();

        compilation_result.compilation_time_cost_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(compilation_end - compilation_start).count();
        compilation_result.peak_workingset_size = memory_usage.peak_working_set_size;
        compilation_result.peak_pagefile_usage = memory_usage.peak_pagefile_usage;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ERROR: Failed to create session: " << e.what() << " error code: " << e.GetOrtErrorCode()
                  << std::endl;
        m_session = Ort::Session(nullptr);
        return {false, compilation_result};
    }

    CreateIoBinding();

    return {true, compilation_result};
}

void PerformanceRunner::CreateIoBinding()
{
    try
    {
        m_input_names = CreateTensorNames(m_session, m_allocator, true);
        std::transform(
            m_input_names.begin(),
            m_input_names.end(),
            std::back_inserter(m_input_name_ptrs),
            [](const auto& name) { return name.c_str(); });

        m_output_names = CreateTensorNames(m_session, m_allocator, false);
        std::transform(
            m_output_names.begin(),
            m_output_names.end(),
            std::back_inserter(m_output_name_ptrs),
            [](const auto& name) { return name.c_str(); });

        m_input_tensors = CreateTensors(m_session, m_allocator, true);
        m_output_tensors = CreateTensors(m_session, m_allocator, false);

        // Validate that input names and tensors have matching sizes
        if (m_input_names.size() != m_input_tensors.size())
        {
            throw std::runtime_error("Mismatch between input names count (" + 
                                   std::to_string(m_input_names.size()) + 
                                   ") and input tensors count (" + 
                                   std::to_string(m_input_tensors.size()) + ")");
        }

        for (const auto& output_name : m_output_names)
        {
            m_dataset_outputs.emplace(output_name, std::vector<Ort::Value>());
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: Exception in CreateIoBinding: " << e.what() << std::endl;
        throw;
    }
}

void PerformanceRunner::GenerateRandomSample(unsigned int seed, Ort::Value& tensor)
{
    try
    {
        // Check if tensor is valid and of type float
        if (!tensor.IsTensor())
        {
            throw std::runtime_error("Provided Ort::Value is not a tensor.");
        }

        const auto& tensor_info = tensor.GetTensorTypeAndShapeInfo();
        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            throw std::runtime_error("Tensor data type is not float.");
        }

        std::vector<int64_t> shape = tensor_info.GetShape();
        size_t total_size = 1;
        for (const auto& dim : shape)
        {
            if (dim <= 0)
            {
                throw std::runtime_error("Invalid tensor shape: dimension must be positive.");
            }

            if (total_size > (std::numeric_limits<size_t>::max)() / static_cast<size_t>(dim))
            {
                throw std::overflow_error("Tensor size overflow detected during shape multiplication.");
            }

            total_size *= static_cast<size_t>(dim);
        }

        // Generate random data
        std::mt19937 gen(seed);
        std::normal_distribution<float> dis(0.0f, 1.0f);

        float* tensor_data = tensor.GetTensorMutableData<float>();
        if (!tensor_data)
        {
            throw std::runtime_error("Failed to access tensor data buffer.");
        }

        for (size_t i = 0; i < total_size; ++i)
        {
            tensor_data[i] = dis(gen);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: Exception in GenerateRandomSample: " << e.what() << std::endl;
        throw;
    }
}

std::pair<bool, PerformanceResult> PerformanceRunner::RunInferences(ITensorsWriter& outputs_writer)
{
    CpuUsageMonitor cpu_monitor;
    try
    {
        cpu_monitor.Start();
    }
    catch (const std::exception& e)
    {
        std::cerr << "WARNING: Failed to start CPU monitor: " << e.what() << std::endl;
        // Continue without CPU monitoring
    }

    PerformanceResult performance_result;

    size_t num_samples = m_dataset_reader ? m_dataset_reader->GetNumSamples() : 100;
    ClearOutputContents();

    for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx)
    {
        if (!m_dataset_reader)
        {
            for (auto& input_tensor : m_input_tensors)
            {
                GenerateRandomSample(42, input_tensor);
            }
        }
        else
        {
            for (size_t input_idx = 0; input_idx < m_input_names.size(); ++input_idx)
            {
                if (!m_dataset_reader->Load(sample_idx, m_input_names[input_idx], m_input_tensors[input_idx]))
                {
                    std::cerr << "Failed to load sample " << sample_idx << ". Skipped." << std::endl;
                    continue;
                }
            }
        }

        try
        {
            auto inference_start = std::chrono::high_resolution_clock::now();
            m_session.Run(
                m_run_options,
                m_input_name_ptrs.data(),
                m_input_tensors.data(),
                m_input_tensors.size(),
                m_output_name_ptrs.data(),
                m_output_tensors.data(),
                m_output_tensors.size());
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();
            performance_result.inference_time_costs_ms.emplace_back(duration);
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "ERROR: Inference failed: " << e.what() << " with error code: " << e.GetOrtErrorCode()
                      << std::endl;
            return {false, {}};
        }
        catch (const std::exception& e)
        {
            std::cerr << "ERROR: Inference failed with exception: " << e.what() << std::endl;
            return {false, {}};
        }

        for (size_t output_idx = 0; output_idx < m_output_names.size(); ++output_idx)
        {
            if (AllNans(m_output_tensors[output_idx]))
            {
                std::cerr << "ERROR: Sample " << sample_idx << " inference produced all NaNs for " << "'"
                          << m_output_names[output_idx] << "'." << std::endl;
                return {false, {}};
            }
        }

        for (const auto& output_tensor : m_output_tensors)
        {
            if (AllNans(output_tensor))
            {
                std::cerr << "ERROR: Sample " << sample_idx << " inference produced all NaNs for output." << std::endl;
                return {false, {}};
            }
        }

        SaveSampleOutputs(m_output_tensors);

        if (m_dataset_reader)
        {
            for (size_t output_idx = 0; output_idx < m_output_names.size(); ++output_idx)
            {
                if (!outputs_writer.Store(sample_idx, m_output_names[output_idx], m_output_tensors[output_idx]))
                {
                    std::cerr << "ERROR: Failed to store outputs for sample " << sample_idx << ", output "
                            << m_output_names[output_idx] << std::endl;
                    continue;
                }
            }
        }
    }

    double avg_cpu_usage = cpu_monitor.GetCurrentValue();
    MemoryUsage memory_usage = GetMemoryUsage();

    performance_result.peak_workingset_size = memory_usage.peak_working_set_size;
    performance_result.peak_pagefile_usage = memory_usage.peak_pagefile_usage;
    performance_result.average_cpu_usage = avg_cpu_usage;

    return {true, performance_result};
}

void PerformanceRunner::SaveSampleOutputs(const std::vector<Ort::Value>& outputs)
{
    for (size_t output_idx = 0; output_idx < m_output_names.size(); ++output_idx)
    {
        const auto& output_name = m_output_names[output_idx];
        auto& output_samples = m_dataset_outputs.at(output_name);

        try
        {
            output_samples.emplace_back(CopyTensor(outputs[output_idx]));
        }
        catch (const std::exception& e)
        {
            std::cerr << "ERROR: Failed to copy tensor for output " << output_idx << ": " << e.what() << std::endl;
            throw;
        }
    }
}

bool PerformanceRunner::IsPerformant(float perf_threshold, const PerformanceResult& performance_result)
{
    if (performance_result.inference_time_costs_ms.empty())
    {
        std::cerr << "ERROR: No inference times recorded." << std::endl;
        return false;
    }

    int64_t total_inference_cost_ms = std::accumulate(
        performance_result.inference_time_costs_ms.begin(),
        performance_result.inference_time_costs_ms.end(),
        int64_t(0));
    auto avg_inference_time = total_inference_cost_ms / performance_result.inference_time_costs_ms.size();

    if (avg_inference_time > static_cast<double>(perf_threshold))
    {
        std::cerr << "ERROR: Performance test failed, average inference time (" << avg_inference_time
                  << "ms) is greater than the threshold (" << perf_threshold << "ms)" << std::endl;
        return false;
    }

    return true;
}

const std::unordered_map<std::string, std::vector<Ort::Value>>& PerformanceRunner::GetOutputs() const
{
    return m_dataset_outputs;
}

const std::vector<std::string>& PerformanceRunner::GetOutputNames() const
{
    return m_output_names;
}

std::pair<bool, PerformanceResult> PerformanceRunner::RunInferencesStreamOnly(ITensorsWriter& outputs_writer)
{
    CpuUsageMonitor cpu_monitor;
    try 
    {
        cpu_monitor.Start();
    }
    catch (const std::exception& e)
    {
        std::cerr << "WARNING: Failed to start CPU monitor: " << e.what() << std::endl;
        // Continue without CPU monitoring
    }
    PerformanceResult performance_result;
    size_t num_samples = m_dataset_reader ? m_dataset_reader->GetNumSamples() : 100;
    
    // Clear any existing outputs to save memory
    ClearOutputContents();
    
    for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx)
    {
        if (!m_dataset_reader)
        {
            for (auto& input_tensor : m_input_tensors)
            {
                GenerateRandomSample(42, input_tensor);
            }
        }
        else
        {
            for (size_t input_idx = 0; input_idx < m_input_names.size(); ++input_idx)
            {
                if (!m_dataset_reader->Load(sample_idx, m_input_names[input_idx], m_input_tensors[input_idx]))
                {
                    std::cerr << "ERROR: Failed to load input sample " << sample_idx << " for input: " << m_input_names[input_idx] << std::endl;
                    continue;
                }
            }
        }
        try {
            auto inference_start = std::chrono::high_resolution_clock::now();
            m_session.Run(
                m_run_options,
                m_input_name_ptrs.data(),
                m_input_tensors.data(),
                m_input_tensors.size(),
                m_output_name_ptrs.data(),
                m_output_tensors.data(),
                m_output_tensors.size());
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();
            performance_result.inference_time_costs_ms.emplace_back(duration);
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "ERROR: Inference failed: " << e.what() << " with error code: " << e.GetOrtErrorCode()
                      << std::endl;
            return {false, {}};
        }
        catch (const std::exception& e)
        {
            std::cerr << "ERROR: Inference failed with exception: " << e.what() << std::endl;
            return {false, {}};
        }
        for (size_t output_idx = 0; output_idx < m_output_names.size(); ++output_idx)
        {
            if (AllNans(m_output_tensors[output_idx]))
            {
                std::cerr << "ERROR: Sample " << sample_idx << " inference produced all NaNs for " << "'"
                          << m_output_names[output_idx] << "'." << std::endl;
                return {false, {}};
            }
        }
        // Write outputs to disk immediately without storing in memory
        // This is the key difference from RunInferences - we skip SaveSampleOutputs
        if (m_dataset_reader) {
            for (size_t output_idx = 0; output_idx < m_output_names.size(); ++output_idx)
            {
                if (!outputs_writer.Store(sample_idx, m_output_names[output_idx], m_output_tensors[output_idx]))
                {
                    std::cerr << "ERROR: Failed to store outputs for sample " << sample_idx << ", output " << m_output_names[output_idx] << std::endl;
                    return { false, {} };
                }
            }
        }
    }
    double avg_cpu_usage = cpu_monitor.GetCurrentValue();
    MemoryUsage memory_usage = GetMemoryUsage();
    performance_result.peak_workingset_size = memory_usage.peak_working_set_size;
    performance_result.peak_pagefile_usage = memory_usage.peak_pagefile_usage;
    performance_result.average_cpu_usage = avg_cpu_usage;
    return { true, performance_result };
}

void PerformanceRunner::ClearOutputs()
{
    for (auto& [_, outputs] : m_dataset_outputs) {
        outputs.clear();
        outputs.shrink_to_fit();
    }
    m_dataset_outputs.clear();
}

void PerformanceRunner::ClearOutputContents()
{
    for (auto& [_, stored_output] : m_dataset_outputs)
    {
        stored_output.clear();
        stored_output.shrink_to_fit();
    }
}
