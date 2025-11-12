#pragma once

#include "arg_parser.h"
#include "string_utils.h"

#include "onnxruntime_cxx_api.h"
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "nlohmann/json.hpp"
#include <fstream>
#ifdef USE_WINML_FEATURES
#include <cstdint>
#include <windows.h>
#include <climits>
#include <WindowsAppSDK-VersionInfo.h>
#include <appmodel.h>
#endif

enum class ExecutionStage
{
    DATAGEN = -1,
    COMPILATION = 0,
    INFERENCE = 1,
    ACCURACY = 2
};
#ifdef USE_WINML_FEATURES
// Global list of execution providers accessible from other .cpp files
inline std::unordered_set<std::string> g_executionProviders = {
    "VitisAIExecutionProvider", "OpenVINOExecutionProvider", "QNNExecutionProvider"};
struct EpInfo
{
    std::string name;
    std::optional<std::string> deviceType;
    std::optional<uint32_t> deviceVendorId;
    std::optional<uint32_t> deviceId;
};

struct WASDKRuntimeInfo
{
    UINT32 majorMinorVersion;
    std::wstring versionTag;
    PACKAGE_VERSION packageVersion;
};

#endif
static bool ValidateModelEncryption(const std::wstring& model_path, const std::string& model_key)
{
    bool is_model_encrypted = (std::filesystem::path(model_path).extension() == L".onnxe");
    if (is_model_encrypted && model_key.empty())
    {
        std::wcerr << "Model is encrypted. Please provide the key through '--modelKey'." << std::endl;
        return false;
    }
    if (!is_model_encrypted && !model_key.empty())
    {
        std::wcerr << "Model is not encrypted. Please remove the key from '--modelKey'." << std::endl;
        return false;
    }
    return true;
}

struct Config
{
public:
    Config()
    {
        m_arg_parser.AddArgument(L"--modelPath", L"-m", L"ONNX or ONNXE model file path.", true);
        m_arg_parser.AddArgument(
            L"--modelKey",
            L"-k",
            L"Encrypted model key. Ignore in case path to ONNX model was passed. Can be encryption key ID ('key', "
            L"'long_key', 'lora_key_v1') or literal key.",
            false);
        m_arg_parser.AddArgument(
            L"--datasetDir", L"-i", L"Dataset folder path. If omitted, random inputs are generated.");
        m_arg_parser.AddArgument(L"--outputDir", L"-o", L"Output files folder path. Created if doesn't exist.", true);
        m_arg_parser.AddArgument(
            L"--refModelPath",
            L"-rm",
            L"Path to fp32 version of the ONNX model."
            "Required if --stage 2.",
            false,
            false);
        m_arg_parser.AddArgument(
            L"--refOutDir", L"-ro", L"Folder containing reference outputs for accuracy validation. ");
        m_arg_parser.AddArgument(
            L"--resultPath",
            L"-rp",
            L"Path to the result file. If not specified, results will only be printed in the console.");
        m_arg_parser.AddArgument(
            L"--stage",
            L"-st",
            L"Stages to run. -1 - data generation, 0 - model compilation only, 1 - inference, 2 - accuracy validation.",
            false,
            false,
            L"1");
        m_arg_parser.AddArgument(
            L"--repeatTimes", L"-r", L"Number of times to repeat the inference.", false, false, L"1");
        m_arg_parser.AddArgument(L"--warmupIterations", L"-w", L"Number of warmup iterations.", false, false, L"0");
        m_arg_parser.AddArgument(
            L"--sessionOptions", L"-s", L"Session options, common for all EPs. In form 'key1|value1 key2|value2 ...'.");
        m_arg_parser.AddArgument(L"--epOptions", L"-e", L"EP-specific options. In form 'key1|value1 key2|value2 ...'.");
        m_arg_parser.AddArgument(L"--graphOptimization", L"-g", L"Graph optimization level.", false, false, L"99");
        m_arg_parser.AddArgument(
            L"--perfThreshold",
            L"-t",
            L"Threshold for model to be considered performant. (average inference time in ms)",
            false,
            false,
            L"0.0");
        m_arg_parser.AddArgument(
            L"--accThreshold",
            L"-q",
            L"Thresholds for model to be considered accurate. In form 'key1|value1 key2|value2 ...'. (L2 norm "
            L"threshold for each output tensor)");
        m_arg_parser.AddArgument(L"--epPolicy", L"-ep", L"Ep Selection Policy");
        m_arg_parser.AddArgument(L"--compile", L"-cp", L"Compile");
        m_arg_parser.AddArgument(L"--compiledModelPath", L"-cmp", L"Compiled Model Path");
        m_arg_parser.AddArgument(L"--executionProvider", L"-x", L"Execution provider to use.");
        m_arg_parser.AddArgument(
            L"--epName", L"-epn", L"EP name like CPUExecutionProvider , QNNExecutionProvider, etc");
        m_arg_parser.AddArgument(L"--epDeviceType", L"-dt", L"Execution provider Device Type. Ex-CPU, NPU, GPU");
        m_arg_parser.AddArgument(L"--epVendorId", L"-vid", L"Id of Vendor like Intel, etc");
        m_arg_parser.AddArgument(L"--epDeviceId", L"-did", L"Device Id associated with EP and Device Type.");
        m_arg_parser.AddArgument(
            L"--wasdkMajorMinor",
            L"-sdkmm",
            L"The major and minor version of the Windows App SDK release, encoded as a uint32 (0xMMMMNNNN where "
            L"M=major, N=minor).");
        m_arg_parser.AddArgument(
            L"--wasdkVersionTag",
            L"-sdkvt",
            L"The Windows App SDK release version tag. For example, preview2,experimental, or empty string for "
            L"stable.");
        m_arg_parser.AddArgument(
            L"--wasdkPackageVersion",
            L"-sdkpv",
            L"The version of the Windows App SDK runtime, as a uint64l for example, 0x03E801BE03240000.");
    }

    bool FromCommandLine(int argc, const wchar_t* argv[])
    {
        bool success = true;

        if (!m_arg_parser.ParseArgs(argc, argv))
        {
            std::wcout << m_arg_parser.GetHelpText() << std::endl;
            return false;
        }

        model_path = m_arg_parser.Get(L"--modelPath");
        model_key = wstring_to_string(m_arg_parser.Get(L"--modelKey"));
        model_name = wstring_to_string(model_path.substr(m_arg_parser.Get(L"--modelPath").find_last_of(L"\\") + 1));
        dataset_dir = m_arg_parser.Get(L"--datasetDir");
        out_dir = m_arg_parser.Get(L"--outputDir");
        ref_model_path = m_arg_parser.Get(L"--refModelPath");
        ref_out_dir = m_arg_parser.Get(L"--refOutDir");
        result_path = m_arg_parser.Get(L"--resultPath");
#ifndef USE_WINML_FEATURES
        execution_provider = wstring_to_string(m_arg_parser.Get(L"--executionProvider"));
#endif
        graph_opt_level = m_arg_parser.GetInt(L"--graphOptimization");
        stage = static_cast<ExecutionStage>(m_arg_parser.GetInt(L"--stage"));

        repeat_times = m_arg_parser.GetInt(L"--repeatTimes");
        warmup_iterations = m_arg_parser.GetInt(L"--warmupIterations");

        success = ValidateModelEncryption(model_path, model_key);
        if (s_opt_levels.find(graph_opt_level) == s_opt_levels.end())
        {
            std::wcout << "Invalid graph optimization level." << std::endl;
            success = false;
        }

        if (!ParseOptions(m_arg_parser.Get(L"--sessionOptions"), session_options))
        {
            std::wcout << "Error parsing session options." << std::endl;
            success = false;
        }

        if (!ParseOptions(m_arg_parser.Get(L"--epOptions"), ep_options))
        {
            std::wcout << "Error parsing EP options." << std::endl;
            success = false;
        }

        if (stage == ExecutionStage::DATAGEN)
        {
            if (dataset_dir.empty())
            {
                std::wcout << "Dataset directory must be specified for data generation stage." << std::endl;
                success = false;
            }
        }

        if (stage >= ExecutionStage::INFERENCE)
        {
            perf_threshold = m_arg_parser.GetFloat(L"--perfThreshold");
            if (perf_threshold == 0.0f)
            {
                std::wcout << "WARNING: Performance threshold not specified. Performance won't be validated."
                           << std::endl;
            }
        }

        if (stage >= ExecutionStage::ACCURACY)
        {
            if (!ParseOptions(m_arg_parser.Get(L"--accThreshold"), acc_threshold))
            {
                std::wcout << "Error parsing accuracy threshold options. "
                              "Argument --accThreshold is missing or in a bad format."
                           << std::endl;
                success = false;
            }
            else if (acc_threshold.empty())
            {
                std::wcout << "WARNING: Accuracy thresholds not set. Accuracy will be reported, but won't be validated."
                           << std::endl;
            }
            if (dataset_dir.empty())
            {
                std::wcout << "Dataset directory must be specified for accuracy stage." << std::endl;
                success = false;
            }
        }
#ifdef USE_WINML_FEATURES
        ep_policy = ParseEpDevicePolicy(m_arg_parser.Get(L"--epPolicy"));
        shouldCompile = m_arg_parser.IsArg(L"--compile");
        compiledModelPath = m_arg_parser.Get(L"--compiledModelPath");
        // Parse EP Name if provided and construct EPInfo object with other params like device type, vendorId, deviceId,
        // etc if provided
        std::string ep_name_str = wstring_to_string(m_arg_parser.Get(L"--epName"));
        if (!ep_name_str.empty())
        {
            ep_info = EpInfo();
            ep_info->name = ep_name_str;
            std::cout << "Epname : " << ep_name_str << std::endl;
            // Parse device type if provided
            std::wstring device_type_str = m_arg_parser.Get(L"--epDeviceType");
            if (!device_type_str.empty())
            {
                ep_info->deviceType = wstring_to_string(device_type_str);
                std::cout << "Device Type : " << *ep_info->deviceType << std::endl;
            }

            // Parse vendor ID if provided
            std::wstring vendor_id_str = m_arg_parser.Get(L"--epVendorId");
            if (!vendor_id_str.empty())
            {
                try
                {
                    ep_info->deviceVendorId = static_cast<uint32_t>(std::stoul(vendor_id_str));
                    std::wcout << "Device vendorid : " << *ep_info->deviceVendorId << std::endl;
                }
                catch (const std::exception&)
                {
                    std::wcerr << L"Invalid vendor ID value: " << vendor_id_str << std::endl;
                    success = false;
                }
            }

            // Parse device ID if provided
            std::wstring device_id_str = m_arg_parser.Get(L"--epDeviceId");
            if (!device_id_str.empty())
            {
                try
                {
                    ep_info->deviceId = static_cast<uint32_t>(std::stoul(device_id_str));
                    std::wcout << "DeviceId : " << *ep_info->deviceId << std::endl;
                }
                catch (const std::exception&)
                {
                    std::wcerr << L"Invalid device ID value: " << device_id_str << std::endl;
                    success = false;
                }
            }
        }
        runtimeInfo = ValidateAndCreateRuntimeInfoObject();

#endif
        return success;
    }

public:
    std::string model_name;
    std::string model_key;
    std::wstring model_path;
    std::wstring ref_model_path;
    std::wstring dataset_dir;
    std::wstring out_dir;
    std::wstring result_path;
    std::wstring ref_out_dir;
#ifndef USE_WINML_FEATURES
    std::string execution_provider;
#endif
#ifdef USE_WINML_FEATURES
    std::optional<OrtExecutionProviderDevicePolicy> ep_policy;
    bool shouldCompile;
    std::wstring compiledModelPath;
    std::optional<EpInfo> ep_info;
    std::optional<WASDKRuntimeInfo> runtimeInfo;
#endif
    std::unordered_map<std::string, std::string> session_options;
    std::unordered_map<std::string, std::string> ep_options;
    int graph_opt_level = ORT_ENABLE_ALL;
    ExecutionStage stage = ExecutionStage::ACCURACY;
    int repeat_times = 1;
    int warmup_iterations = 0;
    float perf_threshold = 0.0;
    std::unordered_map<std::string, float> acc_threshold;

private:
    static inline const std::unordered_set<int> s_opt_levels{
        ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL};

    // Forked from
    // https://github.com/microsoft/onnxruntime/blob/2d05c4bcd940aa25561ed7de26481f219618dd7a/onnxruntime/test/perftest/strings_helper.cc#L15.
    template<typename T>
    bool ParseOptions(const std::wstring& options_string, std::unordered_map<std::string, T>& options)
    {
        std::istringstream ss(wstring_to_string(options_string));
        std::string token;

        while (ss >> token)
        {
            if (token == "")
            {
                continue;
            }

            std::string_view token_sv(token);

            auto pos = token_sv.find("|");
            if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length())
            {
                std::cerr << "Use a '|' to separate the key and value for the option you are trying to use."
                          << std::endl;
                return false;
            }

            std::string key(token_sv.substr(0, pos));
            T value;

            if constexpr (std::is_same_v<T, std::string>)
            {
                value = std::string(token_sv.substr(pos + 1));
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                value = std::stof(std::string(token_sv.substr(pos + 1)));
            }

            auto it = options.find(key);
            if (it != options.end())
            {
                std::cerr << "Specified duplicate configuration entry: " << key << std::endl;
                return false;
            }

            options.insert(std::make_pair(std::move(key), std::move(value)));
        }

        return true;
    }

private:
    ArgParser m_arg_parser;

#ifdef USE_WINML_FEATURES
    std::optional<OrtExecutionProviderDevicePolicy> ParseEpDevicePolicy(const std::wstring& policy_str)
    {
        if (policy_str == L"NPU")
        {
            return OrtExecutionProviderDevicePolicy_PREFER_NPU;
        }
        else if (policy_str == L"CPU")
        {
            return OrtExecutionProviderDevicePolicy_PREFER_CPU;
        }
        else if (policy_str == L"GPU")
        {
            return OrtExecutionProviderDevicePolicy_PREFER_GPU;
        }
        else if (policy_str == L"DEFAULT")
        {
            return OrtExecutionProviderDevicePolicy_DEFAULT;
        }
        else if (policy_str == L"DISABLE" || policy_str.empty())
        {
            return std::nullopt;
        }
        else
        {
            std::wcout << L"Unknown EP policy: " << policy_str << L", using default (DISABLE)\n";
            // return std::nullopt;
            return OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY;
        }
    }
    std::optional<WASDKRuntimeInfo> ValidateAndCreateRuntimeInfoObject()
    {
        std::string majorMinorStr = wstring_to_string(m_arg_parser.Get(L"--wasdkMajorMinor"));
        std::wstring versionTagStr = m_arg_parser.Get(L"--wasdkVersionTag");
        std::string packageVersionStr = wstring_to_string(m_arg_parser.Get(L"--wasdkPackageVersion"));

        bool okMajorMinor = false;
        bool okPkg64 = false;

        UINT32 majorMinorVersion = 0;
        UINT64 packageVersionNum = 0;

        try
        {
            // base = 0 → auto-detects 0x (hex), leading 0 (octal), or decimal
            majorMinorVersion = std::stoul(majorMinorStr, nullptr, 16);
            okMajorMinor = true;
        }
        catch (...)
        {
            okMajorMinor = false;
        }

        try
        {
            // IMPORTANT: use stoull for 64-bit
            packageVersionNum = std::stoull(packageVersionStr, nullptr, 16);
            okPkg64 = true;
        }
        catch (...)
        {
            okPkg64 = false;
        }
        PACKAGE_VERSION packageVersion{};
        packageVersion.Version = packageVersionNum;
        // --- Final check based on actual parsed values & PCWSTR content ---
        if (okMajorMinor && okPkg64)
        {
            WASDKRuntimeInfo info;
            info = {
                majorMinorVersion, // UINT32
                versionTagStr,     // PCWSTR (backed by versionTagStr's storage; keep it alive)
                packageVersion};
            return std::optional<WASDKRuntimeInfo>{info};
        }
        else
        {
            // Optional: diagnostics
            if (!majorMinorStr.empty() && !okMajorMinor)
            {
                std::cerr << "Invalid --wasdkMajorMinor: " << majorMinorStr << std::endl;
            }
            if (!packageVersionStr.empty() && !okPkg64)
            {
                std::cerr << "Invalid --wasdkPackageVersion: " << packageVersionStr << std::endl;
            }
            return std::nullopt;
        }
    }
#endif
};
