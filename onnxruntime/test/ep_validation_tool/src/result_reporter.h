#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

class ResultReporter
{
public:
    ResultReporter(
        const Config config,
        CompilationResult& compilation_result,
        const PerformanceResult& performance_result,
        std::unordered_map<std::string, std::vector<float>>& accuracy_result)
        : m_config(config),
          m_compilation_result(compilation_result),
          m_performance_result(performance_result),
          m_accuracy_result(accuracy_result)
    {
    }

    void PrintToConsole() const
    {
        int64_t total_inference_cost_ms = std::accumulate(
            m_performance_result.inference_time_costs_ms.begin(),
            m_performance_result.inference_time_costs_ms.end(),
            int64_t(0));

        std::cout << "\n----- Test results -----\n"
                  << "Model name: " << m_config.model_name << "\n"
                  << "Model compilation time: " << m_compilation_result.compilation_time_cost_ms << " ms\n"
                  << "Compilation peak working set size: " << m_compilation_result.peak_workingset_size << " bytes\n"
                  << "Compilation peak pagefile usage: " << m_compilation_result.peak_pagefile_usage << " bytes\n"
                  << "Total inference requests: " << m_performance_result.inference_time_costs_ms.size() << "\n"
                  << "Average inference time cost: "
                  << static_cast<float>(total_inference_cost_ms) / m_performance_result.inference_time_costs_ms.size()
                  << " ms\n"
                  << "Inference peak working set size: " << m_performance_result.peak_workingset_size << " bytes\n"
                  << "Inference peak pagefile usage: " << m_performance_result.peak_pagefile_usage << " bytes\n"
                  << "Average CPU usage: " << m_performance_result.average_cpu_usage << " %" << std::endl;

        for (const auto& [tensor_name, norm_diffs] : m_accuracy_result)
        {
            float total_node_norm_diff = std::accumulate(norm_diffs.begin(), norm_diffs.end(), 0.0f);
            float average_norm_diff = total_node_norm_diff / norm_diffs.size();
            std::cout << "Average diff. of L2 norms for '" << tensor_name << "': " << average_norm_diff << "\n"
                      << std::endl;
        }

        std::cout << "----- Test configuration -----\n"
                  << "Session options: " << MapToString(m_config.session_options) << "\n"
                  << "EP options: " << MapToString(m_config.ep_options) << "\n"
                  << "Graph optimization level: " << m_config.graph_opt_level << "\n"
                  << "Performance threshold: " << m_config.perf_threshold << "\n"
                  << "Accuracy thresholds: " << MapToString(m_config.acc_threshold) << "\n"
                  << std::endl;
    }

    void DumpToJson(const std::wstring& result_path) const
    {
        nlohmann::json json_result;
        int64_t total_inference_cost_ms = std::accumulate(
            m_performance_result.inference_time_costs_ms.begin(),
            m_performance_result.inference_time_costs_ms.end(),
            int64_t(0));

        // TODO: Refactor code in order to move the stat calculation outside of the printing functions.
        json_result["model_name"] = m_config.model_name;
        json_result["compilation_time_ms"] = m_compilation_result.compilation_time_cost_ms;
        json_result["compilation_peak_workingset_size"] = m_compilation_result.peak_workingset_size;
        json_result["compilation_peak_pagefile_usage"] = m_compilation_result.peak_pagefile_usage;
        json_result["total_inference_requests"] = m_performance_result.inference_time_costs_ms.size();
        json_result["average_inference_time_cost"] =
            static_cast<float>(total_inference_cost_ms) / m_performance_result.inference_time_costs_ms.size();
        json_result["inference_peak_workingset_size"] = m_performance_result.peak_workingset_size;
        json_result["inference_peak_pagefile_usage"] = m_performance_result.peak_pagefile_usage;
        json_result["average_cpu_usage"] = m_performance_result.average_cpu_usage;

        nlohmann::json accuracy_json;
        for (const auto& [tensor_name, norm_diffs] : m_accuracy_result)
        {
            float total_node_norm_diff = std::accumulate(norm_diffs.begin(), norm_diffs.end(), 0.0f);
            float average_norm_diff = total_node_norm_diff / norm_diffs.size();
            accuracy_json[tensor_name] = average_norm_diff;
        }
        json_result["accuracy"] = accuracy_json;

        nlohmann::json test_configuration_json;
        test_configuration_json["session_options"] = m_config.session_options;
        test_configuration_json["ep_options"] = m_config.ep_options;
        test_configuration_json["graph_optimization_level"] = m_config.graph_opt_level;
        test_configuration_json["performance_threshold"] = m_config.perf_threshold;
        test_configuration_json["accuracy_thresholds"] = m_config.acc_threshold;
        json_result["test_configuration"] = test_configuration_json;

        std::ofstream file(result_path);
        file << json_result.dump(4);
    }

private:
    Config m_config;
    CompilationResult m_compilation_result;
    PerformanceResult m_performance_result;
    std::unordered_map<std::string, std::vector<float>> m_accuracy_result;

    template<typename T>
    std::string MapToString(const std::unordered_map<std::string, T>& map) const
    {
        std::ostringstream oss;
        for (const auto& pair : map)
        {
            oss << pair.first << "|" << pair.second << " ";
        }

        return oss.str();
    }
};
