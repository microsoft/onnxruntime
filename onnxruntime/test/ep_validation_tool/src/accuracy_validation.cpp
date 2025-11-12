#include "accuracy_validation.h"
#include "tensor_utils.h"

#include "onnxruntime_cxx_api.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <unordered_map>

float DiffL2Norm(const Ort::Value& a, const Ort::Value& b)
{
    std::vector<float> data_a;
    CastOrtValueData(a, data_a);

    std::vector<float> data_b;
    CastOrtValueData(b, data_b);

    float curr_norm = 0;
    for (auto i = 0; i < data_a.size(); i++)
    {
        curr_norm += (data_a[i] - data_b[i]) * (data_a[i] - data_b[i]);
    }

    float diff_norm = sqrt(curr_norm);
    return diff_norm;
}

std::pair<float, bool> CalculateOffsets(
    const Ort::Value& f32_cpu_output, const Ort::Value& qdq_npu_output, const float& l2_norm_output, float threshold)
{
    float fp32_vs_npu = DiffL2Norm(f32_cpu_output, qdq_npu_output);
    float diff = std::abs(fp32_vs_npu - l2_norm_output);
    bool is_good_diff = (threshold > 0.0f) ? (diff <= threshold) : true;

    return {diff, is_good_diff};
}

std::pair<std::unordered_map<std::string, std::vector<float>>, bool> CheckAccuracy(
    std::unordered_map<std::string, float>& acc_threshold,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& npu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& f32_cpu_outputs,
    const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs,
    const std::unordered_map<std::string, std::string> output_map)
{
    bool is_tolerable = true;
    std::unordered_map<std::string, std::vector<float>> diffs_per_sample;

    for (const auto& [output_name, _] : output_map)
    {
        std::cout << "Processing output: " << output_name << std::endl;

        auto npu_it = npu_outputs.find(output_name);
        if (npu_it == npu_outputs.end())
        {
            std::cerr << "ERROR: Output '" << output_name << "' not found in NPU outputs." << std::endl;
            return {{}, false};
        }

        auto f32_it = f32_cpu_outputs.find(output_name);
        if (f32_it == f32_cpu_outputs.end())
        {
            std::cerr << "ERROR: Output '" << output_name << "' not found in F32 CPU outputs." << std::endl;
            return {{}, false};
        }

        auto l2_it = l2_norm_outputs.find(output_name);
        if (l2_it == l2_norm_outputs.end())
        {
            std::cerr << "ERROR: Output '" << output_name << "' not found in L2 norm outputs." << std::endl;
            return {{}, false};
        }

        const auto& npu_samples = npu_it->second;
        const auto& f32_samples = f32_it->second;
        const auto& l2_samples = l2_it->second;

        if (npu_samples.size() != f32_samples.size())
        {
            std::cerr << "ERROR: Mismatch in number of samples for output '" << output_name
                      << "' (NPU: " << npu_samples.size() << ", F32 CPU: " << f32_samples.size() << ")." << std::endl;
            return {{}, false};
        }

        if (npu_samples.size() != l2_samples.size())
        {
            std::cerr << "ERROR: Mismatch in number of samples for output '" << output_name
                      << "' (NPU: " << npu_samples.size() << ", L2 norm: " << l2_samples.size() << ")." << std::endl;
            return {{}, false};
        }

        std::vector<float> diffs;
        diffs.reserve(npu_samples.size());

        for (size_t sample_idx = 0; sample_idx < npu_samples.size(); ++sample_idx)
        {
            float threshold =
                acc_threshold.find(output_name) != acc_threshold.end() ? acc_threshold.at(output_name) : 0.0f;

            auto [diff, good_diff] =
                CalculateOffsets(f32_samples[sample_idx], npu_samples[sample_idx], l2_samples[sample_idx], threshold);

            diffs.push_back(diff);
            if (!good_diff)
            {
                is_tolerable = false;
                std::cout << "Sample " << sample_idx << " accuracy for '" << output_name << "' "
                          << "worse than the threshold set." << std::endl;
            }
        }

        diffs_per_sample[output_name] = std::move(diffs);
    }

    return {diffs_per_sample, is_tolerable};
}

std::unordered_map<std::string, std::vector<float>> CalculateCPUL2Norm(
    const std::unordered_map<std::string, std::vector<Ort::Value>>& qdq_cpu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& fp32_cpu_outputs)
{
    std::unordered_map<std::string, std::vector<float>> l2_norm_outputs;

    for (const auto& [output_name, fp32_outputs] : fp32_cpu_outputs)
    {
        auto it = qdq_cpu_outputs.find(output_name);
        if (it == qdq_cpu_outputs.end())
        {
            std::cerr << "ERROR: Output '" << output_name << "' not found in QDQ CPU outputs." << std::endl;
            return {};
        }

        const auto& cpu_outputs = it->second;
        if (cpu_outputs.size() != fp32_outputs.size())
        {
            std::cerr << "ERROR: Mismatch in number of samples for output '" << output_name
                      << "' (QDQ: " << cpu_outputs.size() << ", FP32: " << fp32_outputs.size() << ")." << std::endl;
            return {};
        }

        std::vector<float> l2_norms;
        l2_norms.reserve(cpu_outputs.size());

        for (size_t i = 0; i < cpu_outputs.size(); ++i)
        {
            float norm = DiffL2Norm(cpu_outputs[i], fp32_outputs[i]);
            l2_norms.push_back(norm);
        }

        l2_norm_outputs[output_name] = std::move(l2_norms);
    }

    return l2_norm_outputs;
}
