#include "accuracy_validation.h"
#include "tensor_utils.h"

#include "onnxruntime_cxx_api.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <optional>

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

std::optional<float> CosineSimilarity(const Ort::Value& a, const Ort::Value& b)
{
    if (!a.IsTensor() || !b.IsTensor())
        return std::nullopt;

    // Get tensor shape info
    Ort::TensorTypeAndShapeInfo a_info = a.GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo b_info = b.GetTensorTypeAndShapeInfo();

    // Validate element count
    const size_t num_elems = a_info.GetElementCount();
    if (num_elems == 0 || num_elems != b_info.GetElementCount())
        return std::nullopt;

    // Assumes tensors are contiguous and of type float
    const float* data_a = a.GetTensorData<float>();
    const float* data_b = b.GetTensorData<float>();

    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;

    for (size_t i = 0; i < num_elems; ++i)
    {
        const double va = static_cast<double>(data_a[i]);
        const double vb = static_cast<double>(data_b[i]);

        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }

    const double eps = 1e-12;
    const double denom = std::sqrt(na) * std::sqrt(nb);

    // Invalid comparison: zero or near-zero magnitude
    if (denom < eps)
        return std::nullopt;

    return static_cast<float>(dot / denom);
}

std::pair<float, bool> CalculateOffsetsL2(
    const Ort::Value& f32_cpu_output, const Ort::Value& qdq_npu_output, const float& l2_norm_output, float threshold)
{
    float fp32_vs_npu = DiffL2Norm(f32_cpu_output, qdq_npu_output);
    float diff = std::abs(fp32_vs_npu - l2_norm_output);
    bool is_good_diff = (threshold > 0.0f) ? (diff <= threshold) : true;

    return {diff, is_good_diff};
}

std::pair<AccuracyResult, bool> CheckAccuracy(
    const std::vector<AccuracyMetric>& metrics,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& acc_threshold,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& npu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& f32_cpu_outputs,
    const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs,
    const std::unordered_map<std::string, std::string>& output_map)
{
    bool is_tolerable = true;
    AccuracyResult diffs_per_sample;
    bool do_l2 = false;
    bool do_cos = false;
    for (auto m : metrics)
    {
        if (m == AccuracyMetric::L2Norm) do_l2 = true;
        if (m == AccuracyMetric::Cosine) do_cos = true;
    }

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

        const auto& npu_samples = npu_it->second;
        const auto& f32_samples = f32_it->second;
        const auto* l2_samples_ptr = [&]() -> const std::vector<float>* {
            if (!do_l2)
                return nullptr;
            auto l2_it = l2_norm_outputs.find(output_name);
            if (l2_it == l2_norm_outputs.end())
                return nullptr;
            return &l2_it->second;
        }();

        if (npu_samples.size() != f32_samples.size())
        {
            std::cerr << "ERROR: Mismatch in number of samples for output '" << output_name
                      << "' (NPU: " << npu_samples.size() << ", F32 CPU: " << f32_samples.size() << ")." << std::endl;
            return {{}, false};
        }
        if (do_l2)
        {
            if (!l2_samples_ptr)
            {
                std::cerr << "ERROR: Output '" << output_name << "' not found in L2 norm outputs." << std::endl;
                return {{}, false};
            }
            if (npu_samples.size() != l2_samples_ptr->size())
            {
                std::cerr << "ERROR: Mismatch in number of samples for output '" << output_name
                          << "' (NPU: " << npu_samples.size() << ", L2 norm: " << l2_samples_ptr->size()
                          << ")." << std::endl;
                return {{}, false};
            }
        }

        MetricDiffs metric_diffs;
        if (do_l2)
        {
            metric_diffs[ToString(AccuracyMetric::L2Norm)].reserve(npu_samples.size());
        }
        if (do_cos)
        {
            metric_diffs[ToString(AccuracyMetric::Cosine)].reserve(npu_samples.size());
        }

        for (size_t sample_idx = 0; sample_idx < npu_samples.size(); ++sample_idx)
        {
            // Per-metric, per-tensor thresholds
            float l2_threshold = 0.0f;
            float cosine_threshold = 0.0f;
            {
                // acc_threshold[metric_name][tensor_name]
                auto metric_it = acc_threshold.find(ToString(AccuracyMetric::L2Norm));
                if (metric_it != acc_threshold.end())
                {
                    auto t_it = metric_it->second.find(output_name);
                    if (t_it != metric_it->second.end())
                    {
                        l2_threshold = t_it->second;
                    }
                }

                metric_it = acc_threshold.find(ToString(AccuracyMetric::Cosine));
                if (metric_it != acc_threshold.end())
                {
                    auto t_it = metric_it->second.find(output_name);
                    if (t_it != metric_it->second.end())
                    {
                        cosine_threshold = t_it->second;
                    }
                }
            }

            if (do_cos)
            {
                auto cos_opt = CosineSimilarity(f32_samples[sample_idx], npu_samples[sample_idx]);

                bool good = true;

                if (!cos_opt.has_value())
                {
                    // Invalid cosine comparison
                    good = false;
                }
                else
                {
                    float cos_value = cos_opt.value();

                    // Always record cosine value when requested
                    metric_diffs[ToString(AccuracyMetric::Cosine)].push_back(cos_value);

                    // If no cosine threshold is configured (0.0), treat all valid cosine values as acceptable.
                    good = (cosine_threshold > 0.0f)
                        ? (cos_value >= cosine_threshold)
                        : true;
                }

                if (!good)
                {
                    is_tolerable = false;
                    std::cout << "Sample " << sample_idx << " accuracy for '" << output_name
                        << "' (Cosine) worse than the threshold set." << std::endl;
                }
            }

            if (do_l2)
            {
                auto [l2_diff, l2_good] = CalculateOffsetsL2(
                    f32_samples[sample_idx],
                    npu_samples[sample_idx],
                    (*l2_samples_ptr)[sample_idx],
                    l2_threshold);

                metric_diffs[ToString(AccuracyMetric::L2Norm)].push_back(l2_diff);

                if (!l2_good)
                {
                    is_tolerable = false;
                    std::cout << "Sample " << sample_idx << " accuracy for '" << output_name
                              << "' (L2Norm) worse than the threshold set." << std::endl;
                }
            }
        }

        diffs_per_sample[output_name] = std::move(metric_diffs);
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
