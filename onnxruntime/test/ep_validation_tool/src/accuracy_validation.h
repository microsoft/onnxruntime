#include "tensors_reader_writer.h"
#include "tensor_utils.h"

#include "onnxruntime_cxx_api.h"

#include <string>
#include <unordered_map>
#include "config.h"
#include <cmath>
#include <optional>

// Portable constant for PI; MSVC does not define M_PI by default.
constexpr double kPI = 3.14159265358979323846;

using MetricDiffs = std::unordered_map<std::string, std::vector<float>>;
using AccuracyResult = std::unordered_map<std::string, MetricDiffs>;

// Computes L2 norm of the difference between two tensors
float DiffL2Norm(const Ort::Value& a, const Ort::Value& b);

// Computes cosine similarity between two tensors; returns nullopt on invalid inputs
std::optional<float> CosineSimilarity(const Ort::Value& a, const Ort::Value& b);

// Computes mean angular error (in degrees) between two tensors; returns nullopt on invalid inputs
std::optional<float> MeanAngularError(const Ort::Value& a, const Ort::Value& b);

// Computes mean squared error between two tensors; returns nullopt on invalid inputs
std::optional<float> MeanSquaredError(const Ort::Value& a, const Ort::Value& b);

std::pair<AccuracyResult, bool> CheckAccuracy(
    const std::vector<AccuracyMetric>& metrics,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& acc_threshold,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& npu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& f32_cpu_outputs,
    const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs,
    const std::unordered_map<std::string, std::string>& output_map);

std::unordered_map<std::string, std::vector<float>> CalculateCPUL2Norm(
    const std::unordered_map<std::string, std::vector<Ort::Value>>& qdq_cpu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& fp32_cpu_outputs);
