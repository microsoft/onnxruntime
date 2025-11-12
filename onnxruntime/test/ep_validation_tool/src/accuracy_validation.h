#include "tensors_reader_writer.h"
#include "tensor_utils.h"

#include "onnxruntime_cxx_api.h"

#include <string>
#include <unordered_map>

std::pair<std::unordered_map<std::string, std::vector<float>>, bool> CheckAccuracy(
    std::unordered_map<std::string, float>& acc_threshold,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& npu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& f32_cpu_outputs,
    const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs,
    const std::unordered_map<std::string, std::string> output_map);

std::unordered_map<std::string, std::vector<float>> CalculateCPUL2Norm(
    const std::unordered_map<std::string, std::vector<Ort::Value>>& qdq_cpu_outputs,
    const std::unordered_map<std::string, std::vector<Ort::Value>>& fp32_cpu_outputs);
