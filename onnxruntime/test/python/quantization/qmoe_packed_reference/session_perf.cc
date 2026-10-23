#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

static void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static size_t Used() {
  Check(cudaDeviceSynchronize());
  size_t available, total;
  Check(cudaMemGetInfo(&available, &total));
  return total - available;
}

template <typename Action>
static double Time(Action action) {
  Check(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  action();
  Check(cudaDeviceSynchronize());
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Feed {
  std::string name;
  void* device = nullptr;
  Ort::Value value{nullptr};
  ~Feed() { if (device) cudaFree(device); }
};

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  try {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    Check(cudaSetDevice(0));
    Check(cudaFree(nullptr));
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "qmoe-session-perf");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    OrtCUDAProviderOptions cuda_options{};
    options.AppendExecutionProvider_CUDA(cuda_options);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AddConfigEntry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "67108864");
    const size_t before_feeds = Used();
    const auto feed_start = std::chrono::steady_clock::now();
    std::ifstream manifest(argv[2]);
    if (!manifest) throw std::runtime_error("Cannot read feed manifest");
    std::vector<std::unique_ptr<Feed>> feeds;
    std::vector<Ort::Value> values;
    std::vector<const char*> names;
    const Ort::MemoryInfo gpu("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::string name, path;
    int type, rank;
    while (manifest >> name >> path >> type >> rank) {
      std::vector<int64_t> shape(rank);
      size_t elements = 1;
      for (auto& dimension : shape) { manifest >> dimension; elements *= dimension; }
      const size_t bytes = elements * (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : 2);
      std::vector<char> host(bytes);
      std::ifstream input(path, std::ios::binary);
      if (!input.read(host.data(), bytes)) throw std::runtime_error("Cannot read " + path);
      auto feed = std::make_unique<Feed>();
      feed->name = name;
      Check(cudaMalloc(&feed->device, bytes));
      Check(cudaMemcpy(feed->device, host.data(), bytes, cudaMemcpyHostToDevice));
      values.push_back(Ort::Value::CreateTensor(gpu, feed->device, bytes, shape.data(), shape.size(),
                                               static_cast<ONNXTensorElementDataType>(type)));
      feeds.push_back(std::move(feed));
    }
    for (const auto& feed : feeds) names.push_back(feed->name.c_str());
    const size_t before_init = Used();
    const double feed_setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - feed_start).count();
    std::unique_ptr<Ort::Session> session;
    const double init_ms = Time([&] { session = std::make_unique<Ort::Session>(env, argv[1], options); });
    const size_t after_init = Used();
    const char* output_name = "output";
    std::vector<Ort::Value> outputs;
    const auto run = [&] {
      outputs = session->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), &output_name, 1);
    };
    const double first_ms = Time(run);
    const size_t after_first = Used();
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const bool bfloat = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    const auto* raw = outputs.front().GetTensorData<uint16_t>();
    std::vector<float> output(info.GetElementCount());
    for (size_t index = 0; index < output.size(); ++index) {
      output[index] = bfloat ? float(Ort::BFloat16_t::FromBits(raw[index])) : float(Ort::Float16_t::FromBits(raw[index]));
      if (!std::isfinite(output[index])) throw std::runtime_error("Nonfinite output");
    }
    std::ofstream output_file(std::string(argv[3]) + ".output.f32", std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    for (int warmup = 0; warmup < std::stoi(argv[4]); ++warmup) run();
    std::vector<double> samples;
    for (int repeat = 0; repeat < std::stoi(argv[5]); ++repeat) samples.push_back(Time(run));
    const size_t after_steady = Used();
    outputs.clear();
    session.reset();
    const size_t after_destroy = Used();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0));
    int runtime, driver;
    Check(cudaRuntimeGetVersion(&runtime));
    Check(cudaDriverGetVersion(&driver));
    std::ofstream result(argv[3]);
    result << std::setprecision(12) << "{\"ort_version\":\"" << OrtGetApiBase()->GetVersionString()
           << "\",\"device\":\"" << properties.name << "\",\"uuid_hex\":\"";
    for (unsigned char byte : properties.uuid.bytes) result << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    result << std::dec << "\",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
          << ",\"feed_setup_ms\":" << feed_setup_ms << ",\"peak_host_rss_bytes\":" << usage.ru_maxrss * 1024ULL
           << ",\"init_ms\":" << init_ms << ",\"first_run_ms\":" << first_ms
           << ",\"device_used_bytes\":{\"before_feeds\":" << before_feeds << ",\"before_init\":" << before_init
           << ",\"after_init\":" << after_init << ",\"after_first\":" << after_first
           << ",\"after_steady\":" << after_steady << ",\"after_destroy\":" << after_destroy << "},\"steady_ms\":[";
    for (size_t index = 0; index < samples.size(); ++index) result << (index ? "," : "") << samples[index];
    result << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}