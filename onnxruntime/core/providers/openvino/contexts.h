// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <memory>
#include "core/common/common.h"
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

namespace fs = std::filesystem;

class SharedContext : public WeakSingleton<SharedContext> {
  // Keep the core alive as long as the shared SharedContext are alive.
  std::shared_ptr<OVCore> OVCore_;

 public:
  SharedContext() : OVCore_(OVCore::Get()) {}
  struct SharedWeights {
    struct Metadata {
      struct Key {
        std::string name;
        bool operator==(const Key&) const = default;
      };
      struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
          return std::hash<std::string>()(key.name);
        }
      };
      struct Value {
        std::string location;
        unsigned int data_offset;
        unsigned int size;
        std::vector<size_t> dimensions;
        std::int32_t element_type;
        std::shared_ptr<ov::Tensor> tensor;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      friend std::ostream& operator<<(std::ostream& right, const Metadata::Map& metadata);
      friend std::istream& operator>>(std::istream& right, Metadata::Map& metadata);
    };

    struct WeightsFile {
      ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightsFile);
      WeightsFile() = delete;
      explicit WeightsFile(std::filesystem::path filename);

      void load_weights(size_t file_offset, void* data, size_t size);

     private:
      std::ifstream file_;
      size_t weights_size_;
    };

    fs::path external_weight_filename;
    std::unique_ptr<WeightsFile> mapped_weights;
    Metadata::Map metadata;
  } shared_weights;
};

using config_t = std::map<std::string, ov::AnyMap>;

struct ProviderInfo {
  std::string device_type{""};             // [device_type]: Overrides the accelerator hardware type and
                                           // precision with these values at runtime.
  std::string precision{""};               // [precision]: Sets the inference precision for execution.
                                           // Supported precision for devices are
                                           // CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                           // Not setting precision will execute with optimized precision for
                                           // best inference latency. set Precision=ACCURACY for executing
                                           // models with input precision for best accuracy.
  uint32_t num_of_threads{0};              // [num_of_threads]: Overrides the accelerator default value of
                                           // number of threads with this value at runtime.
  config_t load_config{};                  // JSON config map to load custom OV parameters.
  fs::path cache_dir{""};                  // [cache_dir]: specify the path to
                                           // dump and load the blobs for the model caching/kernel caching
                                           // (GPU) feature. If blob files are already present,
                                           // it will be directly loaded.
  std::string model_priority{"DEFAULT"};   // High-level OpenVINO model priority hint
                                           // Defines what model should be provided with more performant
                                           // bounded resource first
  uint32_t num_streams{1};                 // [num_streams]: Option that specifies the number of parallel
                                           // inference requests to be processed on a given `device_type`.
                                           // Overrides the accelerator default value of number of streams
                                           // with this value at runtime.
  void* context{nullptr};                  // OpenCL context
  bool enable_opencl_throttling{false};    // [enable_opencl_throttling]: Enables OpenCL queue throttling for
                                           // GPU device (Reduces CPU Utilization when using GPU)
  bool disable_dynamic_shapes{false};      // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to
                                           // static shape at runtime and execute.
  bool enable_qdq_optimizer{false};        // Enables QDQ pruning for efficient inference latency with NPU
  bool so_context_enable{false};           // ORT session option
  bool so_disable_cpu_ep_fallback{false};  // ORT session option
  bool so_context_embed_mode{false};       // ORT session option
  bool so_share_ep_contexts{false};        // ORT session option
  fs::path so_context_file_path{};         // ORT session option
};

// Holds context applicable to the entire EP instance.
struct SessionContext : ProviderInfo {
  SessionContext(const ProviderInfo& info) : ProviderInfo{info} {}
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::filesystem::path onnx_model_path_name;
  uint32_t onnx_opset_version{0};
  mutable bool is_wholly_supported_graph = false;  // Value is set to mutable to modify from capability
  mutable bool has_external_weights = false;       // Value is set to mutable to modify from capability
  const std::vector<uint32_t> OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR};
  const std::string openvino_sdk_version = std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR);
};

// Holds context specific to subgraph.
struct SubGraphContext {
  using string_index_map_t = std::unordered_map<std::string, uint32_t>;
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_npu_config = false;
  bool is_constant = false;
  void* context = 0;
  std::string subgraph_name;
  string_index_map_t input_names;
  string_index_map_t output_names;
  std::string model_precision;
  bool is_ep_ctx_graph = false;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
