// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <utility>
#include <optional>
#include <algorithm>
#include <unordered_map>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"

#include <string>

namespace onnxruntime {
namespace openvino_ep {
class OVCore;
class OVInferRequest;
class OVExeNetwork;

typedef ov::Tensor OVTensor;
typedef ov::ProfilingInfo OVProfilingInfo;
typedef ov::Model OVNetwork;
typedef std::shared_ptr<OVInferRequest> OVInferRequestPtr;
typedef std::shared_ptr<OVTensor> OVTensorPtr;

std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type);

template <typename T>
class WeakSingleton {
 public:
  static std::shared_ptr<T> Get() {
    static std::weak_ptr<T> instance;
    static std::mutex mutex;

    auto ptr = instance.lock();
    if (!ptr) {
      std::lock_guard<std::mutex> lock(mutex);
      // ensure another thread didn't create an instance while this thread was waiting
      ptr = instance.lock();
      if (!ptr) {
        ptr = std::make_shared<T>();
        instance = ptr;
      }
    }
    return ptr;
  }

 protected:
  WeakSingleton() = default;
  virtual ~WeakSingleton() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeakSingleton);
};

struct OVCore : WeakSingleton<OVCore> {
  ov::Core core;

  // OV Interface For Reading Model
  std::shared_ptr<OVNetwork> ReadModel(std::string&& model_stream, const std::string& model_path);

  OVExeNetwork StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                    std::string& hw_target,
                                    const ov::AnyMap& device_config);
  // OV Interface for Compiling OV Model Type
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            bool enable_causallm,
                            const std::string& name);
  // OV Interface for Fast Compile
  OVExeNetwork CompileModel(const std::string& onnx_model,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            const std::string& name);
  // OV Interface for Import model Stream
  OVExeNetwork ImportModel(std::istream& model_stream,
                           std::string hw_target,
                           const ov::AnyMap& device_config,
                           std::string name);
  OVExeNetwork ImportEPCtxOVIREncapsulation(std::istream& model_stream,
                                            std::string& hw_target,
                                            const ov::AnyMap& device_config,
                                            bool enable_causallm,
                                            std::filesystem::path model_file_path);

  std::vector<std::string> GetAvailableDevices() const;
  std::vector<std::string> GetAvailableDevices(const std::string& device_type) const;
  void SetCache(const std::string& cache_dir_path);
  void SetStreams(const std::string& device_type, int num_streams);
};

class OVExeNetwork {
  ov::CompiledModel compiled_model_obj;
  std::string target_device;
  bool is_stateful_causallm;

 public:
  explicit OVExeNetwork(ov::CompiledModel compiled_model, std::string device, bool stateful_causallm = false)
      : compiled_model_obj(compiled_model), target_device(device), is_stateful_causallm(stateful_causallm) {}
  OVExeNetwork() : compiled_model_obj(ov::CompiledModel()) {}
  ov::CompiledModel& Get() { return compiled_model_obj; }
  std::shared_ptr<OVInferRequest> CreateInferRequest();
};

class OVInferRequest {
  struct ov_tensor_data_t {
    OVTensorPtr tensor_ptr;
    const void* ort_ptr;
  };

 protected:
  ov::InferRequest ovInfReq;
  std::unordered_map<std::string, ov_tensor_data_t> bindings_cache_;

 public:
  uint32_t GetNumInputs();
  OVTensorPtr GetTensor(const std::string& name);
  std::string GetInputTensorName(uint32_t index);

  // Set tensor described param_info and ort_ptr. Overrides shape in param_info with shape_override. Call infer req tensor if ort_ptr is last set.
  void SetTensor(const std::string& name, const ov::element::Type& type, const ov::Shape& shape, void* ort_ptr) {
    auto& cached_binding = bindings_cache_[name];
    if (cached_binding.ort_ptr != ort_ptr) {
      auto tensor_ptr = std::make_shared<ov::Tensor>(type, shape, const_cast<void*>(ort_ptr));
      SetTensor(name, tensor_ptr);
      cached_binding = {tensor_ptr, ort_ptr};
    }
  }

  void SetTensor(const std::string& name, OVTensorPtr& blob);
  virtual void Infer();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(std::move(obj)) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetNewObj() {
    return ovInfReq;
  }
  virtual void RewindKVCache(size_t index) {}
};

class StatefulOVInferRequest : public OVInferRequest {
 public:
  explicit StatefulOVInferRequest(ov::InferRequest infer_request, std::string device);

  void Infer() override;
  void RewindKVCache(size_t index) override;
  void FillTensor(const std::string& tensor_name, const ov::element::Type& type,
                  const std::vector<size_t>& shape, int32_t fill_value);
  void CacheTensor(const std::string& tensor_name, std::vector<int64_t>& cache);
  void SetTensorFromCache(const std::string& tensor_name, const std::vector<int64_t>& cache_data);
  std::optional<ov::Tensor> FindTensor(const std::string& tensor_name);

 private:
  void PreProcessInferRequest();
  std::string target_device;

  // If prefill_use_full_chat_history is true, cache the "input_ids" & "position_ids" tensors,
  // and ensure that full chat history is passed for each prefill call.
  bool prefill_use_full_chat_history = false;
  std::vector<int64_t> cached_input_ids;
  std::vector<int64_t> cached_position_ids;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
