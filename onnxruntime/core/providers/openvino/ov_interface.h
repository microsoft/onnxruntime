// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <utility>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/frontend/manager.hpp"

#ifdef IO_BUFFER_ENABLED
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif

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

#ifdef IO_BUFFER_ENABLED
typedef ov::intel_gpu::ocl::ClContext* OVRemoteContextPtr;
typedef ov::RemoteContext OVRemoteContext;
#endif

struct OVCore {
  static void Initialize();
  static void Teardown();

  // OV Interface For Reading Model
  static std::shared_ptr<OVNetwork> ReadModel(const std::string& model_stream, const std::string& model_path);

  // OV Interface for Compiling OV Model Type
  static OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                   std::string& hw_target,
                                   ov::AnyMap& device_config,
                                   const std::string& name);
  // OV Interface for Fast Compile
  static OVExeNetwork CompileModel(const std::string& onnx_model,
                                   std::string& hw_target,
                                   ov::AnyMap& device_config,
                                   const std::string& name);
  // OV Interface for Import model Stream
  static OVExeNetwork ImportModel(std::istream& model_stream,
                                  std::string hw_target,
                                  const ov::AnyMap& device_config,
                                  std::string name);
#ifdef IO_BUFFER_ENABLED
  static OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& model,
                                   OVRemoteContextPtr context,
                                   std::string name);
  static OVExeNetwork ImportModel(std::shared_ptr<std::istringstream> model_stream,
                                  OVRemoteContextPtr context,
                                  std::string name);
#endif
  static std::vector<std::string> GetAvailableDevices();
  static void SetCache(const std::string& cache_dir_path);
  inline static ov::Core& Get();
  static void SetStreams(const std::string& device_type, int num_streams);
};

class OVExeNetwork {
  ov::CompiledModel obj;

 public:
  explicit OVExeNetwork(ov::CompiledModel md) : obj(md) {}
  OVExeNetwork() : obj(ov::CompiledModel()) {}
  ov::CompiledModel& Get() { return obj; }
  OVInferRequest CreateInferRequest();
};

class OVInferRequest {
  ov::InferRequest ovInfReq;

 public:
  uint32_t GetNumInputs();
  OVTensorPtr GetTensor(const std::string& name);
  std::string GetInputTensorName(uint32_t index);
  void SetTensor(const std::string& name, OVTensorPtr& blob);
  void StartAsync();
  void Infer();
  void WaitRequest();
  void QueryStatus();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(std::move(obj)) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetNewObj() {
    return ovInfReq;
  }
};
}  // namespace openvino_ep
}  // namespace onnxruntime
