// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

#include "openvino/openvino.hpp"
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

class OVCore {
  ov::Core oe;

 public:
  std::shared_ptr<OVNetwork> ReadModel(const std::string& model_stream, const std::string& model_path) const;
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            std::string name);
  OVExeNetwork CompileModel(const std::string onnx_model_path,
                            std::string& hw_target,
                            std::string precision,
                            std::string cache_dir,
                            ov::AnyMap& device_config,
                            std::string name);
  OVExeNetwork ImportModel(std::shared_ptr<std::istringstream> model_stream,
                           std::string& hw_target,
                           ov::AnyMap& device_config,
                           std::string name);
#ifdef IO_BUFFER_ENABLED
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& model, OVRemoteContextPtr context, std::string& name);
  OVExeNetwork ImportModel(std::shared_ptr<std::istringstream> model_stream, OVRemoteContextPtr context, std::string& name);
#endif
  std::vector<std::string> GetAvailableDevices();
  void SetCache(std::string cache_dir_path, std::string device_type);
  ov::Core& Get() { return oe; }
  void SetStreams(const std::string& device_type, int num_streams);
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
  OVTensorPtr GetTensor(const std::string& name);
  void SetTensor(const std::string& name, OVTensorPtr& blob);
  void StartAsync();
  void Infer();
  void WaitRequest();
  void QueryStatus();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(obj) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetNewObj() {
    return ovInfReq;
  }
};
}  // namespace openvino_ep
}  // namespace onnxruntime
