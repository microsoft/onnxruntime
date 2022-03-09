// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>
#if defined (OPENVINO_2022_1)
#include <openvino/core/core.hpp>
#include <openvino/runtime/runtime.hpp>
#endif 

namespace onnxruntime {
namespace openvino_ep {
class OVCore;
class OVInferRequest; 
class OVExeNetwork; 
 
#if defined (OPENVINO_2022_1) 
  typedef InferenceEngine::Precision OVPrecision;
  typedef ov::RemoteContext OVRemoteContext;
  typedef ov::Tensor OVTensor;
  typedef ov::ProfilingInfo OVProfilingInfo;
  typedef ov::AnyMap OVConfig;
  typedef ov::Model OVNetwork;
#else 
  typedef InferenceEngine::Precision OVPrecision;
  typedef InferenceEngine::RemoteContext::Ptr OVRemoteContext;
  typedef InferenceEngine::Blob OVTensor;
  typedef InferenceEngine::InferenceEngineProfileInfo OVProfilingInfo;
  typedef std::map<std::string, std::string> OVConfig;
  typedef InferenceEngine::CNNNetwork OVNetwork;
#endif 
  typedef std::shared_ptr<OVInferRequest> OVInferRequestPtr;
  typedef std::shared_ptr<OVRemoteContext> OVRemoteContextPtr;
  typedef std::shared_ptr<OVTensor> OVTensorPtr; 
  class OVCore {
  #if defined (OPENVINO_2022_1)
    ov::Core oe;
  #else
    InferenceEngine::Core oe;
  #endif
    public:
        std::shared_ptr<OVNetwork> ReadModel(const std::string& model_stream) const;             
        OVExeNetwork LoadNetwork(std::shared_ptr<OVNetwork>& ie_cnn_network, std::string& hw_target, OVConfig config, std::string name);
        OVExeNetwork ImportModel(const std::string& compiled_blob, std::string hw_target, std::string name);
        void SetCache(std::string cache_dir_path);
        OVExeNetwork LoadNetwork(const std::shared_ptr<const OVNetwork>& model, const OVRemoteContext& context, std::string& name);
        std::vector<std::string> GetAvailableDevices(); 
        #if defined (OPENVINO_2022_1)
        ov::Core& Get() {
            return oe;
        }
        #else 
            InferenceEngine::Core& Get() {
            return oe;
        }
        #endif    
    };

    class OVExeNetwork {
    #if (defined OPENVINO_2022_1)
      ov::CompiledModel obj;
    #else
      InferenceEngine::ExecutableNetwork obj;
    #endif
    public:
    #if defined (OPENVINO_2022_1)
        OVExeNetwork(ov::CompiledModel md) { obj = md; }
        OVExeNetwork() { obj = ov::CompiledModel(); } 
        ov::CompiledModel& Get() { return obj; }
    #else 
        OVExeNetwork(InferenceEngine::ExecutableNetwork md) { obj = md; }
        OVExeNetwork() { obj = InferenceEngine::ExecutableNetwork(); }
        InferenceEngine::ExecutableNetwork& Get() { return obj ; }
    #endif
        OVInferRequest CreateInferRequest();
    };

    class OVInferRequest {
    #if defined (OPENVINO_2022_1)
        ov::InferRequest ovInfReq;
    #else
        InferenceEngine::InferRequest infReq;
    #endif
    public:
        OVTensorPtr GetTensor(std::string& name);
        void SetTensor(std::string& name, OVTensorPtr& blob); 
        void StartAsync();
        void Wait();
        void QueryStatus();
    #if defined (OPENVINO_2022_1)
        explicit OVInferRequest(ov::InferRequest obj) { ovInfReq = obj; }
        OVInferRequest() { ovInfReq = ov::InferRequest(); } 
        ov::InferRequest& GetNewObj() {
        return ovInfReq;
        }
    #else 
        explicit OVInferRequest(InferenceEngine::InferRequest obj) { infReq = obj; }
        OVInferRequest() { infReq = InferenceEngine::InferRequest(); }
        InferenceEngine::InferRequest& GetObj() {
        return infReq;
    }
    #endif
    };
   }
}