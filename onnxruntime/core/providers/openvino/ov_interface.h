// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>
#if defined (OPENVINO_2022_1)
#define OV_API_20
#include "openvino/openvino.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#endif 

#ifdef IO_BUFFER_ENABLED
#include <gpu/gpu_context_api_ocl.hpp>
#include <gpu/gpu_config.hpp>
#if defined (OV_API_20)
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif
#endif

namespace onnxruntime {
namespace openvino_ep {
class OVCore;
class OVInferRequest; 
class OVExeNetwork; 
 
#if defined (OV_API_20) 
  typedef InferenceEngine::Precision OVPrecision;
  typedef ov::Tensor OVTensor;
  typedef ov::ProfilingInfo OVProfilingInfo;
  typedef ov::AnyMap OVConfig;
  typedef ov::Model OVNetwork;
#else 
  typedef InferenceEngine::Precision OVPrecision;
  typedef InferenceEngine::Blob OVTensor;
  typedef InferenceEngine::InferenceEngineProfileInfo OVProfilingInfo;
  typedef std::map<std::string, std::string> OVConfig;
  typedef InferenceEngine::CNNNetwork OVNetwork;
#endif 
  typedef std::shared_ptr<OVInferRequest> OVInferRequestPtr;
  typedef std::shared_ptr<OVTensor> OVTensorPtr;

#ifdef IO_BUFFER_ENABLED
  #ifdef OV_API_20
  typedef ov::intel_gpu::ocl::ClContext* OVRemoteContextPtr;
  typedef ov::RemoteContext OVRemoteContext; 
  #else
  typedef InferenceEngine::RemoteContext::Ptr OVRemoteContextPtr;
  typedef InferenceEngine::RemoteContext OVRemoteContext;
  #endif 
#endif   

  class OVCore {
  #if defined (OV_API_20)
    ov::Core oe;
  #else
    InferenceEngine::Core oe;
  #endif
    public:
        std::shared_ptr<OVNetwork> ReadModel(const std::string& model_stream) const;             
        OVExeNetwork LoadNetwork(std::shared_ptr<OVNetwork>& ie_cnn_network, std::string& hw_target, OVConfig config, std::string name);
        OVExeNetwork ImportModel(const std::string& compiled_blob, std::string hw_target, std::string name);
        void SetCache(std::string cache_dir_path);
        #ifdef IO_BUFFER_ENABLED
        OVExeNetwork LoadNetwork(std::shared_ptr<OVNetwork>& model, OVRemoteContextPtr context, std::string& name);
        #endif 
        std::vector<std::string> GetAvailableDevices(); 
        #if defined (OV_API_20)
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
    #if (defined OV_API_20)
      ov::CompiledModel obj;
    #else
      InferenceEngine::ExecutableNetwork obj;
    #endif
    public:
    #if defined (OV_API_20)
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
    #if defined (OV_API_20)
        ov::InferRequest ovInfReq;
    #else
        InferenceEngine::InferRequest infReq;
    #endif
    public:
        OVTensorPtr GetTensor(std::string& name);
        void SetTensor(std::string& name, OVTensorPtr& blob);
        void StartAsync();
        void WaitRequest();
        void QueryStatus();
    #if defined (OV_API_20)
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