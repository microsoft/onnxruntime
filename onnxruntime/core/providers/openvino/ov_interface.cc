// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "ov_interface.h"
#include <fstream>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"

#if defined (OV_API_20)
using Exception = ov::Exception;
#elif defined (OPENVINO_2021_4) 
using Exception = InferenceEngine::Exception;
using WaitMode = InferenceEngine::InferRequest::WaitMode;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
using WaitMode = InferenceEngine::IInferRequest::WaitMode;
#endif

namespace onnxruntime {
    namespace openvino_ep {

    const std::string log_tag = "[OpenVINO-EP] ";
    std::shared_ptr<OVNetwork> OVCore::ReadModel(const std::string& model) const {
        try {
            #if defined (OV_API_20)
            OVTensor weights;
            return oe.read_model(model, weights);
            #else
            OVTensorPtr blob = {nullptr};
            auto network = oe.ReadNetwork(model, blob);
            auto obj = std::make_shared<OVNetwork>(network);
            return obj;
            #endif
            } catch (const Exception& e) {
                ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
            } catch (...) {
                ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
            }
    }
            
    OVExeNetwork OVCore::LoadNetwork(std::shared_ptr<OVNetwork>& ie_cnn_network, std::string& hw_target, OVConfig config, std::string name) {
        try {
            #if defined (OV_API_20)
                auto obj = oe.compile_model(ie_cnn_network, hw_target, config);
                OVExeNetwork exe(obj);
                return exe;
            #else 
                auto obj = oe.LoadNetwork(*ie_cnn_network, hw_target, config);
                OVExeNetwork exe(obj);
                return exe;
            #endif     
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
        }    
    }

    OVExeNetwork OVCore::ImportModel(const std::string& compiled_blob, std::string hw_target, std::string name) {
        try {
            #if defined (OV_API_20)
            std::ifstream blob_stream_obj(compiled_blob); 
            auto obj = oe.import_model(blob_stream_obj, hw_target, {});
            return OVExeNetwork(obj);
            #else
            auto obj = oe.ImportNetwork(compiled_blob, hw_target, {});
            return OVExeNetwork(obj);
            #endif
        } catch (Exception &e) {
            ORT_THROW(log_tag + " Exception while Importing Network for graph: " + name + ": " + e.what());
        } catch(...) {
            ORT_THROW(log_tag + " Exception while Importing Network for graph: " + name);
        }
    }

    void OVCore::SetCache(std::string cache_dir_path) {
        #if defined(OV_API_20)
        oe.set_property(ov::cache_dir(cache_dir_path));
        #else
        oe.SetConfig({{CONFIG_KEY(CACHE_DIR), cache_dir_path}});
        #endif
    }

    #ifdef IO_BUFFER_ENABLED
    OVExeNetwork OVCore::LoadNetwork(std::shared_ptr<OVNetwork>& model, OVRemoteContextPtr context, std::string& name) {
        try {
            #if defined(OV_API_20)
            auto obj = oe.compile_model(model, *context);
            return OVExeNetwork(obj);
            #else
            auto obj = oe.LoadNetwork(*model, context);
            return OVExeNetwork(obj);
            #endif
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
        }    
    }
    #endif

    std::vector<std::string> OVCore::GetAvailableDevices() {
        #if defined (OV_API_20)
            auto obj = oe.get_available_devices();
            return obj;
        #else 
            auto obj = oe.GetAvailableDevices();
            return obj;
        #endif
    }
 
    OVInferRequest OVExeNetwork::CreateInferRequest() {
        try {
            #if defined (OV_API_20)
                auto infReq = obj.create_infer_request();
                OVInferRequest inf_obj(infReq);
                return inf_obj;
            #else 
                auto infReq = obj.CreateInferRequest();
                OVInferRequest inf_obj(infReq);
                return inf_obj;
            #endif 
        } catch (const Exception& e) {
            ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + "Exception while creating InferRequest object.");
        }
    }
   
    OVTensorPtr OVInferRequest::GetTensor(std::string& input_name) {
        try {
          #if defined (OV_API_20)
          auto tobj = ovInfReq.get_tensor(input_name);
          OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
          return blob;
          #else 
          auto blob = infReq.GetBlob(input_name);
          return blob;
          #endif 
        } catch (const Exception& e) {
          ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
        } catch (...) {
          ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
        }
    }

    void OVInferRequest::SetTensor(std::string& name, OVTensorPtr& blob) {
        try {
          #if defined(OV_API_20)
          ovInfReq.set_tensor(name, *(blob.get()));
          #else
          infReq.SetBlob(name, blob);
          #endif 
        } catch (const Exception& e) {
          ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name + e.what());
        } catch (...) {
          ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name);
        }
    }

    void OVInferRequest::StartAsync() {
        try {
            #if defined (OV_API_20)
            ovInfReq.start_async();
            #else
            infReq.StartAsync();
            #endif 
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " In Error Couldn't start Inference");
        }
    }

    void OVInferRequest::WaitRequest() {
        try {
            #if defined (OV_API_20)
                ovInfReq.wait();
            #else
                infReq.Wait(WaitMode::RESULT_READY); 
            #endif
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Wait Model Failed: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Wait Mode Failed");
        }     
    }

    void OVInferRequest::QueryStatus() {
        #if defined (OV_API_20)
        std::cout << "ovInfReq.query_state()" << " ";
        #else 
        std::cout << &infReq << " "; 
        #endif 
    }
  }
}
