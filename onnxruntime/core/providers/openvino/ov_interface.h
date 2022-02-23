// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>
#include <openvino/core/core.hpp>
#include <openvino/runtime/runtime.hpp>

namespace onnxruntime {

namespace openvino_ep {

class ov_core;
class ov_infer_request; 
class ov_exe_network; 
 
    #if defined (OPENVINO_2022_1) 
        typedef InferenceEngine::Precision ov_precision;
        typedef ov::RemoteContext ov_remote_context;
        typedef ov::Tensor ov_tensor;
        typedef ov::ProfilingInfo ov_profiling_info;
        typedef ov::AnyMap ov_config;
        typedef ov::Model ov_network;
    #else 
        typedef InferenceEngine::Precision ov_precision;
        typedef InferenceEngine::RemoteContext::Ptr ov_remote_context;
        typedef InferenceEngine::Blob ov_tensor;
        typedef InferenceEngine::InferenceEngineProfileInfo ov_profiling_info;
        typedef std::map<std::string, std::string> ov_config;
        typedef InferenceEngine::CNNNetwork ov_network;
    #endif 
    
    typedef std::shared_ptr<ov_infer_request> ov_infer_request_ptr;
    typedef std::shared_ptr<ov_remote_context> ov_remote_context_ptr;
    typedef std::shared_ptr<ov_tensor> ov_tensor_ptr; 

    class ov_core {
        #if defined (OPENVINO_2022_1)
            ov::Core oe;
        #else
            InferenceEngine::Core oe;
        #endif
        public:
            std::shared_ptr<ov_network> read_model(const std::string& model_stream) const;             
            ov_exe_network load_network(std::shared_ptr<ov_network>& ie_cnn_network, std::string& hw_target, ov_config config, std::string name);
            ov_exe_network import_model(const std::string& compiled_blob, std::string hw_target, std::string name);
            void set_cache(std::string cache_dir_path);
            ov_exe_network load_network(const std::shared_ptr<const ov_network>& model, const ov_remote_context& context, std::string& name);
            std::vector<std::string> get_available_devices(); 
            #if defined (OPENVINO_2022_1)
                ov::Core& get() {
                    return oe;
                }
            #else 
                InferenceEngine::Core& get() {
                    return oe;
                }
            #endif    
    };

    class ov_exe_network {
        #if (defined OPENVINO_2022_1)
        ov::CompiledModel obj;
        #else
        InferenceEngine::ExecutableNetwork obj;
        #endif
        
        public:
          #if defined (OPENVINO_2022_1)
          ov_exe_network(ov::CompiledModel md) { obj = md; }
          ov_exe_network() { obj = ov::CompiledModel(); } 
          ov::CompiledModel& get() { return obj; }
          #else 
          ov_exe_network(InferenceEngine::ExecutableNetwork md) { obj = md; }
          ov_exe_network() { obj = InferenceEngine::ExecutableNetwork(); }
          void export_network(std::ofstream& model_stream) {
            obj.Export(model_stream); 
          InferenceEngine::ExecutableNetwork& get() { return obj ; }
          }
          #endif
          ov_infer_request create_infer_request();
    };

    class ov_infer_request {
            #if defined (OPENVINO_2022_1)
            ov::InferRequest ovInfReq;
            #else
            InferenceEngine::InferRequest infReq;
            #endif
        public:
            ov_tensor_ptr get_tensor(std::string& name);
            void set_tensor(ov_tensor& blob,std::string& name); 
            void start_async();
            void wait();
            void query_status();
            
            #if defined (OPENVINO_2022_1)
            explicit ov_infer_request(ov::InferRequest obj) { ovInfReq = obj; }
            ov_infer_request() { ovInfReq = ov::InferRequest(); } 
            ov::InferRequest& getNewObj() {
                return ovInfReq;
            }
            #else 
            explicit ov_infer_request(InferenceEngine::InferRequest obj) { infReq = obj; }
            ov_infer_request() { infReq = InferenceEngine::InferRequest(); }
            InferenceEngine::InferRequest& getObj() {
                return infReq;
            }
            #endif
    };

   }
}