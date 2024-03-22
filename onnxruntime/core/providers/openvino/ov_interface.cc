// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_interface.h"
#include <fstream>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "backend_utils.h"

using Exception = ov::Exception;

namespace onnxruntime {
namespace openvino_ep {

const std::string log_tag = "[OpenVINO-EP] ";
std::shared_ptr<OVNetwork> OVCore::ReadModel(const std::string& model, const std::string& model_path) const {
  try {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream, model_path};

    FE = manager.load_by_model(params);
    if (FE) {
      inputModel = FE->load(params);
    }
    return FE->convert(inputModel);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
  }
}

OVExeNetwork OVCore::LoadNetwork(std::shared_ptr<OVNetwork>& ie_cnn_network,
                                 std::string& hw_target,
                                 ov::AnyMap& device_config,
                                 std::string name) {
  ov::CompiledModel obj;
  try {
    obj = oe.compile_model(ie_cnn_network, hw_target, device_config);

#ifndef NDEBUG
    if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
      // output of the actual settings that the device selected
      auto supported_properties = obj.get_property(ov::supported_properties);
      std::cout << "Model:" << std::endl;
      for (const auto& cfg : supported_properties) {
        if (cfg == ov::supported_properties)
          continue;
        auto prop = obj.get_property(cfg);
        if (cfg == ov::device::properties) {
          auto devices_properties = prop.as<ov::AnyMap>();
          for (auto& item : devices_properties) {
            std::cout << "  " << item.first << ": " << std::endl;
            for (auto& item2 : item.second.as<ov::AnyMap>()) {
              OPENVINO_SUPPRESS_DEPRECATED_START
              if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                  item2.first == "SUPPORTED_METRICS")
                continue;
              OPENVINO_SUPPRESS_DEPRECATED_END
              std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
            }
          }
        } else {
          std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
        }
      }
    }
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

OVExeNetwork OVCore::LoadNetwork(const std::string onnx_model_path,
                                 std::string& hw_target,
                                 ov::AnyMap& device_config,
                                 std::string name) {
  ov::CompiledModel obj;
  try {
    obj = oe.compile_model(onnx_model_path, hw_target, device_config);
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

void OVCore::SetCache(std::string cache_dir_path) {
  oe.set_property(ov::cache_dir(cache_dir_path));
}

#ifdef IO_BUFFER_ENABLED
OVExeNetwork OVCore::LoadNetwork(std::shared_ptr<OVNetwork>& model, OVRemoteContextPtr context, std::string& name) {
  try {
    auto obj = oe.compile_model(model, *context);
    return OVExeNetwork(obj);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}
#endif

std::vector<std::string> OVCore::GetAvailableDevices() {
  auto available_devices = oe.get_available_devices();
  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  oe.set_property(device_type, {ov::num_streams(num_streams)});
}

OVInferRequest OVExeNetwork::CreateInferRequest() {
  try {
    auto infReq = obj.create_infer_request();
    OVInferRequest inf_obj(infReq);
    return inf_obj;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object.");
  }
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  try {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
  }
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  try {
    ovInfReq.set_tensor(name, *(blob.get()));
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name);
  }
}

void OVInferRequest::StartAsync() {
  try {
    ovInfReq.start_async();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::Infer() {
  try {
    ovInfReq.infer();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::WaitRequest() {
  try {
    ovInfReq.wait();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Wait Model Failed: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Wait Mode Failed");
  }
}

void OVInferRequest::QueryStatus() {
  std::cout << "ovInfReq.query_state()"
            << " ";
}
}  // namespace openvino_ep
}  // namespace onnxruntime
