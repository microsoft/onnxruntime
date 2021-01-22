// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include<vector>
#include "data_ops.h"

namespace onnxruntime {
namespace openvino_ep {

class GetCapability {
    private:
        const GraphViewer& graph_viewer;
        std::string device_type;
        version_id_e version_id; 
        Capability* cobj;
    public:
        GetCapability (const GraphViewer& graph_viewer_param, std::string device_type_param);   
        virtual void set_version_id(const std::string version_param);     
        virtual std::vector<std::unique_ptr<ComputeCapability>> Execute(Capability &cobj_param);
};

#if (defined OPENVINO_2020_2) || (defined OPENVINO_2020_3)
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_2(const GraphViewer& graph_viewer, const std::string device_type);
#endif

}  //namespace openvino_ep
}  //namespace onnxruntime
