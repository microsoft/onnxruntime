/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "hailo/hailort.hpp"
#include "hailo_execution_provider.h"

namespace onnxruntime {

using hailort::VDevice;
using hailort::Hef;
using hailort::ConfiguredNetworkGroup;
using hailort::ActivatedNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::MemoryView;
using hailort::InferVStreams;

class HailoKernel final : public OpKernel {
public:
    HailoKernel(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
    virtual ~HailoKernel();

private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HailoKernel);

    std::unique_ptr<Hef> create_hef_from_memory(const void* binary_hef, size_t size);
    std::shared_ptr<ConfiguredNetworkGroup> configure_network_group(VDevice &vdevice);
    std::unique_ptr<InferVStreams> create_vstreams_pipeline(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes,
        ConstPointerContainer<std::vector<NodeArg*>> &output_nodes, std::vector<int64_t> input_quantized_params,
        std::vector<int64_t> input_order_params, std::vector<int64_t> output_quantized_params, std::vector<int64_t> output_order_params);
    void update_output_params(ConstPointerContainer<std::vector<NodeArg*>> &output_nodes,
        std::map<std::string, hailo_vstream_params_t> &output_params, std::vector<int64_t> quantized_params, std::vector<int64_t> order_params);
    void update_input_params(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes,
    std::map<std::string, hailo_vstream_params_t> &input_params, std::vector<int64_t> quantized_params, std::vector<int64_t> order_params);
    hailo_status infer(OpKernelContext* context) const;

    mutable std::mutex m_mutex;
    std::shared_ptr<VDevice> m_vdevice;
    std::unique_ptr<Hef> m_hef;
    std::shared_ptr<ConfiguredNetworkGroup> m_network_group;
    std::unique_ptr<InferVStreams> m_pipeline;
    std::vector<std::string> m_sorted_outputs_names;
    std::vector<std::string> m_sorted_inputs_names;

    // TODO: HRT-5221 Support NCHW transformations
    // Transforming the data from/to Hailo default format order (transformation from other format order implemented at Hailo to NHWC)
    std::vector<bool> m_input_should_double_order_conversion;
    std::vector<bool> m_output_should_double_order_conversion;
};

}  // namespace onnxruntime