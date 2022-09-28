/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "hailo_op.h"
#include "utils.h"

#include <iostream>
#include <mutex>

namespace onnxruntime {

static constexpr const char* HEF_ATTRIBUTE = "hef";
static constexpr const char* SORTED_INPUT_NAMES_ATTRUBUTE = "sorted_input_names";
static constexpr const char* SORTED_OUTPUT_NAMES_ATTRUBUTE = "sorted_output_names";
static constexpr const char* INPUT_QUANTIZED_ATTRIBUTE = "input_quantized";
static constexpr const char* INPUT_ORDER_ATTRIBUTE = "input_format_order";
static constexpr const char* OUTPUT_QUANTIZED_ATTRIBUTE = "output_quantized";
static constexpr const char* OUTPUT_ORDER_ATTRIBUTE = "output_format_order";

HailoKernel::HailoKernel(const OpKernelInfo& info) : OpKernel(info), m_mutex()
{
    std::string binary_hef;
    auto status = info.GetAttr(HEF_ATTRIBUTE, &binary_hef);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  HEF_ATTRIBUTE, "' is not set");

    status = info.GetAttrs(SORTED_INPUT_NAMES_ATTRUBUTE, m_sorted_inputs_names);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  SORTED_INPUT_NAMES_ATTRUBUTE, "' is not set");

    status = info.GetAttrs(SORTED_OUTPUT_NAMES_ATTRUBUTE, m_sorted_outputs_names);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  SORTED_OUTPUT_NAMES_ATTRUBUTE, "' is not set");

    std::vector<int64_t> input_quantized;
    status = info.GetAttrs(INPUT_QUANTIZED_ATTRIBUTE, input_quantized);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  INPUT_QUANTIZED_ATTRIBUTE, "' is not set");

    std::vector<int64_t> input_format_order;
    status = info.GetAttrs(INPUT_ORDER_ATTRIBUTE, input_format_order);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  INPUT_ORDER_ATTRIBUTE, "' is not set");

    std::vector<int64_t> output_quantized;
    status = info.GetAttrs(OUTPUT_QUANTIZED_ATTRIBUTE, output_quantized);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  OUTPUT_QUANTIZED_ATTRIBUTE, "' is not set");

    std::vector<int64_t> output_format_order;
    status = info.GetAttrs(OUTPUT_ORDER_ATTRIBUTE, output_format_order);
    HAILO_ORT_ENFORCE(status.IsOK(), "attribute '",  OUTPUT_ORDER_ATTRIBUTE, "' is not set");

    m_hef = create_hef_from_memory(binary_hef.c_str(), binary_hef.length());

    hailo_vdevice_params_t params;
    auto hailo_status = hailo_init_vdevice_params(&params);
    HAILO_ORT_ENFORCE(HAILO_SUCCESS == hailo_status, "Failed init vdevice_params, status = ", hailo_status);
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.group_id = "SHARED";
    auto expected_vdevice = VDevice::create(params);
    HAILO_CHECK_EXPECTED(expected_vdevice, "Failed to create VDevice");
    m_vdevice = std::move(expected_vdevice.value());

    m_network_group = configure_network_group(*m_vdevice.get());

    auto output_nodes = info.node().OutputDefs();
    auto input_nodes = info.node().InputDefs();
    m_pipeline = create_vstreams_pipeline(input_nodes, output_nodes, input_quantized, input_format_order, output_quantized, output_format_order);
}

HailoKernel::~HailoKernel()
{
    m_pipeline.reset();
    m_network_group.reset();
    m_vdevice.reset();
    m_hef.reset();
}

std::unique_ptr<Hef> HailoKernel::create_hef_from_memory(const void* binary_hef, size_t size)
{
    auto hef_memory_view = MemoryView::create_const(binary_hef, size);
    auto hef = Hef::create(hef_memory_view);
    HAILO_CHECK_EXPECTED(hef, "Create Hef from memory failed");

    return std::make_unique<Hef>(hef.release());
}

std::shared_ptr<ConfiguredNetworkGroup> HailoKernel::configure_network_group(VDevice &vdevice)
{
    auto configure_params = m_hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    HAILO_CHECK_EXPECTED(configure_params, "Creating configure params failed");
    auto network_groups = vdevice.configure(*m_hef.get(), configure_params.value());
    HAILO_CHECK_EXPECTED(network_groups, "Configure network group failed");
    HAILO_ORT_ENFORCE(1 == network_groups->size(), "Multiple network group is not supported, got = ", network_groups->size());

    return std::move(network_groups->at(0));
}

void HailoKernel::update_output_params(ConstPointerContainer<std::vector<NodeArg*>> &output_nodes,
    std::map<std::string, hailo_vstream_params_t> &output_params, std::vector<int64_t> quantized_params, std::vector<int64_t> format_order_params)
{
    HAILO_ORT_ENFORCE(output_nodes.size() == m_sorted_outputs_names.size(),
        "Number of output nodes = ", output_nodes.size(), ", is inconsistent with output vstreams number = ", m_sorted_outputs_names.size());

    for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
        auto output_ort_dtype = output_nodes[i]->TypeAsProto()->tensor_type().elem_type();
        auto output_hailo_dtype = HailoUtils::convert_ort_to_hailo_dtype(output_ort_dtype);

        auto vstream_name = m_sorted_outputs_names[i];
        output_params[vstream_name].user_buffer_format.type = output_hailo_dtype;

        // We transform NCHW->NHWC / NHWC->NCHW in 'HailoKernel::infer()', and transform NHWC->device-format-order in libhailort
        // TODO: remove the double transformation when implementing transformations from/to NCHW
        if (hailo_format_order_t(format_order_params[i]) == HAILO_FORMAT_ORDER_NCHW) {
            output_params[vstream_name].user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
            m_output_should_double_order_conversion.push_back(true);
        } else {
            m_output_should_double_order_conversion.push_back(false);
        }

        if (quantized_params[i]) {
            output_params[vstream_name].user_buffer_format.flags =
                static_cast<hailo_format_flags_t>(output_params[vstream_name].user_buffer_format.flags | HAILO_FORMAT_FLAGS_QUANTIZED);
        }
    }
}

void HailoKernel::update_input_params(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes,
    std::map<std::string, hailo_vstream_params_t> &input_params, std::vector<int64_t> quantized_params, std::vector<int64_t> format_order_params)
{
    HAILO_ORT_ENFORCE(input_nodes.size() == m_sorted_inputs_names.size(),
        "Number of input nodes = ", input_nodes.size(), ", is inconsistent with input vstreams number = ", m_sorted_inputs_names.size());

    for (size_t i = 0; i < m_sorted_inputs_names.size(); i++) {
        auto input_ort_dtype = input_nodes[i]->TypeAsProto()->tensor_type().elem_type();
        auto input_hailo_dtype = HailoUtils::convert_ort_to_hailo_dtype(input_ort_dtype);

        auto vstream_name = m_sorted_inputs_names[i];
        input_params[vstream_name].user_buffer_format.type = input_hailo_dtype;

        // We transform NCHW->NHWC / NHWC->NCHW in 'HailoKernel::infer()', and transform NHWC->device-format-order in libhailort
        // TODO: remove the double transformation when implementing transformations from/to NCHW
        if (hailo_format_order_t(format_order_params[i]) == HAILO_FORMAT_ORDER_NCHW) {
            input_params[vstream_name].user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
            m_input_should_double_order_conversion.push_back(true);
        } else {
            m_input_should_double_order_conversion.push_back(false);
        }

        if (quantized_params[i]) {
            input_params[vstream_name].user_buffer_format.flags =
                static_cast<hailo_format_flags_t>(input_params[vstream_name].user_buffer_format.flags | HAILO_FORMAT_FLAGS_QUANTIZED);
        }
    }
}

std::unique_ptr<InferVStreams> HailoKernel::create_vstreams_pipeline(ConstPointerContainer<std::vector<NodeArg*>> &input_nodes,
    ConstPointerContainer<std::vector<NodeArg*>> &output_nodes, std::vector<int64_t> input_quantized_params,
    std::vector<int64_t> input_format_order_params, std::vector<int64_t> output_quantized_params, std::vector<int64_t> output_format_order_params)
{
    // Create input VStreams params
    auto input_params_expected = m_network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO, 
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    HAILO_CHECK_EXPECTED(input_params_expected, "Failed to make input vstream params");
    auto input_params = input_params_expected.release();
    update_input_params(input_nodes, input_params, input_quantized_params, input_format_order_params);

    // Create output VStreams params
    auto output_params_expected = m_network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    HAILO_CHECK_EXPECTED(output_params_expected, "Failed to make output vstream params");
    auto output_params = output_params_expected.release();
    update_output_params(output_nodes, output_params, output_quantized_params, output_format_order_params);

    auto infer_vstream = InferVStreams::create(*m_network_group, input_params, output_params);
    HAILO_CHECK_EXPECTED(infer_vstream, "Failed to create InferVstreams");

    return std::make_unique<InferVStreams>(infer_vstream.release());
}

hailo_status HailoKernel::infer(OpKernelContext* context) const
{
    // Create inputs datasetmap
    std::map<std::string, MemoryView> input_data_mem_views;
    std::map<std::string, std::vector<uint8_t>> input_buffers;
    auto input_vstreams = m_pipeline->get_input_vstreams();

    // TODO: HRT-6671 remove this after supportting multiple inputs.
    HAILO_ORT_ENFORCE(1 == input_vstreams.size(), "Multiple inputs is not supported.");

    // Init default frames_count value. Will be set by the input tensor's shape.
    size_t frames_count = 1;
    for (size_t i = 0; i < input_vstreams.size(); i++) {
        const auto* input_tensor = context->Input<Tensor>(i);
        HAILO_ORT_ENFORCE(nullptr != input_tensor, "input ", i, " is missing");

        // TODO: HRT-6671 - When supporting multiple input we need to check that all frames_count in the shapes are the same.
        frames_count = input_tensor->Shape()[0];
        auto vstream_name = input_vstreams[i].get().name();

        if (m_input_should_double_order_conversion[i]) {
            auto input_info = input_vstreams[i].get().get_info();
            auto input_format = input_vstreams[i].get().get_user_buffer_format();
            input_buffers.emplace(vstream_name, std::vector<uint8_t>(input_vstreams[i].get().get_frame_size() * frames_count));
            HailoUtils::transform_NCHW_to_NHWC(input_tensor->DataRaw(), input_buffers[vstream_name].data(), &input_info.shape, input_format.type, frames_count);
            auto input_data = MemoryView::create_const(input_buffers[vstream_name].data(), input_buffers[vstream_name].size());
            input_data_mem_views.emplace(vstream_name, input_data);
        } else {
            auto input_data = MemoryView::create_const(input_tensor->DataRaw(), input_tensor->SizeInBytes());
            input_data_mem_views.emplace(input_vstreams[i].get().name(), input_data);
        }
    }

    // Create outputs dataset map
    std::map<std::string, MemoryView> output_data_mem_views;
    std::map<std::string, std::vector<uint8_t>> output_buffers;
    for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
        auto vstream_name = m_sorted_outputs_names[i];
        auto output_vstream = m_pipeline->get_output_by_name(vstream_name);
        HAILO_ORT_ENFORCE(output_vstream, "Failed getting output vstream: ", vstream_name);

        if (m_output_should_double_order_conversion[i]) {
            output_buffers.emplace(vstream_name, std::vector<uint8_t>(output_vstream->get().get_frame_size() * frames_count));
            output_data_mem_views.emplace(vstream_name, MemoryView(output_buffers[vstream_name].data(), output_buffers[vstream_name].size()));
        } else {
            auto output_info = output_vstream->get().get_info();
            Tensor* output_tensor = context->Output(i, HailoUtils::convert_hailo_shape(frames_count, output_info.shape, output_info.format.order));
            HAILO_ORT_ENFORCE(output_tensor != nullptr, "output ", i, " is missing");
            output_data_mem_views.emplace(vstream_name, MemoryView(output_tensor->MutableDataRaw(), output_tensor->SizeInBytes()));
        }
    }

    std::unique_lock<std::mutex> lock(m_mutex);
    hailo_status status = m_pipeline->infer(input_data_mem_views, output_data_mem_views, frames_count);

    for (size_t i = 0; i < m_sorted_outputs_names.size(); i++) {
        if (!m_output_should_double_order_conversion[i]) {
            continue;
        }
        auto output_vstream = m_pipeline->get_output_by_name(m_sorted_outputs_names[i]);
        HAILO_ORT_ENFORCE(output_vstream, "Failed getting output vstream: ", m_sorted_outputs_names[i]);

        auto output_info = output_vstream->get().get_info();
        auto output_format = output_vstream->get().get_user_buffer_format();
        Tensor* output_tensor = context->Output(i, HailoUtils::convert_hailo_shape(frames_count, output_info.shape, output_info.format.order));
        HAILO_ORT_ENFORCE(output_tensor != nullptr, "output ", i, " is missing");

        HailoUtils::transform_NHWC_to_NCHW((void*)output_data_mem_views[m_sorted_outputs_names[i]].data(),
            output_tensor->MutableDataRaw(), &output_info.shape, output_format.type, frames_count);
    }

    return status;
}

Status HailoKernel::Compute(OpKernelContext* context) const
{
    auto status = infer(context);
    if (HAILO_SUCCESS == status) {
        return Status::OK();
    }
    else {
        return Status(common::ONNXRUNTIME, common::FAIL, "Error happend during inference, hailo status = "
            + std::to_string(status));
    }
}

}  // namespace onnxruntime