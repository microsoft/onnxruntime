// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ncnn-src/src/gpu.h"
#include "ncnn-src/src/pipelinecache.h"
#include "ncnn-src/src/layer.h"

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/memcpy.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/partitioning_utils.h"
#include "core/providers/vulkan/vulkan_execution_provider.h"
#include "core/providers/vulkan/vulkan_allocator.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"
#include "core/providers/vulkan/vulkan_kernel.h"
#include "core/providers/vulkan/vulkan_utils.h"

#include "core/providers/vulkan/activation/activations.h"

namespace onnxruntime {
namespace {
using namespace vulkan;

ncnn::VulkanDevice& GetVulkanDevice() {
  // get_gpu_count/get_default_gpu_index/get_gpu_device all implicitly create the gpu instance if it doesn't exist yet.
  // there is also `create_gpu_instance(const char* driver_path = 0);` if we need/want

  int gpu_count = ncnn::get_gpu_count();
  LOGS_DEFAULT(INFO) << "Vulkan capable GPU count:" << gpu_count;

  if (gpu_count == 0) {
    ORT_THROW("No Vulkan capable GPU detected.");
  }

  auto device_index = ncnn::get_default_gpu_index();  // TODO: Make device id configurable.

  // TODO: info on the available devices is logged with NCNN_LOGE by create_gpu_instance if NCNN_STDIO is defined.
  // we could maybe look at doing that via ORT logging so the ORT log severity applies.
  auto* device = ncnn::get_gpu_device(device_index);
  ORT_ENFORCE(device, "Failed to get Vulkan device.");

  return *device;
}

std::vector<std::unique_ptr<ComputeCapability>> GetCapabilityCompiling(const onnxruntime::GraphViewer& graph_viewer,
                                                                       ModelMetadefIdGenerator metadef_id_generator,
                                                                       const logging::Logger& logger) {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  const auto gen_metadef_name =
      [&]() {
        HashValue model_hash;
        int metadef_id = metadef_id_generator.GenerateId(graph_viewer, model_hash);
        return MakeString(kVulkanExecutionProvider, "_", model_hash, "_", metadef_id);
      };

  const utils::IsNodeSupportedFn is_node_supported_fn = [&](const Node& node) -> bool {
    return VulkanKernel::IsSupported(graph_viewer, node, logger);
  };

  // nothing required currently
  const utils::OnGroupClosedFn on_group_closed_fn = nullptr;

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported_fn, on_group_closed_fn,
                                            gen_metadef_name, "Vulkan", kVulkanExecutionProvider);

  return result;
}
}  // namespace

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& info)
    : IExecutionProvider(kVulkanExecutionProvider,
                         OrtDevice(OrtDevice::CPU,  // input on CPU
                                   OrtDevice::MemType::DEFAULT,
                                   info.device_id)),
      ncnn_options_{},
      vulkan_device_{GetVulkanDevice()},
      weight_staging_allocator_{&vulkan_device_},
      weight_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 8 MB. should it be different or configurable?
      staging_allocator_{&vulkan_device_},
      blob_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 16 MB. should it be different or configurable?
      pipeline_cache_{std::make_unique<ncnn::PipelineCache>(&vulkan_device_)} {
  ncnn_options_.use_vulkan_compute = true;

  ncnn_options_.staging_vkallocator = &weight_staging_allocator_;
  ncnn_options_.blob_vkallocator = &weight_allocator_;
  ncnn_options_.pipeline_cache = pipeline_cache_.get();

  // start with fp32
  // NCNN can automatically convert from fp32 to fp16 if use_fp16_storage/use_fp16_packed is true.
  ncnn_options_.use_fp16_packed = false;
  ncnn_options_.use_fp16_storage = false;
  ncnn_options_.use_fp16_arithmetic = false;
  ncnn_options_.use_int8_packed = false;
  ncnn_options_.use_int8_storage = false;
  ncnn_options_.use_int8_arithmetic = false;
  ncnn_options_.use_image_storage = false;
}

common::Status VulkanExecutionProvider::OnSessionInitializationEnd() {
  // after session state initialization is done we switch from VkWeightStagingAllocator/VkWeightAllocator to
  // VkStagingAllocator/VkBlobAllocator
  // NOTE: We currently only support usage of the Vulkan EP in a single InferenceSession so we can use the approach
  //       of switching allocators based on IExecutionProvider::OnSessionInitializationEnd being called.
  //
  // TODO: If we're using Compile we could create a different ncnn::Option instance with the weight allocators
  //       instead of doing a switch here

  // data_transfer_.SetSessionInitialized();
  ncnn_options_.staging_vkallocator = &staging_allocator_;
  ncnn_options_.blob_vkallocator = &blob_allocator_;

  return Status::OK();
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  // TODO: Is there any explicit release of NCNN owned resources required?
}

std::vector<std::unique_ptr<ComputeCapability>>
VulkanExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  const auto& logger = *GetLogger();

  std::vector<std::unique_ptr<ComputeCapability>> result =
      GetCapabilityCompiling(graph_viewer, metadef_id_generator_, logger);

  return result;
}

common::Status VulkanExecutionProvider::UploadConstantInitializers(NcnnModel& model) {
  ncnn::VkTransfer cmd(&vulkan_device_);

  // use the weight allocators for the upload of constant initializers
  ncnn::Option upload_options(ncnn_options_);
  upload_options.staging_vkallocator = &weight_staging_allocator_;
  upload_options.blob_vkallocator = &weight_allocator_;

  for (size_t i = 0, end = model.layers.size(); i < end; i++) {
    ORT_RETURN_IF_ERROR(model.layers[i]->UploadConstantInitializers(cmd, upload_options));
  }

  RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

  return Status::OK();
}

common::Status VulkanExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  // see Net::load_model
  // NCNN execution setup involves looping through all the layers defined in the model to create a pipeline for each.

  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    // create a Net for the fused_node_and_graph
    // loop through the layers in the Net and create a pipeline for each

    NodeComputeInfo compute_info;
    const Node& fused_node = fused_node_and_graph.fused_node;
    const GraphViewer& graph = fused_node_and_graph.filtered_graph;

    auto model = std::make_unique<NcnnModel>();
    std::vector<std::unique_ptr<vulkan::VulkanKernel>>& layers = model->layers;
    model->layers.reserve(graph.NumberOfNodes());

    vulkan::VulkanKernel::ValueIndexes value_indexes;
    // rough guess for number of input and output values for this partition.
    value_indexes.reserve(graph.GetInputs().size() + gsl::narrow_cast<size_t>(graph.NumberOfNodes() * 2));

    // add all inputs to the value indexes
    for (const auto* def : fused_node.InputDefs()) {
      value_indexes.Add(*def);  // 'Add' skips missing optional inputs
    }

    // create layer for each node. we add the outputs from each layer to value_indexes when doing so
    for (const Node& node : graph.Nodes()) {
      std::unique_ptr<vulkan::VulkanKernel> layer;
      ORT_RETURN_IF_ERROR(vulkan::VulkanKernel::Create(*this, graph, node, value_indexes, layer));
      layers.push_back(std::move(layer));
    }

    // save the output indexes as they could come from multiple layers.
    for (const auto* def : fused_node.OutputDefs()) {
      if (def->Exists()) {
        model->output_indexes.push_back(value_indexes[def->Name()]);
      }
    }

    ORT_RETURN_IF_ERROR(UploadConstantInitializers(*model));

    // ModelMetadefIdGenerator is broken if this happens
    ORT_ENFORCE(models_.find(fused_node.Name()) == models_.end(), "Duplicate fused node names");
    models_[fused_node.Name()] = std::move(model);

    compute_info.create_state_func = [&model](ComputeContext* /*context*/, FunctionState* /*state*/) {
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
    };

    compute_info.compute_func = [&fused_node, this](FunctionState /*state*/, const OrtApi* /*c_api*/,
                                                    OrtKernelContext* context) -> Status {
      // undo the indirection to the public API that is required for external operators
      // this is an OpKernelContextInternal if we need that
      auto& ctx = *reinterpret_cast<OpKernelContext*>(context);
      auto model_iter = models_.find(ctx.GetNodeName());
      ORT_ENFORCE(model_iter != models_.end(), "Model for compiled node was not found!");

      NcnnModel& model = *model_iter->second;

      // create Vulkan compute instance to upload inputs (CPU to GPU), execute the nodes, and download outputs
      ncnn::VkCompute cmd(&vulkan_device_);

      // create vector for GPU based input/output values. the last output of the last layer is the highest index value.
      // init with empty instances
      std::vector<ncnn::VkMat> values;
      values.resize(model.layers.back()->Layer().tops.back() + 1);

      // setup inputs
      int input_idx = 0;
      int value_idx = 0;
      for (const auto* def : fused_node.InputDefs()) {
        if (def->Exists()) {
          const Tensor& input_tensor = *ctx.Input<Tensor>(input_idx);
          ncnn::Mat src = TensorToMat(input_tensor);
          // NCNN updates values[value_idx] during record_upload. this means any logic NCNN has around padding/packing
          // is applied directly (vs using TensorToVkMatWithPacking to attempt to replicate it).
          cmd.record_upload(src, values[value_idx], ncnn_options_);
          ++value_idx;
        }
        ++input_idx;
      }

      // TODO: Do we need to wait on the inputs being copied?
      // RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

      for (const auto& kernels : model.layers) {
        const auto& layer = kernels->Layer();
        if (layer.support_inplace) {
          if (layer.one_blob_only) {
            ncnn::VkMat& input = values[layer.bottoms[0]];
            RETURN_IF_NCNN_ERROR(layer.forward_inplace(input, cmd, ncnn_options_));
            values[layer.tops[0]] = input;  // copy VkMat info to output. copy of pointer/refcount not data.
          } else {
            // couldn't find any NCNN layers that support multiple inputs and inplace
            ORT_NOT_IMPLEMENTED("Inplace with multiple inputs not supported.");
          }
        } else {
          if (layer.one_blob_only) {
            RETURN_IF_NCNN_ERROR(layer.forward(values[layer.bottoms[0]], values[layer.tops[0]], cmd, ncnn_options_));
          } else {
            std::vector<ncnn::VkMat> inputs, outputs;

            inputs.reserve(layer.bottoms.size());
            outputs.reserve(layer.tops.size());

            for (int idx : layer.bottoms) {
              inputs.push_back(values[idx]);
            }

            for (int idx : layer.tops) {
              outputs.push_back(values[idx]);
            }

            RETURN_IF_NCNN_ERROR(layer.forward(inputs, outputs, cmd, ncnn_options_));
          }
        }
      }

      // copy data to output tensors.
      std::vector<ncnn::Mat> ncnn_outputs;
      auto num_outputs = model.output_indexes.size();
      ncnn_outputs.resize(num_outputs);

      // pre-allocate any outputs that have a known size so NCNN can write directly to it
      std::unordered_set<size_t> preallocated_outputs;
      preallocated_outputs.reserve(num_outputs);

      int output_idx = 0;
      for (const auto* def : fused_node.OutputDefs()) {
        if (def->Exists()) {
          const auto* tensorproto_shape = def->Shape();
          TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*tensorproto_shape);
          if (shape.Size() != -1) {
            Tensor& output = *ctx.Output(output_idx, shape);
            ncnn_outputs[output_idx] = TensorToMat(output);
            preallocated_outputs.insert(output_idx);
          }
        }

        ++output_idx;
      }

      output_idx = 0;
      for (size_t idx : model.output_indexes) {
        cmd.record_download(values[idx], ncnn_outputs[output_idx++], ncnn_options_);
      }

      // run the kernels and download the results
      RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

      // copy to ORT tensors
      output_idx = 0;
      for (const auto* def : fused_node.OutputDefs()) {
        if (def->Exists()) {
          if (preallocated_outputs.count(output_idx) == 0) {
            const auto& ncnn_output = ncnn_outputs[output_idx];
            auto elements_per_channel = ncnn_output.d * ncnn_output.h * ncnn_output.w;

            // we should fix this is record_download if it happens
            ORT_ENFORCE(elements_per_channel == ncnn_output.cstep || ncnn_output.elempack != 1,
                        "Output needs to be unpacked in record_download.");

            // convert NCNN Mat values to output shape.
            const auto* tensorproto_shape = def->Shape();
            auto rank = tensorproto_shape ? tensorproto_shape->dim_size() : ncnn_output.dims;

            ORT_ENFORCE(tensorproto_shape || ncnn_output.dims < 3,
                        "TODO: Validate handing when ORT output shape is unknown and NCNN output has 3 or more dims.");

            std::vector<int64_t> dims(rank, 1);  // default to 1
            switch (rank) {
              case 1:
                dims[0] = ncnn_output.w;
                break;
              case 2:
                dims[0] = ncnn_output.h;
                dims[1] = ncnn_output.w;
                break;
              case 3:
                // replicate the NCNN logic from mat.h when the input has 3 dims
                assert(ncnn_output.d == 1);  // TODO: not clear if there's a scenario where 'd' != 1 and rank is 3
                dims[0] = ncnn_output.c;
                dims[1] = ncnn_output.h;
                dims[2] = ncnn_output.w;
                break;
              case 4:
                // as NCNN doesn't support batches we assume 4D output with 'd' of 1 -> NCHW
                if (tensorproto_shape && ncnn_output.d == 1) {
                  dims[0] = 1;
                  dims[1] = ncnn_output.c;
                  dims[2] = ncnn_output.h;
                  dims[3] = ncnn_output.w;
                  break;
                }
                [[fallthrough]];
              default:
                ORT_NOT_IMPLEMENTED("Output rank not supported: ", rank,
                                    " Rank known:", tensorproto_shape ? "yes" : "no");
            }

            Tensor& output = *ctx.Output(output_idx, TensorShape(dims));
            auto* output_data = output.MutableDataRaw();
            memcpy(output_data, ncnn_output.data, output.SizeInBytes());
          }

          ++output_idx;
        }
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

// std::vector<AllocatorPtr> VulkanExecutionProvider::CreatePreferredAllocators() {
//   // TODO: We may want to utilize VMA to ensure the preferred memory is used as this could be used for either
//   // discrete or integrated GPUs.
//   // VMA is from AMD
//   // https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
//   // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/choosing_memory_type.html
//
//   // input/output is CPU based so we can use the CPU EP allocator initially.
//   // If we wanted to try and enable input/output on GPU we'd need to hook up the NCNN logic around packing etc.
//   // that is applied when it copies data to device.
//
//   std::vector<AllocatorPtr> allocators;
//   return allocators;
// }

}  // namespace onnxruntime
