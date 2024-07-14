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
namespace vulkan {

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// forward declarations of the kernel classes
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyToHost, 1);
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyFromHost, 1);
// class ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(HardSigmoid, 6, 21);
// class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(HardSigmoid, 22);
class ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 6, 12);
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 13);

//
// TODO: Do we need to apply any NCNN channel alignment logic when copying to/from GPU? Each channel is aligned
// to 16 bytes.
//
// See ncnn::Mat::reshape where channel alignment is implemented.
// We could maybe leverage Mat::reshape - first create a 1D Mat from the OrtValue and then reshape
// to change it to the NCNN format.
//
// We may not as VkTransfer::record_upload has a flag to flatten the data and the NCNN channel alignment may be for
// CPU only. VkCompute::record_upload does not have a flag though
//
// Also need to consider whether this alignment potentially invalidates ORT allocation planning as the required
// buffer size when used by NCNN may differ from ORT. That may be okay as it's CPU vs GPU usage,
//
ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

Status RegisterVulkanKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyToHost, 1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyFromHost, 1)>,
      // BuildKernelCreateInfo<ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(HardSigmoid, 6, 21)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(HardSigmoid, 22)>,
      BuildKernelCreateInfo<ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 6, 12)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 13)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetVulkanKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterVulkanKernels(*kernel_registry));
  return kernel_registry;
}
}  // namespace vulkan

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

std::vector<std::unique_ptr<ComputeCapability>>
GetCapabilityStaticKernels(const onnxruntime::GraphViewer& graph_viewer,
                           const IExecutionProvider::IKernelLookup& kernel_lookup,
                           const logging::Logger& logger) {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph_viewer.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {
      bool is_supported = true;
      // NCNN only supports 4D or less
      for (const auto& def : node.InputDefs()) {
        if (def->Exists()) {
          const auto* shape = def->Shape();
          if (!shape || shape->dim_size() > 4) {
            is_supported = false;
            break;
          }
        }
      }

      if (!is_supported) {
        continue;
      }

      if (!VulkanKernel::IsSupported(node, logger)) {
        continue;
      }

      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

std::vector<std::unique_ptr<ComputeCapability>> GetCapabilityCompiling(const onnxruntime::GraphViewer& graph_viewer,
                                                                       ModelMetadefIdGenerator metadef_id_generator,
                                                                       const logging::Logger& logger) {
  const auto gen_metadef_name =
      [&]() {
        HashValue model_hash;
        int metadef_id = metadef_id_generator.GenerateId(graph_viewer, model_hash);
        return MakeString(kVulkanExecutionProvider, "_", model_hash, "_", metadef_id);
      };

  std::vector<std::unique_ptr<ComputeCapability>> result;

  const utils::IsNodeSupportedFn is_node_supported_fn = [&logger](const Node& node) -> bool {
    return VulkanKernel::IsSupported(node, logger);
  };

  // nothing currently
  const utils::OnGroupClosedFn on_group_closed_fn = nullptr;

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported_fn, on_group_closed_fn,
                                            gen_metadef_name, "Vulkan", kVulkanExecutionProvider);

  return result;
}
}  // namespace

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& info, bool compiling)
    : IExecutionProvider(kVulkanExecutionProvider,
                         OrtDevice(compiling ? OrtDevice::CPU : OrtDevice::GPU,
                                   OrtDevice::MemType::DEFAULT,
                                   info.device_id)),
      ncnn_options_{},
      vulkan_device_{GetVulkanDevice()},
      weight_staging_allocator_{&vulkan_device_},
      weight_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 8 MB. should it be different or configurable?
      staging_allocator_{&vulkan_device_},
      blob_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 16 MB. should it be different or configurable?
      pipeline_cache_{std::make_unique<ncnn::PipelineCache>(&vulkan_device_)},
      data_transfer_{vulkan_device_, ncnn_options_},
      compiling_{compiling} {
  ncnn_options_.use_vulkan_compute = true;

  ncnn_options_.staging_vkallocator = &weight_staging_allocator_;
  ncnn_options_.blob_vkallocator = &weight_allocator_;
  ncnn_options_.pipeline_cache = pipeline_cache_.get();

  // start with fp32. NCNN can automatically convert from fp32 to fp16 if use_fp16_storage or use_fp16_packed is true.
  // - if we enable that we need to think about how that would integrate with the ORT generated allocation plan which
  //   is based on the Node input/output shapes which are not aware of an internal conversion to fp16.
  //   - we don't really want to be allocating 2x the required memory.
  //   - we could have the allocator for Vulkan memory convert an fp32 shape to the fp16 NCNN shape
  //     - needs to take into account any padding/packing that NCNN does and only allocate the required memory.
  //     - that might be OK for limiting allocation but may not work with the memory pattern planner, and we'd end up
  //       over allocating the backing store if that was the case.
  //     - also dangerous to have to implicitly know that the 'real' allocation is based on fp16 but other parts of the
  //       system like the Tensor::Shape() would be for fp32 sizes.
  //
  // also need to consider how/where that would impact when it's an integrated GPU where memory can be more directly
  // shared between CPU and GPU. e.g. we'd be going through the NCNN Packing layer between CPU and GPU kernels and
  // this may add significant unnecessary overhead.
  ncnn_options_.use_fp16_packed = false;
  ncnn_options_.use_fp16_storage = false;
  ncnn_options_.use_fp16_arithmetic = false;
  ncnn_options_.use_int8_packed = false;
  ncnn_options_.use_int8_storage = false;
  ncnn_options_.use_int8_arithmetic = false;
  ncnn_options_.use_packing_layout = false;
  ncnn_options_.use_image_storage = false;
}

// void VulkanExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
//                                                      AllocatorMap& allocators) const {
// }

common::Status VulkanExecutionProvider::OnSessionInitializationEnd() {
  // after session state initialization is done we switch from VkWeightStagingAllocator/VkWeightAllocator to
  // VkStagingAllocator/VkBlobAllocator
  // NOTE: We currently only support usage of the Vulkan EP in a single InferenceSession so we can use the approach
  //       of switching allocators based on IExecutionProvider::OnSessionInitializationEnd being called.

  data_transfer_.SetSessionInitialized();
  ncnn_options_.staging_vkallocator = &staging_allocator_;
  ncnn_options_.blob_vkallocator = &blob_allocator_;

  return Status::OK();
}

// NCNN keeps a cache of allocations. not clear we need multiple of each allocator type. shouldn't matter unless we
// need/want to support concurrent Run calls.
//
// This could also be used for cleaning up CPU scratch buffers if we get to a point we're using them. Needs the Stream
// setup and something roughly equivalent to how the CUDA EP handles 'deferred' CPU allocations.
//
// common::Status VulkanExecutionProvider::OnRunStart(const RunOptions& /*run_options*/) {
//  // https://github.com/Tencent/ncnn/wiki/vulkan-notes
//  // Do we need to acquire/reclaim on a per-Run basis or could we use staging_allocator_/blob_allocator_?
//  // TODO: Figure out how this looks with underlying larger memory allocations to determine what is optional.
//  // Assuming we want a small number of large allocations that are bound to buffers. The buffers may come and go
//  // more frequently.
//  ncnn_options_.blob_vkallocator = vulkan_device_.acquire_blob_allocator();
//  ncnn_options_.workspace_vkallocator = ncnn_options_.blob_vkallocator;
//  ncnn_options_.staging_vkallocator = vulkan_device_.acquire_staging_allocator();
//  return Status::OK();
//}
//
// common::Status VulkanExecutionProvider::OnRunEnd(bool /*sync_stream*/, const RunOptions& /*run_options*/) {
//  vulkan_device_.reclaim_blob_allocator(ncnn_options_.blob_vkallocator);
//  vulkan_device_.reclaim_staging_allocator(ncnn_options_.staging_vkallocator);
//
//  return Status::OK();
//}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  // TODO: Is there any explicit release of NCNN owned resources required?
}

std::vector<std::unique_ptr<ComputeCapability>>
VulkanExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup) const {
  const auto& logger = *GetLogger();
  std::vector<std::unique_ptr<ComputeCapability>> result =
      compiling_ ? GetCapabilityCompiling(graph_viewer, metadef_id_generator_, logger)
                 : GetCapabilityStaticKernels(graph_viewer, kernel_lookup, logger);

  return result;
}

common::Status VulkanExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  // see Net::load_model
  // NCNN execution setup involves looping through all the layers defined in the model to create a pipeline for each.

  // initial thought is we want to create a Net or something similar for each group of nodes.
  // TBD does that change execution though? if not we could just execute each kernel standalone
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
    // rough guess for number of input and output values.
    value_indexes.reserve(graph.GetInputs().size() + gsl::narrow_cast<size_t>(graph.NumberOfNodes() * 2));

    // add all inputs to the value indexes
    for (const auto* def : fused_node.InputDefs()) {
      value_indexes.Add(*def);  // 'Add' skips missing optional inputs
    }

    // create layer for each node. we add the outputs from each layer to value_indexes when doing so
    for (const Node& node : graph.Nodes()) {
      std::unique_ptr<vulkan::VulkanKernel> layer;
      ORT_RETURN_IF_ERROR(vulkan::VulkanKernel::Create(*this, node, value_indexes, layer));
      layers.push_back(std::move(layer));
    }

    // save the output indexes as they could come from multiple layers.
    for (const auto* def : fused_node.OutputDefs()) {
      if (def->Exists()) {
        model->output_indexes.push_back(value_indexes[def->Name()]);
      }
    }

    // ModelMetadefIdGenerator is broken if this happens
    ORT_ENFORCE(models_.find(fused_node.Name()) == models_.end(), "Duplicate fused node names");
    models_[fused_node.Name()] = std::move(model);

    compute_info.create_state_func = [&model](ComputeContext* /*context*/, FunctionState* /*state*/) {
      // *state = model.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
      // delete static_cast<NcnnModel*>(state);
    };

    compute_info.compute_func = [&fused_node, this](FunctionState /*state*/,
                                                    const OrtApi* /*c_api*/,
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

      // If we execute this way we control the copy to a potentially pre-allocated output buffer so could use the
      // NCNN kernels as-is.
      // If we use static kernels we don't control whether the output buffer is pre-allocated or not, so the inplace
      // logic needs to be conditionally used.
      // e.g. IOBindings provides pre-allocated output buffer. Sigmoid is last op. If it does inplace output it will
      // not update the pre-allocated output buffer unless we were able to wire that in somehow.
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

      // copy data to output tensors. gpu to gpu copy.
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

std::shared_ptr<KernelRegistry> VulkanExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = vulkan::GetVulkanKernelRegistry();
  return kernel_registry;
}

std::vector<AllocatorPtr> VulkanExecutionProvider::CreatePreferredAllocators() {
  // TODO: We may want to utilize VMA to ensure the preferred memory is used as this could be used for either
  // discrete or integrated GPUs.
  // VMA is from AMD
  // https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/choosing_memory_type.html

  std::vector<AllocatorPtr> allocators;

  if (compiling_) {
    // input/output is CPU based so we can use the CPU EP allocator.
  } else {
    // These allocators refer to the values in ncnn_options_ so that they see the change from OnSessionInitializationEnd
    // TODO: This assumes we never free memory used to copy initializers to the device as we wouldn't have the matching
    // allocator unless we have a way to swap back.
    allocators.push_back(std::make_unique<vulkan::VulkanBufferAllocator>(
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*device_id*/ 0),
        ncnn_options_.blob_vkallocator));

    // using 'CUDA_PINNED' for staging memory
    allocators.push_back(std::make_unique<vulkan::VulkanBufferAllocator>(
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, /*device_id*/ 0),
        ncnn_options_.staging_vkallocator));
  };

  return allocators;
}

}  // namespace onnxruntime
