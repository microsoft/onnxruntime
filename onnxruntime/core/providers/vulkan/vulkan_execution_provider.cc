// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "include/ncnn/gpu.h"
#include "include/ncnn/pipelinecache.h"
#include "include/ncnn/layer.h"

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
class ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MatMul, 6, 12);
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MatMul, 13);

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
      BuildKernelCreateInfo<ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MatMul, 6, 12)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MatMul, 13)>,
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

void DumpKomputeManagerInfo(kp::Manager& manager) {
  auto dump_props = [](vk::PhysicalDeviceProperties& props) {
    LOGS_DEFAULT(VERBOSE) << "Device: " << props.deviceName << " (id: " << props.deviceID << ")";
    LOGS_DEFAULT(VERBOSE) << "  Type: " << vk::to_string(props.deviceType);
    LOGS_DEFAULT(VERBOSE) << "  API Version: " << VK_VERSION_MAJOR(props.apiVersion) << "." << VK_VERSION_MINOR(props.apiVersion) << "." << VK_VERSION_PATCH(props.apiVersion);
    LOGS_DEFAULT(VERBOSE) << "  Driver Version: " << props.driverVersion;
    LOGS_DEFAULT(VERBOSE) << "  Vendor ID: " << props.vendorID;
    LOGS_DEFAULT(VERBOSE) << "  Device ID: " << props.deviceID;
  };

  auto devices = manager.listDevices();
  for (auto& device : devices) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vk::PhysicalDeviceProperties props(deviceProperties);
    dump_props(props);
  }

  LOGS_DEFAULT(VERBOSE) << "Current device: " << manager.getDeviceProperties().deviceID;
}

ncnn::VulkanDevice& GetVulkanDevice(int device_id = -1) {
  // get_gpu_count/get_default_gpu_index/get_gpu_device all implicitly create the gpu instance if it doesn't exist yet.
  // there is also `create_gpu_instance(const char* driver_path = 0);` if we need/want

  int gpu_count = ncnn::get_gpu_count();
  LOGS_DEFAULT(INFO) << "Vulkan capable GPU count:" << gpu_count;

  if (gpu_count == 0) {
    ORT_THROW("No Vulkan capable GPU detected.");
  }

  auto device_index = device_id >= 0 ? device_id : ncnn::get_default_gpu_index();

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
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node); kernel_create_info != nullptr) {
      if (!VulkanKernel::IsSupported(/*use_kompute*/ true, graph_viewer, node, logger)) {
        continue;
      }

      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

std::vector<std::unique_ptr<ComputeCapability>> GetCapabilityCompiling(bool use_kompute,
                                                                       const onnxruntime::GraphViewer& graph_viewer,
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
    return VulkanKernel::IsSupported(use_kompute, graph_viewer, node, logger);
  };

  // nothing required currently
  const utils::OnGroupClosedFn on_group_closed_fn = nullptr;

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported_fn, on_group_closed_fn,
                                            gen_metadef_name, "Vulkan", kVulkanExecutionProvider, nullptr,
                                            /* drop constant_initializers */ false);  // TBD if we can/should

  return result;
}

// NetPrivate::do_forward_layer.
// has logic to release buffers, but not clear how/where synchronization of the release with the GPU execution happens
int do_forward_layer(const ncnn::Layer* layer, std::vector<ncnn::VkMat>& blob_mats_gpu, ncnn::VkCompute& cmd,
                     const ncnn::Option& opt) {
  using namespace ncnn;

  if (layer->one_blob_only) {
    // load bottom blob
    int bottom_blob_index = layer->bottoms[0];
    int top_blob_index = layer->tops[0];

    VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
    VkMat bottom_blob;

    if (opt.lightmode) {
      // deep copy for inplace forward if data is shared
      if (layer->support_inplace && *bottom_blob_ref.refcount != 1) {
        cmd.record_clone(bottom_blob_ref, bottom_blob, opt);
        // NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(),
        //            bottom_blob.buffer(), bottom_blob.buffer_offset());
      }
    }

    if (bottom_blob.dims == 0) {
      bottom_blob = bottom_blob_ref;
    }

    // forward
    if (opt.lightmode && layer->support_inplace) {
      VkMat& bottom_top_blob = bottom_blob;
      int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
      if (ret != 0)
        return ret;

      // store top blob
      blob_mats_gpu[top_blob_index] = bottom_top_blob;
    } else {
      VkMat top_blob;
      int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
      if (ret != 0)
        return ret;

      // store top blob
      blob_mats_gpu[top_blob_index] = top_blob;
    }

    if (opt.lightmode) {
      // delete after taken in light mode
      blob_mats_gpu[bottom_blob_index].release();
    }
  } else {
    // load bottom blobs
    std::vector<VkMat> bottom_blobs(layer->bottoms.size());
    for (size_t i = 0; i < layer->bottoms.size(); i++) {
      int bottom_blob_index = layer->bottoms[i];

      VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
      bottom_blobs[i].release();

      if (opt.lightmode) {
        // deep copy for inplace forward if data is shared
        if (layer->support_inplace && *bottom_blob_ref.refcount != 1) {
          cmd.record_clone(bottom_blob_ref, bottom_blobs[i], opt);
          // NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(),
          //            bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset());
        }
      }

      if (bottom_blobs[i].dims == 0) {
        bottom_blobs[i] = bottom_blob_ref;
      }
    }

    // forward
    if (opt.lightmode && layer->support_inplace) {
      std::vector<VkMat>& bottom_top_blobs = bottom_blobs;
      int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
      if (ret != 0)
        return ret;

      // store top blobs
      for (size_t i = 0; i < layer->tops.size(); i++) {
        int top_blob_index = layer->tops[i];

        blob_mats_gpu[top_blob_index] = bottom_top_blobs[i];
      }
    } else {
      std::vector<VkMat> top_blobs(layer->tops.size());
      int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
      if (ret != 0)
        return ret;

      // store top blobs
      for (size_t i = 0; i < layer->tops.size(); i++) {
        int top_blob_index = layer->tops[i];

        blob_mats_gpu[top_blob_index] = top_blobs[i];
      }
    }

    if (opt.lightmode) {
      for (size_t i = 0; i < layer->bottoms.size(); i++) {
        int bottom_blob_index = layer->bottoms[i];

        // delete after taken in light mode
        blob_mats_gpu[bottom_blob_index].release();
      }
    }
  }

  return 0;
}

}  // namespace

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& info)
    : IExecutionProvider(kVulkanExecutionProvider,
                         OrtDevice(OrtDevice::CPU,  // input on CPU
                                   OrtDevice::MemType::DEFAULT,
                                   // we do not use device_id here when compiling as the memory is on CPU for the
                                   // input/output of the compiled partition
                                   0)),
      ncnn_options_{},
      ncnn_vulkan_device_{GetVulkanDevice()},
      weight_staging_allocator_{&ncnn_vulkan_device_},
      weight_allocator_{&ncnn_vulkan_device_},  // TODO: preferred_block_size is 8 MB. should it be different or configurable?
      staging_allocator_{&ncnn_vulkan_device_},
      blob_allocator_{&ncnn_vulkan_device_},  // TODO: preferred_block_size is 16 MB. should it be different or configurable?
      pipeline_cache_{std::make_unique<ncnn::PipelineCache>(&ncnn_vulkan_device_)},
      // TODO: Figure out required extensions.
      // llama.cpp has a few it checks for
      // https://github.com/ggerganov/llama.cpp/blob/a21c6fd45032a20180e026773582d21294c85619/ggml/src/ggml-kompute.cpp#L217
      // https://github.com/ggerganov/llama.cpp/blob/a21c6fd45032a20180e026773582d21294c85619/ggml/src/ggml-kompute.cpp#L1801-L1802
      // ncnn checks for a lot of VK_KHR_... values
      // https://github.com/search?q=repo%3ATencent%2Fncnn%20VK_KHR_&type=code
      // the 'support_' properties of GpuInfo might provide the superset of relevant ones, but what is
      // 'required' vs 'available' needs to be figured out as we add shaders.
      // https://github.com/Tencent/ncnn/blob/b9debee8fb92263cd3a087208d3657081a2e4f37/src/gpu.h#L262-L312
      kompute_manager_{narrow<uint32_t>(info.device_id), {}, {"VK_KHR_maintenance4"}},
      data_transfer_{kompute_manager_} {
  if (use_kompute_) {
    DumpKomputeManagerInfo(kompute_manager_);
    vma_allocator_;
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = kompute_manager_.getDeviceProperties().apiVersion;
    allocatorInfo.physicalDevice = *kompute_manager_.getVkPhysicalDevice();
    allocatorInfo.device = *kompute_manager_.getVkDevice();
    allocatorInfo.instance = *kompute_manager_.getVkInstance();

    vmaCreateAllocator(&allocatorInfo, &vma_allocator_);

  } else {
    ncnn_options_.use_vulkan_compute = true;

    // this was the setup when using static kernels to try and ensure we use these allocators for weight upload only.
    // with the compiling setup we plug these in explicitly during UploadConstantInitializers.
    // ncnn_options_.staging_vkallocator = &weight_staging_allocator_;
    // ncnn_options_.blob_vkallocator = &weight_allocator_;

    ncnn_options_.staging_vkallocator = &staging_allocator_;
    ncnn_options_.blob_vkallocator = &blob_allocator_;
    ncnn_options_.workspace_vkallocator = &blob_allocator_;

    ncnn_options_.pipeline_cache = pipeline_cache_.get();

    // start with fp32
    // NCNN can automatically convert from fp32 to fp16 if use_fp16_storage/use_fp16_packed is true.
    ncnn_options_.use_fp16_packed = false;
    ncnn_options_.use_fp16_storage = false;
    ncnn_options_.use_fp16_arithmetic = false;
    ncnn_options_.use_int8_packed = false;
    ncnn_options_.use_int8_storage = false;
    ncnn_options_.use_int8_arithmetic = false;
    ncnn_options_.use_packing_layout = false;
    ncnn_options_.use_image_storage = false;
  }
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  staging_allocator_.clear();
  blob_allocator_.clear();
  weight_staging_allocator_.clear();
  weight_allocator_.clear();

  ncnn_options_.pipeline_cache->clear();

  ncnn::destroy_gpu_instance();
}

std::vector<std::unique_ptr<ComputeCapability>>
VulkanExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  const auto& logger = *GetLogger();

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result = GetCapabilityCompiling(use_kompute_, graph_viewer, metadef_id_generator_, logger);

  return result;
}

common::Status VulkanExecutionProvider::UploadNcnnConstantInitializers(NcnnModel& model) {
  ncnn::VkTransfer cmd(&ncnn_vulkan_device_);

  // use the weight allocators for the upload of constant initializers
  ncnn::Option upload_options(ncnn_options_);
  upload_options.staging_vkallocator = &weight_staging_allocator_;
  upload_options.blob_vkallocator = &weight_allocator_;

  for (size_t i = 0, end = model.layers.size(); i < end; i++) {
    ORT_RETURN_IF_ERROR(model.layers[i]->UploadNcnnConstantInitializers(cmd, upload_options));
  }

  RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

  return Status::OK();
}

common::Status VulkanExecutionProvider::UploadKomputeConstantInitializers(const GraphViewer& graph_viewer,
                                                                          KomputeModel& model) {
  // assembly all the kp::Tensor instances to upload
  for (size_t i = 0, end = model.layers.size(); i < end; i++) {
    model.layers[i]->KomputeProcessConstantInitializers(graph_viewer, kompute_manager_, model.constant_initializers);
  }

  // create a kp::Tensor for each initializer
  std::vector<std::shared_ptr<kp::Tensor>> tensors;
  tensors.reserve(model.constant_initializers.size());
  std::for_each(model.constant_initializers.begin(), model.constant_initializers.end(),
                [&tensors](const auto& pair) {
                  tensors.push_back(pair.second);
                });

  auto seq = kompute_manager_.sequence();
  seq->record<kp::OpTensorSyncDevice>(tensors);
  seq->eval();

  return Status::OK();
}

common::Status VulkanExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  if (use_kompute_) {
    return CompileKompute(fused_nodes_and_graphs, node_compute_funcs);
  } else {
    return CompileNcnn(fused_nodes_and_graphs, node_compute_funcs);
  }
}

common::Status VulkanExecutionProvider::CompileKompute(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                       std::vector<NodeComputeInfo>& node_compute_funcs) {
  // https://kompute.cc/#mobile-enabled is a good example of creating pipeline including plugging in push constants

  // Maybe the following would work:
  // VulkanKernel ctor would create algo instance with spec constants and default push constants
  // a Compute method would add the OpAlgoDispatch call

  // to execute model we
  // - create input Tensor and call OpTensorSyncDevice to copy to device
  // - call Compute for each kernel to bind push constants
  // - call OpTensorSyncLocal to download outputs

  // Questions:
  // - How do we map inputs/outputs between kernels and overall outputs
  // - do we need to use streams so any scratch buffers required can be freed?
  //   - see BindToDeviceStream in session_state.cc
  //   - might be able to create a sequence instance via the create stream func
  //     - IStreamCommandHandleRegistry::RegisterCreateStreamFn
  //   - individual kernels can get the Stream from OpKernelContext
  // - How do we free tensors once they're done?
  //   - can this also be done via streams using the CUDA EP as a base implementation?

  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    const Node& fused_node = fused_node_and_graph.fused_node;
    const GraphViewer& graph = fused_node_and_graph.filtered_graph;

    auto model = std::make_unique<KomputeModel>();
    std::vector<std::unique_ptr<vulkan::VulkanKernel>>& layers = model->layers;
    model->layers.reserve(graph.NumberOfNodes());

    // initializers that have been copied to GPU
    // TODO: if we use static kernels we would handle this via DataTransferManager and ideally plugin a step
    // to allocate memory for all initializers that we're copying to GPU in a single allocation (or by doing a
    // something equivalent to a 'reserve' with the memory allocator we use for the initializers).
    // Need to figure out how VMA can be used to simplify this sort of thing.
    auto num_initializers = graph.GetAllInitializedTensors().size();
    model->constant_initializers.reserve(num_initializers);

    vulkan::VulkanKernel::ValueIndexes value_indexes;  // currently unused in Kompute based execution

    // create layer for each node. we add the outputs from each layer to value_indexes when doing so
    for (const Node& node : graph.Nodes()) {
      std::unique_ptr<vulkan::VulkanKernel> layer;
      ORT_RETURN_IF_ERROR(vulkan::VulkanKernel::Create(*this, use_kompute_, &graph, node, value_indexes, layer));
      layers.push_back(std::move(layer));
    }

    ORT_RETURN_IF_ERROR(UploadKomputeConstantInitializers(graph, *model));

    // ModelMetadefIdGenerator is broken if this happens
    ORT_ENFORCE(kompute_models_.find(fused_node.Name()) == kompute_models_.end(), "Duplicate fused node names");
    kompute_models_[fused_node.Name()] = std::move(model);

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

      auto model_iter = kompute_models_.find(ctx.GetNodeName());
      ORT_ENFORCE(model_iter != kompute_models_.end(), "Model for compiled node was not found!");

      std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>> values;
      // if the kernels track value_indexes we can do lookup into `values` using that instead of hashing ptr.
      // not sure it would make much difference.
      // would need to populate constant_initializers differently though
      values = model_iter->second->constant_initializers;  // copy

      // copy input to device. might be able to just create a kp::Tensor as that will be to staging memory.
      // if necessary use OpTensorSyncDevice
      int input_idx = 0;
      std::vector<std::shared_ptr<kp::Tensor>> input_tensors;
      input_tensors.reserve(fused_node.InputDefs().size());

      for (const auto* def : fused_node.InputDefs()) {
        assert(def->Exists());  // fused node shouldn't have missing optional inputs
        const Tensor& input_tensor = *ctx.Input<Tensor>(input_idx);
        // create kp::Tensor from input_tensor
        input_tensors.push_back(
            kompute_manager_.tensor(const_cast<void*>(input_tensor.DataRaw()),
                                    narrow<uint32_t>(input_tensor.Shape().Size()),
                                    sizeof(float), kp::Tensor::TensorDataTypes::eFloat));
      }

      int output_idx = 0;
      std::vector<std::shared_ptr<kp::Tensor>> output_tensors;
      std::vector<Tensor*> ort_output_tensors;
      output_tensors.reserve(fused_node.OutputDefs().size());
      ort_output_tensors.reserve(fused_node.OutputDefs().size());

      for (const auto* def : fused_node.OutputDefs()) {
        auto shape = utils::GetTensorShapeFromTensorShapeProto(*def->Shape());
        ORT_ENFORCE(shape.Size() > 0, "Output shape must be known and not empty.");

        Tensor& output_tensor = *ctx.Output(output_idx, shape);
        ort_output_tensors.push_back(&output_tensor);

        // create kp::Tensor from output_tensor
        output_tensors.push_back(
            kompute_manager_.tensor(output_tensor.MutableDataRaw(), narrow<uint32_t>(shape.Size()), sizeof(float),
                                    kp::Tensor::TensorDataTypes::eFloat));
      }

      auto seq = kompute_manager_.sequence();
      // copy inputs to device
      seq->record<kp::OpTensorSyncDevice>(input_tensors);

      // for each kernel, call KomputeExecute
      for (const auto& layer : model_iter->second->layers) {
        ORT_RETURN_IF_ERROR(layer->KomputeExecute(kompute_manager_, *seq, values));
      }

      seq->record<kp::OpTensorSyncLocal>(output_tensors);
      seq->eval();

      // do we need to copy from the values in output_tensors to the ORT output tensors?
      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

common::Status VulkanExecutionProvider::CompileNcnn(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
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
      ORT_RETURN_IF_ERROR(vulkan::VulkanKernel::Create(*this, use_kompute_, &graph, node, value_indexes, layer));
      layers.push_back(std::move(layer));
    }

    // save the output indexes as they could come from multiple layers.
    for (const auto* def : fused_node.OutputDefs()) {
      if (def->Exists()) {
        model->output_indexes.push_back(value_indexes[def->Name()]);
      }
    }

    // TODO: This is insufficient as it doesn't currently handle the case where the ORT input is a constant initializer
    // and the NCNN layer doesn't have a member for that. We could maybe extend the NCNN layer classes to handle that
    // but that could get quite extensive.
    //
    // An alternative would be to create a base std::vector<ncnn::VkMat> `values` which includes the constant
    // initializers that need to be provided as an input to the NCNN layer. At the start of execution we could copy
    // that vector. Potentially significantly inefficient if there are a lot of initializers that fall into this
    // category. The VulkanKernel would need to handle uploading the data to create the VkMat in this step.
    ORT_RETURN_IF_ERROR(UploadNcnnConstantInitializers(*model));

    // ModelMetadefIdGenerator is broken if this happens
    ORT_ENFORCE(ncnn_models_.find(fused_node.Name()) == ncnn_models_.end(), "Duplicate fused node names");
    ncnn_models_[fused_node.Name()] = std::move(model);

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

      auto model_iter = ncnn_models_.find(ctx.GetNodeName());
      ORT_ENFORCE(model_iter != ncnn_models_.end(), "Model for compiled node was not found!");

      NcnnModel& model = *model_iter->second;

      // create Vulkan compute instance to upload inputs (CPU to GPU), execute the nodes, and download outputs
      ncnn::VkCompute cmd(&ncnn_vulkan_device_);

      // create vector for GPU based input/output values. the last output of the last layer is the highest index value.
      // init with empty instances
      std::vector<ncnn::VkMat> values;
      values.resize(model.layers.back()->NcnnLayer().tops.back() + 1);

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

      // Do we need to wait on the inputs being copied?
      // RETURN_IF_NCNN_ERROR(cmd.submit_and_wait()); <-- Don't do this. Results in empty output.

      for (const auto& kernels : model.layers) {
        const auto& layer = kernels->NcnnLayer();
        RETURN_IF_NCNN_ERROR(do_forward_layer(&layer, values, cmd, ncnn_options_));
      }

      // copy data to output tensors.
      std::vector<ncnn::Mat> ncnn_outputs;
      auto num_outputs = model.output_indexes.size();
      ncnn_outputs.resize(num_outputs);

      // pre-allocate any outputs that have a known size so NCNN can write directly to it
      std::unordered_set<size_t> preallocated_outputs;
      preallocated_outputs.reserve(num_outputs);

      int output_idx = 0;
      // Can't use this without patching the record_download to not overwrite the provided Mat input.
      //
      // for (const auto* def : fused_node.OutputDefs()) {
      //   if (def->Exists()) {
      //     const auto* tensorproto_shape = def->Shape();
      //     TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*tensorproto_shape);
      //     if (shape.Size() != -1) {
      //       Tensor& output = *ctx.Output(output_idx, shape);
      //       ncnn_outputs[output_idx] = TensorToMat(output);
      //       preallocated_outputs.insert(output_idx);
      //     }
      //   }

      //  ++output_idx;
      //}

      output_idx = 0;
      for (size_t idx : model.output_indexes) {
        // we need a customized record_download to write directly to the ORT output
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

std::vector<AllocatorPtr> VulkanExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;

  if (use_kompute_) {
    allocators.push_back(std::make_unique<vulkan::VulkanBufferAllocator>(
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*device_id*/ 0),
        vma_allocator_));

    // using 'CUDA_PINNED' for staging memory
    allocators.push_back(std::make_unique<vulkan::VulkanBufferAllocator>(
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, /*device_id*/ 0),
        vma_allocator_));
  };

  return allocators;
}

}  // namespace onnxruntime
