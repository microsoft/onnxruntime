// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

std::vector<std::unique_ptr<ComputeCapability>>
GetCapabilityStaticKernels(const onnxruntime::GraphViewer& graph_viewer,
                           const IExecutionProvider::IKernelLookup& kernel_lookup,
                           const logging::Logger& logger) {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  for (const auto& node : graph_viewer.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node); kernel_create_info != nullptr) {
      if (!VulkanKernel::IsSupported(graph_viewer, node, logger)) {
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
                                            gen_metadef_name, "Vulkan", kVulkanExecutionProvider, nullptr,
                                            /* drop constant_initializers */ false);  // TBD if we can/should

  return result;
}
}  // namespace

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& info)
    : IExecutionProvider(kVulkanExecutionProvider,
                         OrtDevice(OrtDevice::CPU,  // input on CPU
                                   OrtDevice::MemType::DEFAULT,
                                   // we do not use device_id here when compiling as the memory is on CPU for the
                                   // input/output of the compiled partition
                                   0)),
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
  DumpKomputeManagerInfo(kompute_manager_);
  vma_allocator_;
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = kompute_manager_.getDeviceProperties().apiVersion;
  allocatorInfo.physicalDevice = *kompute_manager_.getVkPhysicalDevice();
  allocatorInfo.device = *kompute_manager_.getVkDevice();
  allocatorInfo.instance = *kompute_manager_.getVkInstance();

  vmaCreateAllocator(&allocatorInfo, &vma_allocator_);
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
}

std::vector<std::unique_ptr<ComputeCapability>>
VulkanExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  const auto& logger = *GetLogger();

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result = GetCapabilityCompiling(graph_viewer, metadef_id_generator_, logger);

  return result;
}

common::Status VulkanExecutionProvider::UploadConstantInitializers(const GraphViewer& graph_viewer,
                                                                   KomputeModel& model) {
  // assembly all the kp::Tensor instances to upload
  for (size_t i = 0, end = model.layers.size(); i < end; i++) {
    model.layers[i]->ProcessConstantInitializers(graph_viewer, kompute_manager_, model.constant_initializers);
  }

  // create vector with constant initializers to upload
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

common::Status VulkanExecutionProvider::CreateKernels(KomputeModel& model) {
  for (size_t i = 0, end = model.layers.size(); i < end; i++) {
    ORT_RETURN_IF_ERROR(
        model.layers[i]->CreateKernel(kompute_manager_, model.constant_initializers));
  }

  return Status::OK();
}

common::Status VulkanExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
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

    // these were used in the NCNN based implementation in a similar way to how we map OrtValue's to indexes.
    // currently unused in Kompute based execution but could potentially be if we wanted to put the values in a
    // vector instead of a map keyed on the NodeArg*. not clear if it's worth the effort to do so though.
    vulkan::VulkanKernel::ValueIndexes value_indexes;

    // create layer for each node
    for (const Node& node : graph.Nodes()) {
      std::unique_ptr<vulkan::VulkanKernel> layer;
      ORT_RETURN_IF_ERROR(vulkan::VulkanKernel::Create(*this, &graph, node, value_indexes, layer));
      layers.push_back(std::move(layer));
    }

    ORT_RETURN_IF_ERROR(UploadConstantInitializers(graph, *model));
    ORT_RETURN_IF_ERROR(CreateKernels(*model));

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
      // copy into map where we add output values as we execute the partition
      values = model_iter->second->constant_initializers;

      int input_idx = 0;
      std::vector<std::shared_ptr<kp::Tensor>> input_tensors;
      input_tensors.reserve(fused_node.InputDefs().size());

      for (const auto* def : fused_node.InputDefs()) {
        assert(def->Exists());  // fused node shouldn't have missing optional inputs
        if (values.find(def) == values.end()) {
          // not a constant initializer so create kp::Tensor and add to values
          const Tensor& input_tensor = *ctx.Input<Tensor>(input_idx);

          input_tensors.push_back(kompute_manager_.tensor(const_cast<void*>(input_tensor.DataRaw()),
                                                          narrow<uint32_t>(input_tensor.Shape().Size()),
                                                          sizeof(float), kp::Tensor::TensorDataTypes::eFloat));
          values[def] = input_tensors.back();
        }
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
        output_tensors.push_back(kompute_manager_.tensor(output_tensor.MutableDataRaw(),
                                                         narrow<uint32_t>(shape.Size()), sizeof(float),
                                                         kp::Tensor::TensorDataTypes::eFloat));
        values[def] = output_tensors.back();
      }

      auto seq = kompute_manager_.sequence();
      seq->record<kp::OpTensorSyncDevice>(input_tensors);

      for (const auto& layer : model_iter->second->layers) {
        ORT_RETURN_IF_ERROR(layer->Execute(kompute_manager_, *seq, values));
      }

      seq->record<kp::OpTensorSyncLocal>(output_tensors);
      seq->eval();

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
  std::vector<AllocatorPtr> allocators{
      std::make_unique<vulkan::VulkanBufferAllocator>(
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*device_id*/ 0), vma_allocator_),

      // using 'CUDA_PINNED' for staging memory
      std::make_unique<vulkan::VulkanBufferAllocator>(
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, /*device_id*/ 0), vma_allocator_)};

  return allocators;
}

}  // namespace onnxruntime
