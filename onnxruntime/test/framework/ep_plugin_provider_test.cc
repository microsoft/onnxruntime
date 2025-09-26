// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"

#include <filesystem>
#include "gsl/gsl"
#include "gtest/gtest.h"

#include "core/common/logging/sinks/file_sink.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/session/abi_devices.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {

// Helper class to access public ORT APIs.
struct ApiPtrs {
  ApiPtrs() : ort_api{::OrtGetApiBase()->GetApi(ORT_API_VERSION)},
              ep_api{ort_api->GetEpApi()} {
  }

  const gsl::not_null<const ::OrtApi*> ort_api;
  const gsl::not_null<const ::OrtEpApi*> ep_api;
};

static void CheckStringInFile(const PathString& filename, const std::string& look_for) {
  std::ifstream ifs{filename};
  std::string content(std::istreambuf_iterator<char>{ifs},
                      std::istreambuf_iterator<char>{});

  EXPECT_NE(content.find(look_for), std::string::npos);
}

// Normally, a plugin EP would be implemented in a separate library.
// The `test_plugin_ep` namespace contains a local implementation intended for unit testing.
namespace test_plugin_ep {

struct TestOrtEp : ::OrtEp, ApiPtrs {
  TestOrtEp() : ::OrtEp{}, ApiPtrs{} {
    ort_version_supported = ORT_API_VERSION;

    GetName = GetNameImpl;

    // Individual tests should fill out the other function pointers as needed.
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* /*this_ptr*/) noexcept {
    constexpr const char* ep_name = "TestOrtEp";
    return ep_name;
  }
};

// This factory doesn't do anything other than implement ReleaseEp().
// It is only used to create the UniqueOrtEp that is required by PluginExecutionProvider.
struct TestOrtEpFactory : ::OrtEpFactory {
  TestOrtEpFactory() : ::OrtEpFactory{} {
    ort_version_supported = ORT_API_VERSION;
    ReleaseEp = ReleaseEpImpl;
  }

  static void ORT_API_CALL ReleaseEpImpl(::OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<TestOrtEp*>(ep);
  }
};

static TestOrtEpFactory g_test_ort_ep_factory{};

std::unique_ptr<OrtHardwareDevice> MakeTestOrtHardwareDevice(OrtHardwareDeviceType type) {
  auto hw_device = std::make_unique<OrtHardwareDevice>();
  hw_device->type = type;
  hw_device->vendor_id = 0xBE57;
  hw_device->device_id = 0;
  hw_device->vendor = "Contoso";
  return hw_device;
}

std::unique_ptr<OrtEpDevice> MakeTestOrtEpDevice(const OrtHardwareDevice* hardware_device,
                                                 const OrtMemoryInfo* device_memory_info = nullptr,
                                                 const OrtMemoryInfo* host_accessible_memory_info = nullptr) {
  auto ep_device = std::make_unique<OrtEpDevice>();
  ep_device->ep_name = "TestOrtEp";
  ep_device->ep_vendor = "Contoso";
  ep_device->device = hardware_device;
  ep_device->ep_factory = &g_test_ort_ep_factory;
  ep_device->device_memory_info = device_memory_info;
  ep_device->host_accessible_memory_info = host_accessible_memory_info;
  return ep_device;
}

OrtDevice MakeTestOrtDevice(OrtDevice::DeviceType device_type, OrtDevice::MemoryType memory_type) {
  return OrtDevice(device_type, memory_type, /*vendor_id*/ 0xBE57, /*device_id*/ 0, /*alignment*/ 16);
}

struct MakeTestOrtEpResult {
  std::unique_ptr<IExecutionProvider> ep;  // the IExecutionProvider wrapping the TestOrtEp
  gsl::not_null<TestOrtEp*> ort_ep;        // the wrapped TestOrtEp, owned by `ep`
};

// Creates an IExecutionProvider that wraps a TestOrtEp.
// The TestOrtEp is also exposed so that tests can manipulate its function pointers directly.
MakeTestOrtEpResult MakeTestOrtEp(std::vector<const OrtEpDevice*> ep_devices = {}) {
  // Default OrtHardwareDevice and OrtEpDevice used if the caller does not explicitly provide ep_devices.
  static std::unique_ptr<OrtHardwareDevice> ort_hw_device = MakeTestOrtHardwareDevice(OrtHardwareDeviceType_CPU);
  static std::unique_ptr<OrtEpDevice> ort_ep_device = MakeTestOrtEpDevice(ort_hw_device.get());

  auto ort_ep_raw = std::make_unique<TestOrtEp>().release();
  auto ort_ep = UniqueOrtEp(ort_ep_raw, OrtEpDeleter{g_test_ort_ep_factory});
  auto ort_session_options = Ort::SessionOptions{};

  if (ep_devices.empty()) {
    ep_devices.push_back(ort_ep_device.get());
  }

  auto& logging_manager = DefaultLoggingManager();
  auto ep = std::make_unique<PluginExecutionProvider>(std::move(ort_ep),
                                                      *static_cast<const OrtSessionOptions*>(ort_session_options),
                                                      g_test_ort_ep_factory,
                                                      ep_devices,
                                                      logging_manager.DefaultLogger());

  auto result = MakeTestOrtEpResult{std::move(ep), ort_ep_raw};
  return result;
}

class MockKernelLookup : public IExecutionProvider::IKernelLookup {
  const KernelCreateInfo* LookUpKernel(const Node& /*node*/) const override { return nullptr; }
};

}  // namespace test_plugin_ep

TEST(PluginExecutionProviderTest, GetPreferredLayout) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->GetPreferredDataLayout = nullptr;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

  {
    auto prefer_nhwc_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept -> ::OrtStatus* {
      *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = prefer_nhwc_fn;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto invalid_layout_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept -> ::OrtStatus* {
      *preferred_data_layout = static_cast<OrtEpDataLayout>(-1);
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = invalid_layout_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }

  {
    auto failing_fn = [](OrtEp* this_ptr, OrtEpDataLayout* /*preferred_data_layout*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "I can't decide what data layout I prefer.");
    };
    ort_ep->GetPreferredDataLayout = failing_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

TEST(PluginExecutionProviderTest, ShouldConvertDataLayoutForOp) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->ShouldConvertDataLayoutForOp = nullptr;
    ASSERT_EQ(ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC), std::nullopt);
  }

  {
    auto custom_nhwc_op_determination_fn = [](OrtEp* /*this_ptr*/,
                                              const char* /*node_domain*/,
                                              const char* node_op_type,
                                              OrtEpDataLayout target_data_layout,
                                              int* should_convert) noexcept -> ::OrtStatus* {
      EXPECT_EQ(target_data_layout, OrtEpDataLayout::OrtEpDataLayout_NHWC);

      if (node_op_type == std::string_view{"Conv"}) {
        *should_convert = 1;
      } else if (node_op_type == std::string_view{"BatchNormalization"}) {
        *should_convert = 0;
      } else {
        *should_convert = -1;
      }
      return nullptr;
    };
    ort_ep->ShouldConvertDataLayoutForOp = custom_nhwc_op_determination_fn;

    std::optional<bool> should_convert{};

    should_convert = ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC);
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, true);

    should_convert = ep->ShouldConvertDataLayoutForOp("", "BatchNormalization", DataLayout::NHWC);
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, false);

    should_convert = ep->ShouldConvertDataLayoutForOp("", "GridSample", DataLayout::NHWC);
    ASSERT_EQ(should_convert, std::nullopt);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto failing_fn = [](OrtEp* this_ptr,
                         const char* /*node_domain*/,
                         const char* /*node_op_type*/,
                         OrtEpDataLayout /*target_data_layout*/,
                         int* /*should_convert*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL,
                                                "To convert to NHWC or not to convert to NHWC...");
    };
    ort_ep->ShouldConvertDataLayoutForOp = failing_fn;
    ASSERT_THROW(ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

TEST(PluginExecutionProviderTest, InferOrtDeviceFromDeviceMemoryInfo) {
  // 1 OrtEpDevice without a device_memory_info.
  // PluginExecutionProvider should decide to use a default OrtDevice.
  {
    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_CPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), OrtDevice());
  }

  // 1 OrtEpDevice with a device_memory_info.
  // PluginExecutionProvider should decide to use the OrtDevice from the device_memory_info.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(),
                                                             /*device_memory_info*/ ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 2 OrtEpDevice instances with the same device_memory_info.
  // PluginExecutionProvider should decide to use the OrtDevice from the device_memory_info.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 2 OrtEpDevice instances with the different (but equivalent) device_memory_info pointers.
  // PluginExecutionProvider should decide to use a OrtDevice that is equal to the devices used by both
  // device_memory_info pointers.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_0 = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                             ort_device, OrtMemTypeDefault);
    auto ort_memory_info_1 = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                             ort_device, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info_0.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info_1.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 1 OrtEpDevice with only a host_accessible_memory_info.
  // PluginExecutionProvider should decide to use a default OrtDevice (cpu).
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(),
                                                             /*device_memory_info*/ nullptr,
                                                             /*host_accessible_memory_info*/ ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), OrtDevice());
  }

#if !defined(ORT_NO_EXCEPTIONS)
  // 2 OrtEpDevice instances with DIFFERENT device_memory_info instances.
  // Should throw an exception on construction of PluginExecutionProvider.
  {
    auto ort_device_gpu = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_gpu = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                               ort_device_gpu, OrtMemTypeDefault);

    auto ort_device_npu = test_plugin_ep::MakeTestOrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_npu = std::make_unique<OrtMemoryInfo>("TestOrtEp NPU", OrtAllocatorType::OrtDeviceAllocator,
                                                               ort_device_npu, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info_gpu.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info_npu.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    ASSERT_THROW(test_plugin_ep::MakeTestOrtEp(ep_devices), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

static void LoadModelAndAssignNodesToEp(const ORTCHAR_T* model_path,
                                        const char* ep_name,
                                        const std::unordered_set<std::string>& ep_node_names,
                                        /*out*/ std::shared_ptr<Model>& model) {
  ASSERT_STATUS_OK(Model::Load(model_path, model, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  Graph& graph = model->MainGraph();

  for (Node& node : graph.Nodes()) {
    if (ep_node_names.count(node.Name()) > 0) {
      node.SetExecutionProviderType(ep_name);
    }
  }
}

static OrtStatus* ORT_API_CALL GetCapabilityTakeAllNodesOneGroup(OrtEp* this_ptr, const OrtGraph* graph,
                                                                 OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* this_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);

  size_t num_nodes = 0;
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNumNodes(graph, &num_nodes); st != nullptr) {
    return st;
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNodes(graph, nodes.data(), nodes.size()); st != nullptr) {
    return st;
  }

  if (OrtStatus* st = this_ep->ep_api->EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                         nodes.data(), nodes.size(), nullptr);
      st != nullptr) {
    return st;
  }

  return nullptr;
}

static OrtStatus* ORT_API_CALL GetCapabilityTakeAllNodesTwoGroups(OrtEp* this_ptr, const OrtGraph* graph,
                                                                  OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* this_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);

  size_t num_nodes = 0;
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNumNodes(graph, &num_nodes); st != nullptr) {
    return st;
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNodes(graph, nodes.data(), nodes.size()); st != nullptr) {
    return st;
  }

  // Expect at least 2 nodes. If not, this is really a testing/setup error.
  if (num_nodes < 2) {
    return this_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL,
                                          "Expected at least two nodes in call to GetCapability");
  }

  std::vector<const OrtNode*> node_group1;
  std::vector<const OrtNode*> node_group2;

  for (size_t i = 0; i < num_nodes; i++) {
    if (i < num_nodes / 2) {
      node_group1.push_back(nodes[i]);
    } else {
      node_group2.push_back(nodes[i]);
    }
  }

  if (OrtStatus* st = this_ep->ep_api->EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                         node_group1.data(), node_group1.size(),
                                                                         nullptr);
      st != nullptr) {
    return st;
  }

  if (OrtStatus* st = this_ep->ep_api->EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                         node_group2.data(), node_group2.size(),
                                                                         nullptr);
      st != nullptr) {
    return st;
  }

  return nullptr;
}

static OrtStatus* ORT_API_CALL GetCapabilityTakeSingleNode(OrtEp* this_ptr, const OrtGraph* graph,
                                                           OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* this_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);

  size_t num_nodes = 0;
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNumNodes(graph, &num_nodes); st != nullptr) {
    return st;
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  if (OrtStatus* st = this_ep->ort_api->Graph_GetNodes(graph, nodes.data(), nodes.size()); st != nullptr) {
    return st;
  }

  // Take only the first node using EpGraphSupportInfo_AddSingleNode().
  if (OrtStatus* st = this_ep->ep_api->EpGraphSupportInfo_AddSingleNode(graph_support_info, nodes[0]);
      st != nullptr) {
    return st;
  }

  return nullptr;
}

// Tests that GetCapability() doesn't crash if a plugin EP tries to claim a mix of unassigned nodes and
// nodes that are already assigned to another EP.
TEST(PluginExecutionProviderTest, GetCapability_ClaimNodesAssignedToOtherEP) {
  std::filesystem::path log_file = ORT_TSTR("log_get_capability.txt");

  // Helper function that loads a model (Add -> Mul -> Add) and assigns some or all of the nodes to another EP.
  // Then, IExecutionProvider::GetCapability() is called to test the expected behavior.
  auto run_test = [&log_file](IExecutionProvider& ep,
                              const std::unordered_set<std::string>& nodes_for_other_ep,
                              const std::unordered_set<std::string>& nodes_for_this_ep,
                              const char* expected_log_string) {
    std::shared_ptr<Model> model;
    ASSERT_NO_FATAL_FAILURE(LoadModelAndAssignNodesToEp(ORT_TSTR("testdata/add_mul_add.onnx"),
                                                        "OtherEp", nodes_for_other_ep, model));

    std::filesystem::remove(log_file);

    // Call IExecutionProvider::GetCapability and check results + logs.
    {
      logging::LoggingManager log_manager{std::make_unique<logging::FileSink>(log_file, false, false),
                                          logging::Severity::kWARNING, false,
                                          logging::LoggingManager::InstanceType::Temporal};
      auto file_logger = log_manager.CreateLogger("FileLogger");
      ep.SetLogger(file_logger.get());  // Make EP log to a file.

      GraphViewer graph_viewer(model->MainGraph());
      auto compute_capabilities = ep.GetCapability(graph_viewer,
                                                   test_plugin_ep::MockKernelLookup{},
                                                   GraphOptimizerRegistry(nullptr, nullptr, file_logger.get()),
                                                   nullptr);

      ASSERT_EQ(compute_capabilities.size(), nodes_for_this_ep.empty() ? 0 : 1);

      if (compute_capabilities.size() == 1) {
        ASSERT_EQ(compute_capabilities[0]->sub_graph->nodes.size(), nodes_for_this_ep.size());

        for (NodeIndex node_index : compute_capabilities[0]->sub_graph->nodes) {
          const Node* node = graph_viewer.GetNode(node_index);
          ASSERT_NE(node, nullptr);
          EXPECT_EQ(nodes_for_this_ep.count(node->Name()), 1);
        }
      }
    }

    ASSERT_TRUE(std::filesystem::exists(log_file));
    EXPECT_NO_FATAL_FAILURE(CheckStringInFile(log_file, expected_log_string));
  };

  constexpr std::array<const char*, 3> node_names = {"add_0", "mul_0", "add_1"};

  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  // Load a model and assign all of its nodes to another EP named 'OtherEp'.
  // The plugin EP tries to claim all nodes in a single group via EpGraphSupportInfo_AddNodesToFuse.
  // IExecutionProvider::GetCapability() should return an empty result and log a warning.
  ort_ep->GetCapability = GetCapabilityTakeAllNodesOneGroup;
  std::unordered_set<std::string> nodes_for_other_ep = {"add_0", "mul_0", "add_1"};
  std::unordered_set<std::string> nodes_for_this_ep;
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           "Found one or more nodes that were already assigned to a different EP named 'OtherEp'");

  // Load a model and assign only one node to another EP named 'OtherEp'.
  // The plugin EP tries to claim all nodes in a single group.
  // IExecutionProvider::GetCapability() should return an empty result and log a warning.
  ort_ep->GetCapability = GetCapabilityTakeAllNodesOneGroup;
  for (const char* node_name : node_names) {
    nodes_for_other_ep = std::unordered_set<std::string>{node_name};
    nodes_for_this_ep = std::unordered_set<std::string>{};
    run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
             "Found one or more nodes that were already assigned to a different EP named 'OtherEp'");
  }

  // Load a model and assign only the last Add node to another EP named 'OtherEp'.
  // The plugin EP tries to claim all nodes in the following 2 groups: (add_0), (mul_0, add_1).
  // IExecutionProvider::GetCapability() will only return (add_0) because the second group has a node
  // that was assigned to 'OtherEp'.
  ort_ep->GetCapability = GetCapabilityTakeAllNodesTwoGroups;
  nodes_for_other_ep = std::unordered_set<std::string>{"add_1"};
  nodes_for_this_ep = std::unordered_set<std::string>{"add_0"};
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           "Found one or more nodes that were already assigned to a different EP named 'OtherEp'");

  // Load a model and assign only the first Add node to another EP named 'OtherEp'.
  // The plugin EP tries to claim all nodes in the following 2 groups: (add_0), (mul_0, add_1).
  // IExecutionProvider::GetCapability() will only return (mul_0, add_1) because the first group has a node
  // that was assigned to 'OtherEp'.
  ort_ep->GetCapability = GetCapabilityTakeAllNodesTwoGroups;
  nodes_for_other_ep = std::unordered_set<std::string>{"add_0"};
  nodes_for_this_ep = std::unordered_set<std::string>{"mul_0", "add_1"};
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           "Found one or more nodes that were already assigned to a different EP named 'OtherEp'");

  // Load a model and assign the first Add node to another EP named 'OtherEp'.
  // The plugin EP will try to take only the first Add node with a single call to EpGraphSupportInfo_AddSingleNode.
  // IExecutionProvider::GetCapability() will return an empty result and log a warning.
  ort_ep->GetCapability = GetCapabilityTakeSingleNode;
  nodes_for_other_ep = std::unordered_set<std::string>{"add_0"};
  nodes_for_this_ep = std::unordered_set<std::string>{};
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           "Found one or more nodes that were already assigned to a different EP named 'OtherEp'");

  std::filesystem::remove(log_file);
}

}  // namespace onnxruntime::test
