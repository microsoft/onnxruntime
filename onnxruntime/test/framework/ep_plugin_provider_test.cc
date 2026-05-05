// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include "gsl/gsl"
#include "gtest/gtest.h"

#include "core/common/logging/sinks/file_sink.h"
#include "core/framework/config_options.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/framework/resource_accountant.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/session/abi_devices.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/api_asserts.h"
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
  ASSERT_TRUE(ifs);
  std::string content(std::istreambuf_iterator<char>{ifs},
                      std::istreambuf_iterator<char>{});

  EXPECT_NE(content.find(look_for), std::string::npos);
}

static void CheckFileIsEmpty(const PathString& filename) {
  std::ifstream ifs{filename};
  ASSERT_TRUE(ifs);
  std::string content(std::istreambuf_iterator<char>{ifs},
                      std::istreambuf_iterator<char>{});

  EXPECT_TRUE(content.empty());
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
                                                      /*kernel_registry*/ nullptr,
                                                      logging_manager.DefaultLogger());

  auto result = MakeTestOrtEpResult{std::move(ep), ort_ep_raw};
  return result;
}

using LookUpKernelFunc = std::function<const KernelCreateInfo*(const Node&)>;

class MockKernelLookup : public IExecutionProvider::IKernelLookup {
 public:
  explicit MockKernelLookup(LookUpKernelFunc lookup = nullptr) : lookup_{lookup} {}

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    return lookup_ != nullptr ? lookup_(node) : nullptr;
  }

 private:
  LookUpKernelFunc lookup_ = nullptr;
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

  // Take only the first node that has a registered kernel for this EP.
  for (const OrtNode* node : nodes) {
    const OrtKernelDef* kernel_def = nullptr;
    OrtStatus* status = this_ep->ep_api->EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def);

    if (status != nullptr) {
      return status;
    }

    if (kernel_def != nullptr) {
      if (OrtStatus* st = this_ep->ep_api->EpGraphSupportInfo_AddSingleNode(graph_support_info, node);
          st != nullptr) {
        return st;
      }

      break;
    }
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
                              const char* expected_log_string,
                              test_plugin_ep::LookUpKernelFunc lookup_kernel_func = nullptr) {
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
                                                   test_plugin_ep::MockKernelLookup(lookup_kernel_func),
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

    if (expected_log_string != nullptr) {
      EXPECT_NO_FATAL_FAILURE(CheckStringInFile(log_file, expected_log_string));
    } else {
      EXPECT_NO_FATAL_FAILURE(CheckFileIsEmpty(log_file));
    }
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

  // Build dummy kernel definition for an Add node. Retrieved by OrtEp using EpGraphSupportInfo_LookUpKernel().
  KernelDefBuilder builder;
  builder.SetName("Add").SinceVersion(1).Provider("TestOrtEp");
  auto add_kernel_create_info = std::make_unique<KernelCreateInfo>(builder.Build(), nullptr);

  auto mock_kernel_lookup_fn = [&add_kernel_create_info](const Node& node) -> const KernelCreateInfo* {
    // Only return a result for an Add node.
    if (add_kernel_create_info->kernel_def->OpName() == node.OpType()) {
      return add_kernel_create_info.get();
    }
    return nullptr;
  };

  // Load a model and assign the first Add node to another EP named 'OtherEp'.
  // The plugin EP will try to take only the first Add node with a single call to EpGraphSupportInfo_AddSingleNode.
  // IExecutionProvider::GetCapability() will return an empty result and log a warning.
  ort_ep->GetCapability = GetCapabilityTakeSingleNode;
  nodes_for_other_ep = std::unordered_set<std::string>{"add_0"};
  nodes_for_this_ep = std::unordered_set<std::string>{};
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           "Found one or more nodes that were already assigned to a different EP named 'OtherEp'",
           mock_kernel_lookup_fn);

  // Load a model and assign the last Add node to another EP named 'OtherEp'.
  // The plugin EP will try to take only the first Add node with a single call to EpGraphSupportInfo_AddSingleNode.
  // IExecutionProvider::GetCapability() will return a single capability and will not log warnings.
  ort_ep->GetCapability = GetCapabilityTakeSingleNode;
  nodes_for_other_ep = std::unordered_set<std::string>{"add_1"};
  nodes_for_this_ep = std::unordered_set<std::string>{"add_0"};
  run_test(*ep, nodes_for_other_ep, nodes_for_this_ep,
           /*expected_log_string*/ nullptr, mock_kernel_lookup_fn);

  std::filesystem::remove(log_file);
}

// Test plugin EP's use of the EpGraphSupportInfo_LookUpKernel API.
TEST(PluginExecutionProviderTest, GetCapability_LookUpKernel) {
  // Helper that calls IExecutionProvider::GetCapability and checks expected results.
  auto run_test = [](IExecutionProvider& ep, const std::unordered_set<std::string>& expected_claimed_nodes,
                     test_plugin_ep::LookUpKernelFunc lookup_kernel_func) {
    const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/add_mul_add.onnx"), model, nullptr,
                                 DefaultLoggingManager().DefaultLogger()));

    {
      ep.SetLogger(&logger);

      GraphViewer graph_viewer(model->MainGraph());
      auto compute_capabilities = ep.GetCapability(graph_viewer,
                                                   test_plugin_ep::MockKernelLookup(lookup_kernel_func),
                                                   GraphOptimizerRegistry(nullptr, nullptr, &logger),
                                                   nullptr);

      ASSERT_EQ(compute_capabilities.size(), expected_claimed_nodes.empty() ? 0 : 1);

      if (compute_capabilities.size() == 1) {
        ASSERT_EQ(compute_capabilities[0]->sub_graph->nodes.size(), expected_claimed_nodes.size());

        for (NodeIndex node_index : compute_capabilities[0]->sub_graph->nodes) {
          const Node* node = graph_viewer.GetNode(node_index);
          ASSERT_NE(node, nullptr);
          EXPECT_EQ(expected_claimed_nodes.count(node->Name()), 1);
        }
      }
    }
  };

  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  // Build dummy kernel lookup function that always returns null. Used by OrtEp using EpGraphSupportInfo_LookUpKernel().
  // Expect that the plugin EP will not claim any nodes because no valid kernel definitions are registered.
  {
    auto mock_kernel_lookup_fn = [](const Node& /*node*/) -> const KernelCreateInfo* {
      return nullptr;
    };

    ort_ep->GetCapability = GetCapabilityTakeSingleNode;
    std::unordered_set<std::string> expected_claimed_nodes;  // Empty. No nodes should be claimed.
    run_test(*ep, expected_claimed_nodes, mock_kernel_lookup_fn);
  }

  // Test a kernel lookup function that only returns a kernel definition for a Mul node.
  // Expect that plugin EP will take only the Mul node.
  {
    KernelDefBuilder builder;
    builder.SetName("Mul").SinceVersion(1).Provider("TestOrtEp");
    auto kernel_create_info = std::make_unique<KernelCreateInfo>(builder.Build(), nullptr);

    auto mock_kernel_lookup_fn = [&kernel_create_info](const Node& node) -> const KernelCreateInfo* {
      if (kernel_create_info->kernel_def->OpName() == node.OpType()) {
        return kernel_create_info.get();
      }
      return nullptr;
    };

    ort_ep->GetCapability = GetCapabilityTakeSingleNode;
    std::unordered_set<std::string> expected_claimed_nodes = {"mul_0"};
    run_test(*ep, expected_claimed_nodes, mock_kernel_lookup_fn);
  }
}

TEST(PluginExecutionProviderTest, KernelDefCxxApis) {
  auto check_kernel_def = [&](const KernelDef& expected, Ort::ConstKernelDef actual) -> void {
    EXPECT_EQ(expected.OpName(), actual.GetOperatorType());
    EXPECT_EQ(expected.Domain(), actual.GetDomain());

    auto [expected_start, expected_end] = expected.SinceVersion();
    auto [actual_start, actual_end] = actual.GetSinceVersion();

    EXPECT_EQ(expected_start, actual_start);

    if (expected_end != actual_end) {
      // Instead of using INT_MAX, the public API just sets the start version equal to the end version.
      EXPECT_EQ(actual_start, actual_end);
      EXPECT_EQ(expected_end, std::numeric_limits<int>::max());
    }

    EXPECT_EQ(expected.Provider(), actual.GetExecutionProvider());
    EXPECT_EQ(expected.InputMemoryType(0), actual.GetInputMemType(0));
    EXPECT_EQ(expected.InputMemoryType(1), actual.GetInputMemType(1));
    EXPECT_EQ(expected.OutputMemoryType(1), actual.GetOutputMemType(1));
  };

  // Check that C++ APIs for Ort::KernelDef return the expected values.
  {
    KernelDefBuilder builder;
    std::unique_ptr<KernelDef> expected_def = builder.SetName("Mul")
                                                  .SetDomain("TestDomain")
                                                  .SinceVersion(3, 13)
                                                  .Provider("TestOrtEp")
                                                  .InputMemoryType(OrtMemTypeCPUInput, 0)
                                                  .InputMemoryType(OrtMemTypeCPUInput, 1)
                                                  .OutputMemoryType(OrtMemTypeCPUOutput, 1)
                                                  .Build();

    Ort::KernelDefBuilder api_builder;
    Ort::KernelDef actual_def = api_builder.SetOperatorType("Mul")
                                    .SetDomain("TestDomain")
                                    .SetSinceVersion(3, 13)
                                    .SetExecutionProvider("TestOrtEp")
                                    .SetInputMemType(0, OrtMemTypeCPUInput)
                                    .SetInputMemType(1, OrtMemTypeCPUInput)
                                    .SetOutputMemType(1, OrtMemTypeCPUOutput)
                                    .Build();

    EXPECT_NO_FATAL_FAILURE(check_kernel_def(*expected_def, actual_def.GetConst()));
  }

  // SinceVersion with no explicit end (defaults to start version)
  {
    KernelDefBuilder builder;
    std::unique_ptr<KernelDef> expected_def = builder.SetName("Mul")
                                                  .SetDomain("TestDomain")
                                                  .Provider("TestOrtEp")
                                                  .SinceVersion(3)  // end should default to INT_MAX (means not set)
                                                  .Build();

    Ort::KernelDefBuilder api_builder;
    Ort::KernelDef actual_def = api_builder.SetOperatorType("Mul")
                                    .SetDomain("TestDomain")
                                    .SetExecutionProvider("TestOrtEp")
                                    .SetSinceVersion(3, 3)  // start == end (only one version supported)
                                    .Build();
    EXPECT_NO_FATAL_FAILURE(check_kernel_def(*expected_def, actual_def.GetConst()));
  }
}

TEST(PluginExecutionProviderTest, IsConcurrentRunSupported) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->IsConcurrentRunSupported = nullptr;
    ASSERT_TRUE(ep->ConcurrentRunSupported());
  }

  {
    auto concurrent_run_is_unsupported = [](OrtEp* /*this_ptr*/, bool* is_supported) noexcept -> ::OrtStatus* {
      *is_supported = false;
      return nullptr;
    };

    ort_ep->IsConcurrentRunSupported = concurrent_run_is_unsupported;
    ASSERT_FALSE(ep->ConcurrentRunSupported());
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto failing_fn = [](OrtEp* this_ptr, bool* /*is_supported*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "Concurrency? What's that?");
    };

    ort_ep->IsConcurrentRunSupported = failing_fn;
    ASSERT_THROW(ep->ConcurrentRunSupported(), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

// Tests for the Ort::OpSchema C++ wrapper API and Ort::GetOpSchema free function.
// These test the C++ layer over the OrtEpApi OpSchema functions using well-known ONNX operator schemas
// from the global ONNX schema registry.

// Test that GetOpSchema returns null for various not-found cases.
TEST(OpSchemaCxxApiTest, GetOpSchema_NotFound) {
  // Unknown op name
  Ort::OpSchema schema_unknown = Ort::GetOpSchema("NonExistentOpXYZ_12345", 20, "");
  EXPECT_EQ(static_cast<OrtOpSchema*>(schema_unknown), nullptr);

  // Relu was introduced in opset 1, so max_inclusive_version=0 should not find it.
  Ort::OpSchema schema_v0 = Ort::GetOpSchema("Relu", 0, "");
  EXPECT_EQ(static_cast<OrtOpSchema*>(schema_v0), nullptr);

  // Wrong domain
  Ort::OpSchema schema_bad_domain = Ort::GetOpSchema("Relu", 20, "com.nonexistent.domain");
  EXPECT_EQ(static_cast<OrtOpSchema*>(schema_bad_domain), nullptr);
}

// Test version differentiation and "ai.onnx" domain alias normalization.
TEST(OpSchemaCxxApiTest, DifferentVersionsAndDomainAlias) {
  // Relu was introduced in opset 1 and updated in opset 6, 13, and 14.
  // Querying at version 5 should return the opset 1 schema.
  Ort::OpSchema schema_v5 = Ort::GetOpSchema("Relu", 5, "");
  ASSERT_NE(static_cast<OrtOpSchema*>(schema_v5), nullptr);
  EXPECT_EQ(schema_v5.GetSinceVersion(), 1);

  // Querying at version 6 with "ai.onnx" domain alias should return the opset 6 schema.
  Ort::OpSchema schema_v6 = Ort::GetOpSchema("Relu", 6, kOnnxDomainAlias);
  ASSERT_NE(static_cast<OrtOpSchema*>(schema_v6), nullptr);
  EXPECT_EQ(schema_v6.GetSinceVersion(), 6);

  // "ai.onnx" and "" should resolve to the same schema at the same version.
  Ort::OpSchema schema_canonical = Ort::GetOpSchema("Relu", 20, "");
  Ort::OpSchema schema_alias = Ort::GetOpSchema("Relu", 20, kOnnxDomainAlias);
  ASSERT_NE(static_cast<OrtOpSchema*>(schema_canonical), nullptr);
  ASSERT_NE(static_cast<OrtOpSchema*>(schema_alias), nullptr);
  EXPECT_EQ(schema_canonical.GetSinceVersion(), schema_alias.GetSinceVersion());
}

// Test OpSchema methods on the "Add" operator (2 inputs, 1 output, shared constraint T).
// Also tests pointer identity: inputs/output sharing a constraint return the same pointer.
TEST(OpSchemaCxxApiTest, AddSchemaProperties) {
  int opset_version = 20;
  Ort::OpSchema schema = Ort::GetOpSchema("Add", opset_version, "");
  ASSERT_NE(static_cast<OrtOpSchema*>(schema), nullptr);

  // The "since version" will be <= to the opset version used to retrieve the schema.
  EXPECT_LT(schema.GetSinceVersion(), opset_version + 1);
  EXPECT_GT(schema.GetSinceVersion(), 0);

  // Add has 2 inputs: A, B
  ASSERT_EQ(schema.GetNumInputs(), 2u);
  EXPECT_EQ(schema.GetInputName(0), "A");
  EXPECT_EQ(schema.GetInputName(1), "B");

  // Both inputs should have a type constraint named "T"
  Ort::ConstOpSchemaTypeConstraint tc_input0 = schema.GetInputTypeConstraint(0);
  Ort::ConstOpSchemaTypeConstraint tc_input1 = schema.GetInputTypeConstraint(1);
  ASSERT_NE(static_cast<const OrtOpSchemaTypeConstraint*>(tc_input0), nullptr);
  ASSERT_NE(static_cast<const OrtOpSchemaTypeConstraint*>(tc_input1), nullptr);
  EXPECT_EQ(tc_input0.GetTypeParamName(), "T");
  EXPECT_EQ(tc_input1.GetTypeParamName(), "T");

  // Add has 1 output: C
  ASSERT_EQ(schema.GetNumOutputs(), 1u);
  EXPECT_EQ(schema.GetOutputName(0), "C");

  Ort::ConstOpSchemaTypeConstraint tc_output0 = schema.GetOutputTypeConstraint(0);
  ASSERT_NE(static_cast<const OrtOpSchemaTypeConstraint*>(tc_output0), nullptr);
  EXPECT_EQ(tc_output0.GetTypeParamName(), "T");

  // Both inputs and the output share constraint "T" — should return the same pointer.
  EXPECT_EQ(static_cast<const OrtOpSchemaTypeConstraint*>(tc_input0),
            static_cast<const OrtOpSchemaTypeConstraint*>(tc_input1));
  EXPECT_EQ(static_cast<const OrtOpSchemaTypeConstraint*>(tc_input0),
            static_cast<const OrtOpSchemaTypeConstraint*>(tc_output0));
}

// Tests for the OrtOpSchemaTypeConstraint API (per-constraint entity).

// Test type constraints for the Add operator (single constraint T on all inputs/outputs).
TEST(OpSchemaTypeConstraintTest, Add_SingleConstraint) {
  Ort::OpSchema schema = Ort::GetOpSchema("Add", 20, "");
  ASSERT_NE(static_cast<OrtOpSchema*>(schema), nullptr);

  ASSERT_EQ(schema.GetTypeConstraintCount(), 1u);

  // Constraint "T"
  Ort::ConstOpSchemaTypeConstraint tc = schema.GetTypeConstraint(0);
  EXPECT_EQ(tc.GetTypeParamName(), "T");

  // T should allow tensor(float) and tensor(double) among others
  auto allowed_types = tc.GetAllowedTypes();
  EXPECT_GT(allowed_types.size(), 1u);
  EXPECT_THAT(allowed_types, ::testing::Contains("tensor(float)")) << "Expected T to allow tensor(float)";
  EXPECT_THAT(allowed_types, ::testing::Contains("tensor(double)")) << "Expected T to allow tensor(double)";

  // Both inputs use T
  auto input_indices = tc.GetInputIndices();
  ASSERT_EQ(input_indices.size(), 2u);
  EXPECT_EQ(input_indices[0], 0u);
  EXPECT_EQ(input_indices[1], 1u);

  // Output uses T
  auto output_indices = tc.GetOutputIndices();
  ASSERT_EQ(output_indices.size(), 1u);
  EXPECT_EQ(output_indices[0], 0u);
}

// Test type constraints for LSTM (multiple constraints: T and T1).
TEST(OpSchemaTypeConstraintTest, LSTM_MultipleConstraints) {
  Ort::OpSchema schema = Ort::GetOpSchema("LSTM", 20, "");
  ASSERT_NE(static_cast<OrtOpSchema*>(schema), nullptr);

  // LSTM has at least T and T1
  ASSERT_GE(schema.GetTypeConstraintCount(), 2u);

  // Find the T and T1 constraints by name
  const OrtOpSchemaTypeConstraint* t_ptr = nullptr;
  const OrtOpSchemaTypeConstraint* t1_ptr = nullptr;
  Ort::ConstOpSchemaTypeConstraint t_tc{nullptr};
  Ort::ConstOpSchemaTypeConstraint t1_tc{nullptr};
  for (size_t i = 0; i < schema.GetTypeConstraintCount(); ++i) {
    auto tc = schema.GetTypeConstraint(i);
    if (tc.GetTypeParamName() == "T") {
      t_ptr = static_cast<const OrtOpSchemaTypeConstraint*>(tc);
      t_tc = tc;
    } else if (tc.GetTypeParamName() == "T1") {
      t1_ptr = static_cast<const OrtOpSchemaTypeConstraint*>(tc);
      t1_tc = tc;
    }
  }

  ASSERT_NE(t_ptr, nullptr) << "Expected to find type constraint 'T'";
  ASSERT_NE(t1_ptr, nullptr) << "Expected to find type constraint 'T1'";

  // T should include tensor(float) and tensor(double)
  auto t_types = t_tc.GetAllowedTypes();
  EXPECT_GT(t_types.size(), 0u);
  EXPECT_THAT(t_types, ::testing::Contains("tensor(float)")) << "Expected T to allow tensor(float)";
  EXPECT_THAT(t_types, ::testing::Contains("tensor(double)")) << "Expected T to allow tensor(double)";

  // T1 should include tensor(int32) (sequence_lens is int32)
  auto t1_types = t1_tc.GetAllowedTypes();
  EXPECT_GT(t1_types.size(), 0u);

  // T1 is for sequence_lens which is int32
  EXPECT_THAT(t1_types, ::testing::Contains("tensor(int32)")) << "Expected T1 to allow tensor(int32)";

  // T should map to inputs X (0), W (1), R (2), B (3), initial_h (5), initial_c (6), P (7)
  auto t_inputs = t_tc.GetInputIndices();
  EXPECT_EQ(t_inputs.size(), 7u);
  EXPECT_EQ(t_inputs[0], 0u);  // X
  EXPECT_EQ(t_inputs[1], 1u);  // W
  EXPECT_EQ(t_inputs[2], 2u);  // R
  EXPECT_EQ(t_inputs[3], 3u);  // B
  EXPECT_EQ(t_inputs[4], 5u);  // initial_h
  EXPECT_EQ(t_inputs[5], 6u);  // initial_c
  EXPECT_EQ(t_inputs[6], 7u);  // P

  // T should map to outputs Y (0), Y_h (1), Y_c (2)
  auto t_outputs = t_tc.GetOutputIndices();
  ASSERT_EQ(t_outputs.size(), 3u);
  EXPECT_EQ(t_outputs[0], 0u);  // Y
  EXPECT_EQ(t_outputs[1], 1u);  // Y_h
  EXPECT_EQ(t_outputs[2], 2u);  // Y_c

  // T1 should map to the sequence_lens input (index 4)
  auto t1_inputs = t1_tc.GetInputIndices();
  ASSERT_EQ(t1_inputs.size(), 1u);
  EXPECT_EQ(t1_inputs[0], 4u);  // sequence_lens is the 5th input (index 4)

  // T1 should not map to any outputs
  auto t1_outputs = t1_tc.GetOutputIndices();
  EXPECT_EQ(t1_outputs.size(), 0u);
}

#if !defined(ORT_NO_EXCEPTIONS)
// Test out-of-range index for type constraint accessors.
TEST(OpSchemaTypeConstraintTest, OutOfRangeIndex) {
  Ort::OpSchema schema = Ort::GetOpSchema("Add", 20, "");
  ASSERT_NE(static_cast<OrtOpSchema*>(schema), nullptr);

  size_t count = schema.GetTypeConstraintCount();

  // Accessing beyond the count should throw
  EXPECT_THROW(schema.GetTypeConstraint(count), Ort::Exception);
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

TEST(PluginExecutionProviderTest, CreateProfilingEvent_AllCategories) {
  const OrtProfilingEventCategory categories[] = {
      OrtProfilingEventCategory_SESSION,
      OrtProfilingEventCategory_NODE,
      OrtProfilingEventCategory_KERNEL,
      OrtProfilingEventCategory_API,
  };

  for (auto cat : categories) {
    OrtProfilingEvent* event = nullptr;
    Ort::Status status{Ort::GetEpApi().CreateProfilingEvent(
        cat, -1, -1, "test", 0, 0, nullptr, nullptr, 0, &event)};
    Ort::ProfilingEvent cxx_event(event);

    ASSERT_TRUE(status.IsOK()) << "Failed for category " << static_cast<int>(cat);
    ASSERT_NE(event, nullptr);

    OrtProfilingEventCategory actual_cat{};
    ASSERT_ORTSTATUS_OK(Ort::GetEpApi().ProfilingEvent_GetCategory(event, &actual_cat));
    EXPECT_EQ(actual_cat, cat);
  }
}

TEST(PluginExecutionProviderTest, CreateProfilingEvent_NullOutput) {
  const auto& ep_api = Ort::GetEpApi();

  Ort::Status status{ep_api.CreateProfilingEvent(
      OrtProfilingEventCategory_KERNEL, -1, -1,
      "event", 0, 0, nullptr, nullptr, 0, /*out=*/nullptr)};

  ASSERT_FALSE(status.IsOK());
  ASSERT_THAT(status.GetErrorMessage(), ::testing::HasSubstr("output parameter is NULL"));
}

TEST(PluginExecutionProviderTest, ProfilingEvent_GetArgValue_NullKey) {
  const auto& ep_api = Ort::GetEpApi();

  OrtProfilingEvent* event = nullptr;
  ASSERT_ORTSTATUS_OK(ep_api.CreateProfilingEvent(
      OrtProfilingEventCategory_KERNEL, -1, -1,
      "event", 0, 0, nullptr, nullptr, 0, &event));

  Ort::ProfilingEvent cxx_event(event);

  const char* val = nullptr;
  Ort::Status status{ep_api.ProfilingEvent_GetArgValue(event, /*key=*/nullptr, &val)};
  ASSERT_FALSE(status.IsOK());
  ASSERT_THAT(status.GetErrorMessage(), ::testing::HasSubstr("Key parameter is NULL"));
}

TEST(PluginExecutionProviderTest, ProfilingEvent_GetArgValue_NullOutput) {
  const auto& ep_api = Ort::GetEpApi();

  OrtProfilingEvent* event = nullptr;
  ASSERT_ORTSTATUS_OK(ep_api.CreateProfilingEvent(
      OrtProfilingEventCategory_KERNEL, -1, -1,
      "event", 0, 0, nullptr, nullptr, 0, &event));

  Ort::ProfilingEvent cxx_event(event);

  Ort::Status status{ep_api.ProfilingEvent_GetArgValue(event, "key", /*out=*/nullptr)};
  ASSERT_FALSE(status.IsOK());
  ASSERT_THAT(status.GetErrorMessage(), ::testing::HasSubstr("Output parameter is NULL"));
}

#if !defined(ORT_NO_EXCEPTIONS)
TEST(PluginExecutionProviderTest, ProfilingEvent_CxxWrapper) {
  // Test the owning ProfilingEvent C++ wrapper with args.
  std::unordered_map<std::string, std::string> args = {{"op_name", "Conv"},
                                                       {"parent_name", "Conv_node_event"}};

  Ort::ProfilingEvent event(OrtProfilingEventCategory_NODE, /*process_id=*/1, /*thread_id=*/2,
                            "node_exec", /*timestamp_us=*/5000, /*duration_us=*/300,
                            args);

  EXPECT_EQ(event.GetCategory(), OrtProfilingEventCategory_NODE);
  EXPECT_STREQ(event.GetName(), "node_exec");
  EXPECT_EQ(event.GetTimestampUs(), 5000);
  EXPECT_EQ(event.GetDurationUs(), 300);
  EXPECT_EQ(event.GetArgValue("op_name"), args["op_name"]);
  EXPECT_EQ(event.GetArgValue("parent_name"), args["parent_name"]);
  EXPECT_EQ(event.GetArgValue("missing"), nullptr);
}

TEST(PluginExecutionProviderTest, ProfilingEvent_CxxWrapper_ArgsCArrays) {
  std::array<const char*, 2> arg_keys = {"op_name", "parent_name"};
  std::array<const char*, 2> arg_values = {"Conv", "Conv_node_event"};

  Ort::ProfilingEvent event(OrtProfilingEventCategory_NODE, /*process_id=*/1, /*thread_id=*/2,
                            "node_exec", /*timestamp_us=*/5000, /*duration_us=*/300,
                            arg_keys.data(), arg_values.data(), arg_keys.size());

  EXPECT_EQ(event.GetCategory(), OrtProfilingEventCategory_NODE);
  EXPECT_STREQ(event.GetName(), "node_exec");
  EXPECT_EQ(event.GetTimestampUs(), 5000);
  EXPECT_EQ(event.GetDurationUs(), 300);
  EXPECT_STREQ(event.GetArgValue("op_name"), arg_values[0]);
  EXPECT_STREQ(event.GetArgValue("parent_name"), arg_values[1]);
  EXPECT_EQ(event.GetArgValue("missing"), nullptr);
}

TEST(PluginExecutionProviderTest, ProfilingEvent_CxxWrapper_NoArgs) {
  Ort::ProfilingEvent event(OrtProfilingEventCategory_API, -1, -1,
                            "api_call", 0, 100);

  EXPECT_EQ(event.GetCategory(), OrtProfilingEventCategory_API);
  EXPECT_STREQ(event.GetName(), "api_call");
  EXPECT_EQ(event.GetTimestampUs(), 0);
  EXPECT_EQ(event.GetDurationUs(), 100);
  EXPECT_EQ(event.GetArgValue("any_key"), nullptr);
}

TEST(PluginExecutionProviderTest, ProfilingEvent_ConstWrapper) {
  // Create an event, then wrap the raw pointer as ConstProfilingEvent (non-owning).
  Ort::ProfilingEvent owned_event(OrtProfilingEventCategory_KERNEL, 10, 20,
                                  "kernel_op", 999, 111);

  Ort::ConstProfilingEvent const_event = owned_event.GetConst();

  EXPECT_EQ(const_event.GetCategory(), OrtProfilingEventCategory_KERNEL);
  EXPECT_STREQ(const_event.GetName(), "kernel_op");
  EXPECT_EQ(const_event.GetTimestampUs(), 999);
  EXPECT_EQ(const_event.GetDurationUs(), 111);
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

// ---------------------------------------------------------------------------
// Test that CreatePreferredAllocators wraps a Shrink-capable plugin allocator
// as IArena (not just IAllocator), so ShrinkMemoryArenas can find it.
// ---------------------------------------------------------------------------

namespace {

// Minimal fake OrtAllocator with Shrink support.
// Tracks Shrink calls via a counter.
struct FakeArenaOrtAllocator : OrtAllocator {
  int shrink_call_count = 0;
  OrtMemoryInfo* mem_info = nullptr;
};

static void* ORT_API_CALL FakeAlloc(OrtAllocator*, size_t) noexcept { return nullptr; }
static void ORT_API_CALL FakeFree(OrtAllocator*, void*) noexcept {}
static const OrtMemoryInfo* ORT_API_CALL FakeInfo(const OrtAllocator* self) noexcept {
  return static_cast<const FakeArenaOrtAllocator*>(self)->mem_info;
}
static OrtStatus* ORT_API_CALL FakeShrink(OrtAllocator* self) noexcept {
  static_cast<FakeArenaOrtAllocator*>(self)->shrink_call_count++;
  return nullptr;
}
static OrtStatus* ORT_API_CALL FakeGetStats(const OrtAllocator*, OrtKeyValuePairs** out) noexcept {
  ::OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateKeyValuePairs(out);
  return nullptr;
}

static FakeArenaOrtAllocator MakeFakeArenaAllocator(OrtMemoryInfo* mem_info, bool with_shrink = true) {
  FakeArenaOrtAllocator fa{};
  static_assert(std::is_standard_layout_v<OrtAllocator>);
  std::memset(static_cast<OrtAllocator*>(&fa), 0, sizeof(OrtAllocator));
  fa.version = ORT_API_VERSION;
  fa.mem_info = mem_info;
  fa.Alloc = FakeAlloc;
  fa.Free = FakeFree;
  fa.Info = FakeInfo;
  fa.Shrink = with_shrink ? FakeShrink : nullptr;
  fa.GetStats = FakeGetStats;
  return fa;
}

// Namespace-level storage so C function pointers can access the fake allocator.
static OrtAllocator* g_fake_allocator_for_test = nullptr;

static OrtStatus* ORT_API_CALL FakeCreateAllocator(OrtEp*, const OrtMemoryInfo*,
                                                   OrtAllocator** out) noexcept {
  *out = g_fake_allocator_for_test;
  return nullptr;
}

static void ORT_API_CALL FakeReleaseAllocator(OrtEpFactory*, OrtAllocator*) noexcept {
  // No-op: tests own the fake allocator lifetime.
}

}  // namespace

TEST(PluginExecutionProviderTest, CreatePreferredAllocators_ShrinkCapableAllocatorExposedAsArena) {
  // Set up a device with device_memory_info so CreatePreferredAllocators iterates it.
  auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
  auto ort_memory_info = std::make_unique<OrtMemoryInfo>("FakeGPU", OrtAllocatorType::OrtDeviceAllocator,
                                                         ort_device, OrtMemTypeDefault);

  // Create the fake arena allocator with Shrink support.
  auto fake_allocator = MakeFakeArenaAllocator(ort_memory_info.get(), /*with_shrink=*/true);
  FakeArenaOrtAllocator* fake_alloc_ptr = &fake_allocator;

  auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
  auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(), ort_memory_info.get());
  std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);

  g_fake_allocator_for_test = fake_alloc_ptr;
  ort_ep->CreateAllocator = FakeCreateAllocator;
  test_plugin_ep::g_test_ort_ep_factory.ReleaseAllocator = FakeReleaseAllocator;

  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_EQ(allocators.size(), 1u);

  // The allocator supports Shrink, so it should be wrapped as IArena.
  auto* arena = allocators[0]->AsArena();
  ASSERT_NE(arena, nullptr) << "Shrink-capable plugin allocator must be exposed as IArena";

  // Shrink should forward to the fake allocator's Shrink callback.
  ASSERT_EQ(fake_alloc_ptr->shrink_call_count, 0);
  auto status = arena->Shrink();
  ASSERT_TRUE(status.IsOK());
  EXPECT_EQ(fake_alloc_ptr->shrink_call_count, 1);
}

TEST(PluginExecutionProviderTest, CreatePreferredAllocators_NonShrinkAllocatorNotExposedAsArena) {
  auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
  auto ort_memory_info = std::make_unique<OrtMemoryInfo>("FakeGPU", OrtAllocatorType::OrtDeviceAllocator,
                                                         ort_device, OrtMemTypeDefault);

  auto fake_allocator = MakeFakeArenaAllocator(ort_memory_info.get(), /*with_shrink=*/false);
  FakeArenaOrtAllocator* fake_alloc_ptr = &fake_allocator;

  auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
  auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(), ort_memory_info.get());
  std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);

  g_fake_allocator_for_test = fake_alloc_ptr;
  ort_ep->CreateAllocator = FakeCreateAllocator;
  test_plugin_ep::g_test_ort_ep_factory.ReleaseAllocator = FakeReleaseAllocator;

  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_EQ(allocators.size(), 1u);

  // Without Shrink, the allocator should NOT be exposed as IArena.
  EXPECT_EQ(allocators[0]->AsArena(), nullptr)
      << "Non-Shrink allocator must not be exposed as IArena";
}

TEST(PluginExecutionProviderTest, IsGraphCaptureEnabled) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    // NULL function pointer should return false (default behavior).
    ort_ep->IsGraphCaptureEnabled = nullptr;
    ASSERT_FALSE(ep->IsGraphCaptureEnabled());
  }

  {
    // Non-NULL implementation returning true.
    // IsGraphCaptured and ReplayGraph must also be set for IsGraphCaptureEnabled() to return true.
    auto graph_capture_enabled = [](const OrtEp* /*this_ptr*/) noexcept -> bool {
      return true;
    };
    auto is_graph_captured = [](const OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> bool {
      return false;
    };
    auto replay_graph = [](OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> ::OrtStatus* {
      return nullptr;
    };
    ort_ep->IsGraphCaptureEnabled = graph_capture_enabled;
    ort_ep->IsGraphCaptured = is_graph_captured;
    ort_ep->ReplayGraph = replay_graph;
    ASSERT_TRUE(ep->IsGraphCaptureEnabled());
    ort_ep->IsGraphCaptureEnabled = nullptr;  // Restore.
    ort_ep->IsGraphCaptured = nullptr;        // Restore.
    ort_ep->ReplayGraph = nullptr;            // Restore.
  }

  {
    // Non-NULL implementation returning false.
    auto graph_capture_disabled = [](const OrtEp* /*this_ptr*/) noexcept -> bool {
      return false;
    };
    ort_ep->IsGraphCaptureEnabled = graph_capture_disabled;
    ASSERT_FALSE(ep->IsGraphCaptureEnabled());
  }

  {
    // Backward compatibility: version < 26 should return false even if function pointer is set.
    auto graph_capture_enabled = [](const OrtEp* /*this_ptr*/) noexcept -> bool {
      return true;
    };
    ort_ep->IsGraphCaptureEnabled = graph_capture_enabled;
    ort_ep->ort_version_supported = 25;
    ASSERT_FALSE(ep->IsGraphCaptureEnabled());
    ort_ep->ort_version_supported = ORT_API_VERSION;  // Restore.
  }

  {
    // IsGraphCaptureEnabled returns true but IsGraphCaptured is NULL.
    // Should return false because ORT-managed graph capture requires IsGraphCaptured.
    auto graph_capture_enabled = [](const OrtEp* /*this_ptr*/) noexcept -> bool {
      return true;
    };
    auto replay_graph = [](OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> ::OrtStatus* {
      return nullptr;
    };
    ort_ep->IsGraphCaptureEnabled = graph_capture_enabled;
    ort_ep->IsGraphCaptured = nullptr;
    ort_ep->ReplayGraph = replay_graph;
    ASSERT_FALSE(ep->IsGraphCaptureEnabled());
    ort_ep->IsGraphCaptureEnabled = nullptr;  // Restore.
    ort_ep->ReplayGraph = nullptr;            // Restore.
  }

  {
    // IsGraphCaptureEnabled returns true but ReplayGraph is NULL.
    // Should return false because ORT-managed graph capture requires ReplayGraph.
    auto graph_capture_enabled = [](const OrtEp* /*this_ptr*/) noexcept -> bool {
      return true;
    };
    auto is_graph_captured = [](const OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> bool {
      return false;
    };
    ort_ep->IsGraphCaptureEnabled = graph_capture_enabled;
    ort_ep->IsGraphCaptured = is_graph_captured;
    ort_ep->ReplayGraph = nullptr;
    ASSERT_FALSE(ep->IsGraphCaptureEnabled());
    ort_ep->IsGraphCaptureEnabled = nullptr;  // Restore.
    ort_ep->IsGraphCaptured = nullptr;        // Restore.
  }
}

TEST(PluginExecutionProviderTest, IsGraphCaptured) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    // NULL function pointer should return false (default behavior).
    ort_ep->IsGraphCaptured = nullptr;
    ASSERT_FALSE(ep->IsGraphCaptured(0));
  }

  {
    // Non-NULL implementation that checks graph_annotation_id.
    auto graph_captured_for_id_42 = [](const OrtEp* /*this_ptr*/, int graph_annotation_id) noexcept -> bool {
      return graph_annotation_id == 42;
    };
    ort_ep->IsGraphCaptured = graph_captured_for_id_42;
    ASSERT_TRUE(ep->IsGraphCaptured(42));
    ASSERT_FALSE(ep->IsGraphCaptured(0));
    ASSERT_FALSE(ep->IsGraphCaptured(-1));
  }

  {
    // Backward compatibility: version < 26 should return false even if function pointer is set.
    auto always_captured = [](const OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> bool {
      return true;
    };
    ort_ep->IsGraphCaptured = always_captured;
    ort_ep->ort_version_supported = 25;
    ASSERT_FALSE(ep->IsGraphCaptured(0));
    ort_ep->ort_version_supported = ORT_API_VERSION;  // Restore.
  }
}

TEST(PluginExecutionProviderTest, ReplayGraph) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    // NULL function pointer should return OK (default behavior).
    ort_ep->ReplayGraph = nullptr;
    ASSERT_STATUS_OK(ep->ReplayGraph(0));
  }

  {
    // Non-NULL implementation returning OK.
    auto replay_ok = [](OrtEp* /*this_ptr*/, int /*graph_annotation_id*/) noexcept -> ::OrtStatus* {
      return nullptr;
    };
    ort_ep->ReplayGraph = replay_ok;
    ASSERT_STATUS_OK(ep->ReplayGraph(0));
  }

  {
    // Non-NULL implementation returning an error.
    auto replay_fail = [](OrtEp* this_ptr, int /*graph_annotation_id*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "Graph replay failed");
    };
    ort_ep->ReplayGraph = replay_fail;
    auto status = ep->ReplayGraph(0);
    ASSERT_FALSE(status.IsOK());
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Graph replay failed"));
  }

  {
    // Backward compatibility: version < 26 should return OK even if function pointer is set.
    auto replay_fail = [](OrtEp* this_ptr, int /*graph_annotation_id*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "Should not be called");
    };
    ort_ep->ReplayGraph = replay_fail;
    ort_ep->ort_version_supported = 25;
    ASSERT_STATUS_OK(ep->ReplayGraph(0));
    ort_ep->ort_version_supported = ORT_API_VERSION;  // Restore.
  }
}

TEST(PluginExecutionProviderTest, GetGraphCaptureNodeAssignmentPolicy) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    // NULL function pointer should return ALL_NODES_ON_EP (strictest default).
    ort_ep->GetGraphCaptureNodeAssignmentPolicy = nullptr;
    ASSERT_EQ(ep->GetGraphCaptureNodeAssignmentPolicy(), OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP);
  }

  {
    // Non-NULL implementation returning ALL_NODES_ON_EP.
    auto all_nodes_on_ep = [](const OrtEp* /*this_ptr*/) noexcept -> OrtGraphCaptureNodeAssignmentPolicy {
      return OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP;
    };
    ort_ep->GetGraphCaptureNodeAssignmentPolicy = all_nodes_on_ep;
    ASSERT_EQ(ep->GetGraphCaptureNodeAssignmentPolicy(), OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP);
  }

  {
    // Non-NULL implementation returning ALLOW_CPU_FOR_SHAPES.
    auto allow_cpu = [](const OrtEp* /*this_ptr*/) noexcept -> OrtGraphCaptureNodeAssignmentPolicy {
      return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
    };
    ort_ep->GetGraphCaptureNodeAssignmentPolicy = allow_cpu;
    ASSERT_EQ(ep->GetGraphCaptureNodeAssignmentPolicy(), OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES);
  }

  {
    // Backward compatibility: version < 26 should return ALL_NODES_ON_EP even if function pointer is set.
    auto allow_cpu = [](const OrtEp* /*this_ptr*/) noexcept -> OrtGraphCaptureNodeAssignmentPolicy {
      return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
    };
    ort_ep->GetGraphCaptureNodeAssignmentPolicy = allow_cpu;
    ort_ep->ort_version_supported = 25;
    ASSERT_EQ(ep->GetGraphCaptureNodeAssignmentPolicy(), OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP);
    ort_ep->ort_version_supported = ORT_API_VERSION;  // Restore.
  }
}

// Helper: create a no-threshold resource accountant via the real factory (config ",").
static IResourceAccountant* CreateNoThresholdAccountant(std::optional<ResourceAccountantMap>& acc_map) {
  ConfigOptions config;
  EXPECT_STATUS_OK(config.AddConfigEntry(kOrtSessionOptionsResourceCudaPartitioningSettings, ","));
  EXPECT_STATUS_OK(CreateAccountants(config, /*model_path=*/{}, acc_map));
  auto it = acc_map->find(kCudaExecutionProvider);
  return it != acc_map->end() ? it->second.get() : nullptr;
}

// Helper: call GetCapability on a mock EP with a no-threshold accountant, returning the accountant for inspection.
static IResourceAccountant* CallGetCapabilityWithAccountant(
    IExecutionProvider& ep,
    test_plugin_ep::TestOrtEp* ort_ep,
    std::optional<ResourceAccountantMap>& acc_map) {
  ort_ep->GetCapability = GetCapabilityTakeAllNodesOneGroup;

  std::shared_ptr<Model> model;
  EXPECT_STATUS_OK(Model::Load(ORT_TSTR("testdata/add_mul_add.onnx"), model, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  auto* accountant = CreateNoThresholdAccountant(acc_map);
  EXPECT_NE(accountant, nullptr);
  EXPECT_FALSE(accountant->GetThreshold().has_value());

  GraphViewer graph_viewer(model->MainGraph());
  auto& logger = DefaultLoggingManager().DefaultLogger();
  ep.SetLogger(&logger);
  ep.GetCapability(graph_viewer,
                   test_plugin_ep::MockKernelLookup(),
                   GraphOptimizerRegistry(nullptr, nullptr, &logger),
                   accountant);
  return accountant;
}

// GetAvailableResource returns TotalBytes → threshold should be set to that value.
TEST(PluginExecutionProviderTest, GetAvailableResource_SetsThresholdFromTotalBytes) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  constexpr uint64_t kBudget = 42000;

  ort_ep->GetAvailableResource = [](const OrtEp* /*this_ptr*/, OrtResourceCount* available) noexcept -> OrtStatus* {
    *available = OrtResourceCount::FromTotalBytes(42000);
    return nullptr;
  };

  std::optional<ResourceAccountantMap> acc_map;
  auto* accountant = CallGetCapabilityWithAccountant(*ep, ort_ep, acc_map);

  ASSERT_TRUE(accountant->GetThreshold().has_value());
  EXPECT_EQ(std::get<size_t>(*accountant->GetThreshold()), static_cast<size_t>(kBudget));
}

// GetAvailableResource returns None → threshold should remain unset (EP has no info).
TEST(PluginExecutionProviderTest, GetAvailableResource_NoneKindLeavesThresholdUnset) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  ort_ep->GetAvailableResource = [](const OrtEp* /*this_ptr*/, OrtResourceCount* available) noexcept -> OrtStatus* {
    *available = OrtResourceCount::None();
    return nullptr;
  };

  std::optional<ResourceAccountantMap> acc_map;
  auto* accountant = CallGetCapabilityWithAccountant(*ep, ort_ep, acc_map);

  EXPECT_FALSE(accountant->GetThreshold().has_value());
}

// GetAvailableResource returns an error status → threshold should remain unset.
TEST(PluginExecutionProviderTest, GetAvailableResource_ErrorLeavesThresholdUnset) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  ort_ep->GetAvailableResource = [](const OrtEp* this_ptr, OrtResourceCount* /*available*/) noexcept -> OrtStatus* {
    auto* test_ep = static_cast<const test_plugin_ep::TestOrtEp*>(this_ptr);
    return test_ep->ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, "device unavailable");
  };

  std::optional<ResourceAccountantMap> acc_map;
  auto* accountant = CallGetCapabilityWithAccountant(*ep, ort_ep, acc_map);

  EXPECT_FALSE(accountant->GetThreshold().has_value());
}

// GetAvailableResource is nullptr (old EP) → threshold should remain unset, no crash.
TEST(PluginExecutionProviderTest, GetAvailableResource_NullCallbackLeavesThresholdUnset) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  ort_ep->GetAvailableResource = nullptr;

  std::optional<ResourceAccountantMap> acc_map;
  auto* accountant = CallGetCapabilityWithAccountant(*ep, ort_ep, acc_map);

  EXPECT_FALSE(accountant->GetThreshold().has_value());
}

}  // namespace onnxruntime::test
