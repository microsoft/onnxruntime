// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/framework/execution_providers.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/layering_annotations.h"
#include "core/session/abi_devices.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ortdevice.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"  // For Model, Graph
#include "gtest/gtest.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

TEST(LayeringRuleMatcherTest, ExactMatches) {
  LayeringRules rules;
  rules.rules.push_back({"Device1", "Annotation1", false});  // Index 0
  rules.rules.push_back({"Device2", "Annotation2", false});  // Index 1

  LayeringRuleMatcher matcher(rules);

  {
    auto result = matcher.Match("Annotation1");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }
  {
    auto result = matcher.Match("Annotation2");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }
  {
    auto result = matcher.Match("Annotation3");
    EXPECT_FALSE(result.has_value());
  }
}

TEST(LayeringRuleMatcherTest, PrefixMatches) {
  LayeringRules rules;
  rules.rules.push_back({"Device1", "Prefix1", true});  // Index 0: =Prefix1
  rules.rules.push_back({"Device2", "Pre", true});      // Index 1: =Pre

  LayeringRuleMatcher matcher(rules);

  // "Prefix1Suffix" matches "Prefix1" (idx 0) and "Pre" (idx 1). 0 < 1, so 0.
  {
    auto result = matcher.Match("Prefix1Suffix");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // "PreSuffix" matches "Pre" (idx 1). "Prefix1" does not match.
  {
    auto result = matcher.Match("PreSuffix");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }

  // "Other" matches nothing
  {
    auto result = matcher.Match("Other");
    EXPECT_FALSE(result.has_value());
  }
}

TEST(LayeringRuleMatcherTest, PriorityPrefixOverExact) {
  // Prefix matches should take precedence over exact matches regardless of order.

  // Case 1: Prefix rule comes before Exact rule
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "A", true});    // Index 0: =A (Prefix)
    rules.rules.push_back({"Device2", "AB", false});  // Index 1: AB (Exact)

    LayeringRuleMatcher matcher(rules);
    // "AB" matches prefix "A" (idx 0) and exact "AB" (idx 1).
    // Since prefix matches are checked first and returned if found, we expect 0.
    auto result = matcher.Match("AB");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // Case 2: Exact rule comes before Prefix rule
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "AB", false});  // Index 0: AB (Exact)
    rules.rules.push_back({"Device2", "A", true});    // Index 1: =A (Prefix)

    LayeringRuleMatcher matcher(rules);
    // "AB" matches exact "AB" (idx 0) and prefix "A" (idx 1).
    // Priority says Prefix matches are returned first.
    auto result = matcher.Match("AB");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }
}

TEST(LayeringRuleMatcherTest, LongestOrShortestPrefixPriority) {
  // If multiple prefix rules match, the one with the lowest index (earliest in config) wins.

  // Case 1: Shorter prefix first
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "A", true});   // Index 0
    rules.rules.push_back({"Device2", "AB", true});  // Index 1

    LayeringRuleMatcher matcher(rules);
    // "ABC" matches "A" (0) and "AB" (1). Since 0 < 1, best match is 0.
    auto result = matcher.Match("ABC");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // Case 2: Longer prefix first
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "AB", true});  // Index 0
    rules.rules.push_back({"Device2", "A", true});   // Index 1

    LayeringRuleMatcher matcher(rules);
    // "ABC" matches "AB" (0) and "A" (1). Since 0 < 1, best match is 0.
    auto result = matcher.Match("ABC");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }
}

TEST(LayeringRuleMatcherTest, OverlappingExactMatchPriority) {
  // If duplicates exist, first one wins.
  LayeringRules rules;
  rules.rules.push_back({"Device1", "A", false});  // Index 0
  rules.rules.push_back({"Device2", "A", false});  // Index 1

  LayeringRuleMatcher matcher(rules);
  auto result = matcher.Match("A");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 0u);
}

TEST(LayeringRuleMatcherTest, OverlappingPrefixMatchPriority) {
  // If duplicates exist, first one wins.
  LayeringRules rules;
  rules.rules.push_back({"Device1", "A", true});  // Index 0
  rules.rules.push_back({"Device2", "A", true});  // Index 1

  LayeringRuleMatcher matcher(rules);
  auto result = matcher.Match("AB");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 0u);
}

namespace {

// Helper to construct OrtEpDevice wrappers for testing
struct TestEpDevice {
  std::string ep_name;
  OrtHardwareDevice hw_device;
  bool has_hw_device = false;
  OrtMemoryInfo mem_info;
  bool has_mem_info = false;

  // We need to keep the structures alive while OrtEpDevice points to them
  OrtEpDevice Get() const {
    OrtEpDevice ep;
    ep.ep_name = ep_name;
    ep.device = has_hw_device ? &hw_device : nullptr;
    ep.device_memory_info = has_mem_info ? &mem_info : nullptr;
    return ep;
  }
};

TestEpDevice CreateEp(const std::string& name) {
  TestEpDevice ep;
  ep.ep_name = name;
  return ep;
}

TestEpDevice CreateHwEp(const std::string& name, OrtHardwareDeviceType type, uint32_t vendor_id = 0,
                        uint32_t device_id = 0, const std::string& vendor_str = std::string()) {
  TestEpDevice ep;
  ep.ep_name = name;
  ep.hw_device = {type, vendor_id, device_id, vendor_str, {}};
  ep.has_hw_device = true;
  return ep;
}

TestEpDevice CreateMemEp(const std::string& name, OrtDevice::DeviceType type, int device_id = 0) {
  TestEpDevice ep;
  ep.ep_name = name;
  // Note: OrtMemoryInfo name doesn't matter for logic now, but required for ctor
  ep.mem_info = OrtMemoryInfo("TestMem", OrtAllocatorType::OrtDeviceAllocator,
                              OrtDevice(type, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                                        static_cast<OrtDevice::DeviceId>(device_id)),
                              OrtMemType::OrtMemTypeDefault);
  ep.has_mem_info = true;
  return ep;
}

}  // namespace

TEST(EpLayeringMatcherTest, MatchCPU) {
  LayerAnnotation rule = {"CPU", "Anno1", false};

  // Case 1: EP Name kCpuExecutionProvider
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCpuExecutionProvider);
  }

  // Case 2: Hardware Device CPU
  {
    auto test_ep = CreateHwEp("SomeCPU_EP", OrtHardwareDeviceType_CPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "SomeCPU_EP");
  }

  // Case 3: Memory Info CPU
  {
    auto test_ep = CreateMemEp("MemCPU_EP", OrtDevice::CPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MemCPU_EP");
  }
}

TEST(EpLayeringMatcherTest, MatchGPU) {
  LayerAnnotation rule = {"GPU", "Anno1", false};

  // Case 1: Hardware Device GPU
  {
    auto test_ep = CreateHwEp("MyGPU_EP", OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyGPU_EP");
  }

  // Case 2: Memory Info GPU
  {
    auto test_ep = CreateMemEp("MemGPU_EP", OrtDevice::GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MemGPU_EP");
  }

  // Case 3: Heuristic kCudaExecutionProvider
  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }

  // Case 4: Heuristic kDmlExecutionProvider
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kDmlExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorString) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};

  // Case 1: Vendor String Match
  {
    auto test_ep = CreateHwEp("MyNvidia_EP", OrtHardwareDeviceType_GPU, 0, 0, "NVIDIA");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyNvidia_EP");
  }

  // Case 2: Vendor String Mismatch
  {
    auto test_ep = CreateHwEp("MyAMD_EP", OrtHardwareDeviceType_GPU, 0, 0, "AMD");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorId) {
  LayerAnnotation rule_intel = {"gpu:intel", "Anno1", false};
  LayerAnnotation rule_nvidia = {"gpu:nvidia", "Anno2", false};
  LayerAnnotation rule_amd = {"gpu:amd", "Anno3", false};

  // Case 1: Vendor ID Match Intel
  {
    auto test_ep = CreateHwEp("Intel_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::INTEL);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_intel);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Intel_EP");
  }

  // Case 2: Vendor ID Match Nvidia
  {
    auto test_ep = CreateHwEp("Nvidia_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::NVIDIA);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_nvidia);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Nvidia_EP");
  }

  // Case 3: Vendor ID Match AMD
  {
    auto test_ep = CreateHwEp("AMD_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::AMD);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_amd);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "AMD_EP");
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Heuristic) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};

  // Case 1: kCudaExecutionProvider -> nvidia
  {
    // Need an EP with GPU HW type but generic vendor info to trigger the heuristic
    auto test_ep_hw = CreateHwEp(kCudaExecutionProvider, OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep_hw.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};

    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Index) {
  LayerAnnotation rule = {"gpu:1", "Anno1", false};

  // Case 1: ID Match
  {
    auto test_ep = CreateHwEp("GPU1", OrtHardwareDeviceType_GPU, 0, 1);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};

    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "GPU1");
  }

  // Case 2: ID Mismatch
  {
    auto test_ep = CreateHwEp("GPU0", OrtHardwareDeviceType_GPU, 0, 0);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchAccelerator) {
  LayerAnnotation rule = {"accelerator", "Anno1", false};

  // Case 1: CPU EP should NOT match
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }

  // Case 2: Custom EP, No HW/Mem info, considered accelerator
  {
    auto test_ep = CreateEp("MyCustomAccel");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyCustomAccel");
  }

  // Case 3: GPU HW is an accelerator
  {
    auto test_ep = CreateHwEp("MyGPU", OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyGPU");
  }
}

TEST(EpLayeringMatcherTest, MatchNPU) {
  LayerAnnotation rule = {"npu", "Anno1", false};

  // Case 1: Hardware NPU
  {
    auto test_ep = CreateHwEp("MyNPU", OrtHardwareDeviceType_NPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyNPU");
  }

  // Case 2: QNN Heuristic
  {
    auto test_ep = CreateEp(kQnnExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kQnnExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchFPGA) {
  LayerAnnotation rule = {"fpga", "Anno1", false};

  // Case 1: MemInfo says FPGA
  {
    auto test_ep = CreateMemEp("MyFPGA", OrtDevice::FPGA);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyFPGA");
  }
}

TEST(EpLayeringMatcherTest, MatchDirectDesignators) {
  LayerAnnotation rule_cuda = {"cuda", "A", false};
  LayerAnnotation rule_dml = {"dml", "B", false};

  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_cuda);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_dml);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kDmlExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchExactEPName) {
  LayerAnnotation rule = {"MyCustomEP", "Anno1", false};

  {
    auto test_ep = CreateEp("MyCustomEP");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyCustomEP");
  }
}

namespace {

// Minimal concrete implementation of IExecutionProvider for testing
class MockExecutionProvider : public IExecutionProvider {
 public:
  MockExecutionProvider(const std::string& type, OrtDevice device)
      : IExecutionProvider(type, device) {}

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override { return nullptr; }
};

}  // namespace

TEST(EpLayeringMatcherTest, MatchExecutionProviders_CPU) {
  LayerAnnotation rule = {"CPU", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU provider
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add a GPU provider (should be skipped for CPU rule)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCpuExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_GPU) {
  LayerAnnotation rule = {"GPU", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU provider (should be skipped)
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add CUDA provider (GPU)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCudaExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_GPU_Specific) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};  // Assumes heuristics or vendor ID logic
  ExecutionProviders providers;

  // Add CPU provider
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add CUDA provider (NVIDIA vendor ID)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider,
                                                        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCudaExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_NoMatch) {
  LayerAnnotation rule = {"GPU", "Anno1", false};
  ExecutionProviders providers;

  // Only CPU provider available
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  EXPECT_FALSE(result.has_value());
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_Accelerator) {
  LayerAnnotation rule = {"accelerator", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add custom accelerator
  auto accel_ep = std::make_shared<MockExecutionProvider>("MyAccel", OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add("MyAccel", accel_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "MyAccel");
}

TEST(LayeringIndexTest, AssignNodesBasedOnAnnotations) {
  // 1. Setup Graph
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  // Create nodes
  // Node 0: "AnnotatedNode" -> Annotated with "RuleA"
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  NodeArg* input_arg = &graph.GetOrCreateNodeArg("input", &type_proto);
  NodeArg* output_arg0 = &graph.GetOrCreateNodeArg("output0", &type_proto);
  Node& node0 = graph.AddNode("node0", "Abs", "Node 0", {input_arg}, {output_arg0});
  node0.SetLayeringAnnotation("RuleA");

  // Node 1: "UnannotatedNode" -> No annotation
  NodeArg* output_arg1 = &graph.GetOrCreateNodeArg("output1", &type_proto);
  Node& node1 = graph.AddNode("node1", "Abs", "Node 1", {output_arg0}, {output_arg1});
  // No annotation

  // Node 2: "AnnotatedNode2" -> Annotated with "RuleB"
  NodeArg* output_arg2 = &graph.GetOrCreateNodeArg("output2", &type_proto);
  Node& node2 = graph.AddNode("node2", "Abs", "Node 2", {output_arg1}, {output_arg2});
  node2.SetLayeringAnnotation("RuleB");

  ASSERT_STATUS_OK(graph.Resolve());

  // 2. Setup Rules and Matcher
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0
  rules.rules.push_back({"DeviceB", "RuleB", false});  // Index 1
  LayeringRuleMatcher matcher(rules);

  // 3. Setup Pre-computed Mappings (simulating Partitioning Manager)
  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  ep_map["DeviceB"].insert(1);

  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";
  rule_map[1] = "DeviceB";

  // 4. Create LayeringIndex
  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // 5. Verify Assignments
  // Node 0: Annotated "RuleA" -> Index 0 -> DeviceA
  auto assign0 = index.GetNodeAssignment(graph, node0.Index());
  ASSERT_TRUE(assign0.has_value());
  EXPECT_EQ(*assign0, 0u);

  // Node 1: Unannotated -> Should generally map to nothing (unless defaulting logic exists,
  // but current impl leaves unannotated in main graph as unassigned)
  auto assign1 = index.GetNodeAssignment(graph, node1.Index());
  EXPECT_FALSE(assign1.has_value());

  // Node 2: Annotated "RuleB" -> Index 1 -> DeviceB
  auto assign2 = index.GetNodeAssignment(graph, node2.Index());
  ASSERT_TRUE(assign2.has_value());
  EXPECT_EQ(*assign2, 1u);
}

TEST(LayeringIndexTest, AssignNodeWithInvalidEpMapping) {
  // Scenario: Node annotated with a rule that maps to an EP that is NOT present/valid

  // 1. Setup Graph with one node annotated "RuleX"
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  NodeArg* input_arg = &graph.GetOrCreateNodeArg("input", &type_proto);
  NodeArg* output_arg = &graph.GetOrCreateNodeArg("output", &type_proto);

  Node& node = graph.AddNode("node", "Abs", "Node", {input_arg}, {output_arg});
  node.SetLayeringAnnotation("RuleX");

  ASSERT_STATUS_OK(graph.Resolve());

  // 2. Setup Rules: RuleX exists at index 0, maps to "PhantomDevice"
  LayeringRules rules;
  rules.rules.push_back({"PhantomDevice", "RuleX", false});  // Index 0

  // 3. Setup Mappings: But "PhantomDevice" is NOT in the mappings (simulating EP unavailable)
  LayeringIndex::EpNameToLayeringIndices ep_map;
  // ep_map["PhantomDevice"] is empty/missing

  LayeringIndex::LayeringIndexToEpName rule_map;
  // rule_map[0] is missing

  // 4. Create Index
  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));
  // 5. Verify: Node should NOT be assigned because the mapped EP is missing
  auto assign = index.GetNodeAssignment(graph, node.Index());
  EXPECT_FALSE(assign.has_value());
}

TEST(LayeringIndexTest, SubgraphInheritance) {
  // Scenario: Annotated Node containing a subgraph.
  // Nodes inside subgraph (unannotated) should inherit parent's assignment.

  // 1. Setup Parent Graph
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  NodeArg* cond_arg = &graph.GetOrCreateNodeArg("cond", &type_proto);
  NodeArg* output_arg = &graph.GetOrCreateNodeArg("output", &type_proto);

  // Create "If" node
  Node& if_node = graph.AddNode("if_node", "If", "If Node", {cond_arg}, {output_arg});
  if_node.SetLayeringAnnotation("RuleA");  // Annotate Parent

  auto build_subgraph = [](ONNX_NAMESPACE::GraphProto& proto, const std::string& graph_name,
                           const std::string& node_name, const std::string& input_name, const std::string& output_name) {
    proto.set_name(graph_name);
    // Inputs: Implicit from outer scope for 'cond'

    auto* node = proto.add_node();
    node->set_name(node_name);
    node->set_op_type("Identity");
    node->add_input(input_name);
    node->add_output(output_name);

    auto* out_vi = proto.add_output();
    out_vi->set_name(output_name);
    out_vi->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  };

  // Create Subgraph (then_branch)
  ONNX_NAMESPACE::GraphProto then_graph_proto;
  build_subgraph(then_graph_proto, "then_graph", "sub_node", "cond", "sub_out");
  if_node.AddAttribute("then_branch", then_graph_proto);

  // Create 'else_branch'
  ONNX_NAMESPACE::GraphProto else_graph_proto;
  build_subgraph(else_graph_proto, "else_graph", "else_sub_node", "cond", "else_sub_out");
  if_node.AddAttribute("else_branch", else_graph_proto);

  // First Resolve to create subgraph instances
  ASSERT_STATUS_OK(graph.Resolve());

  // Get subgraph instances (checked to ensure they exist)
  Graph* then_graph = if_node.GetMutableGraphAttribute("then_branch");
  ASSERT_NE(then_graph, nullptr);
  Graph* else_graph = if_node.GetMutableGraphAttribute("else_branch");
  ASSERT_NE(else_graph, nullptr);

  // 2. Setup Rules
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  // 3. Create Index
  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // 4. Verify Parent Assignment
  auto assign_parent = index.GetNodeAssignment(graph, if_node.Index());
  ASSERT_TRUE(assign_parent.has_value());
  EXPECT_EQ(*assign_parent, 0u);

  // 5. Verify Subgraph Node Assignment (Inheritance)
  bool validated_then = false;
  for (const auto& node : then_graph->Nodes()) {
    if (node.OpType() == "Identity") {
      auto assign_sub = index.GetNodeAssignment(*then_graph, node.Index());
      ASSERT_TRUE(assign_sub.has_value()) << "Subgraph node should inherit parent annotation";
      EXPECT_EQ(*assign_sub, 0u);
      validated_then = true;
    }
  }
  ASSERT_TRUE(validated_then);
}

TEST(LayeringIndexTest, SubgraphOverride) {
  // Scenario: Annotated Node containing a subgraph.
  // Node inside subgraph HAS annotation -> Should override inheritance.

  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  NodeArg* cond_arg = &graph.GetOrCreateNodeArg("cond", &type_proto);
  NodeArg* output_arg = &graph.GetOrCreateNodeArg("output", &type_proto);

  Node& if_node = graph.AddNode("if_node", "If", "If Node", {cond_arg}, {output_arg});
  if_node.SetLayeringAnnotation("RuleA");  // Annotate Parent = Rule A (Index 0)

  auto build_subgraph = [](ONNX_NAMESPACE::GraphProto& proto, const std::string& graph_name,
                           const std::string& node_name, const std::string& input_name, const std::string& output_name) {
    proto.set_name(graph_name);

    auto* node = proto.add_node();
    node->set_name(node_name);
    node->set_op_type("Identity");
    node->add_input(input_name);
    node->add_output(output_name);

    auto* out_vi = proto.add_output();
    out_vi->set_name(output_name);
    out_vi->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  };

  ONNX_NAMESPACE::GraphProto then_graph_proto;
  build_subgraph(then_graph_proto, "then_graph", "sub_node", "cond", "sub_out");
  if_node.AddAttribute("then_branch", then_graph_proto);

  ONNX_NAMESPACE::GraphProto else_graph_proto;
  build_subgraph(else_graph_proto, "else_graph", "else_sub_node", "cond", "else_sub_out");
  if_node.AddAttribute("else_branch", else_graph_proto);

  ASSERT_STATUS_OK(graph.Resolve());

  Graph* then_graph = if_node.GetMutableGraphAttribute("then_branch");
  ASSERT_NE(then_graph, nullptr);

  // Find sub_node to set annotation
  Node* sub_node = nullptr;
  for (auto& node : then_graph->Nodes()) {
    if (node.Name() == "sub_node") {
      sub_node = &node;
      break;
    }
  }
  ASSERT_NE(sub_node, nullptr);

  // OVERRIDE: Annotate sub_node with Rule B
  sub_node->SetLayeringAnnotation("RuleB");

  // Rules: RuleA(0)->DeviceA, RuleB(1)->DeviceB
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});
  rules.rules.push_back({"DeviceB", "RuleB", false});

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  ep_map["DeviceB"].insert(1);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";
  rule_map[1] = "DeviceB";

  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // Verify Parent = 0
  auto assign_parent = index.GetNodeAssignment(graph, if_node.Index());
  ASSERT_TRUE(assign_parent.has_value());
  EXPECT_EQ(*assign_parent, 0u);

  // Verify Sub = 1 (Override)
  auto assign_sub = index.GetNodeAssignment(*then_graph, sub_node->Index());
  ASSERT_TRUE(assign_sub.has_value());
  EXPECT_EQ(*assign_sub, 1u);
}

TEST(LayeringIndexTest, UpdateIndex) {
  // 1. Setup Graph with one node
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  NodeArg* input_arg = &graph.GetOrCreateNodeArg("input", &type_proto);
  NodeArg* output_arg = &graph.GetOrCreateNodeArg("output", &type_proto);

  Node& node = graph.AddNode("node", "Abs", "Node", {input_arg}, {output_arg});
  ASSERT_STATUS_OK(graph.Resolve());

  // 2. Setup Rules and Index
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  // Creates index (node has no annotation, so not assigned)
  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));
  EXPECT_FALSE(index.GetNodeAssignment(graph, node.Index()).has_value());

  // 3. Update Node with Annotation
  node.SetLayeringAnnotation("RuleA");

  // 4. Call Update
  std::vector<NodeIndex> nodes_to_update = {node.Index()};
  index.Update(graph, nodes_to_update);

  // 5. Verify Assignment
  auto assignment = index.GetNodeAssignment(graph, node.Index());
  ASSERT_TRUE(assignment.has_value());
  EXPECT_EQ(*assignment, 0u);
}

TEST(LayeringRulesTest, LayeringRulesParsing) {
  // Test empty string
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("", rules));
    EXPECT_TRUE(rules.rules.empty());
  }

  // Test simple valid string
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Annotation1)", rules));
    ASSERT_EQ(rules.rules.size(), 1u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_FALSE(rules.rules[0].prefix_match);
  }

  // Test multiple annotations for one device
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Annotation1, Annotation2)", rules));
    ASSERT_EQ(rules.rules.size(), 2u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_FALSE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP1");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_FALSE(rules.rules[1].prefix_match);
  }

  // Test multiple devices
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Annotation1); EP2(Annotation2)", rules));
    ASSERT_EQ(rules.rules.size(), 2u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_FALSE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP2");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_FALSE(rules.rules[1].prefix_match);
  }

  // Test prefix match
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(=Annotation1)", rules));
    ASSERT_EQ(rules.rules.size(), 1u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_TRUE(rules.rules[0].prefix_match);
  }

  // Test trimming whitespace
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("  EP1  (  Annotation1  ,  =Annotation2  )  ;  EP2  (  Annotation3  )  ", rules));
    ASSERT_EQ(rules.rules.size(), 3u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_FALSE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP1");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_TRUE(rules.rules[1].prefix_match);
    EXPECT_EQ(rules.rules[2].device, "EP2");
    EXPECT_EQ(rules.rules[2].annotation, "Annotation3");
    EXPECT_FALSE(rules.rules[2].prefix_match);
  }
}

TEST(LayeringRulesTest, FromConfigString_InvalidFormat) {
  LayeringRules rules;

  // Error: Missing parentheses structure entirely
  EXPECT_FALSE(LayeringRules::FromConfigString("Device1Annotation1", rules).IsOK());

  // Error: Missing closing parenthesis
  EXPECT_FALSE(LayeringRules::FromConfigString("Device1(Annotation1", rules).IsOK());

  // Error: Missing opening parenthesis (or only closing present)
  EXPECT_FALSE(LayeringRules::FromConfigString("Device1Annotation1)", rules).IsOK());

  // Error: Parentheses reversed
  EXPECT_FALSE(LayeringRules::FromConfigString("Device1)Annotation1(", rules).IsOK());

  // Error: Empty device name (starts with parenthesis)
  EXPECT_FALSE(LayeringRules::FromConfigString("(Annotation1)", rules).IsOK());
}

TEST(LayeringRulesTest, FromConfigString_IgnoresEmptyEntries) {
  LayeringRules rules;
  // "; ;" should result in 0 rules but Status::OK
  ASSERT_STATUS_OK(LayeringRules::FromConfigString(";   ;", rules));
  EXPECT_TRUE(rules.rules.empty());
}

TEST(LayeringRulesTest, FromConfigString_RejectsDuplicateAnnotations) {
  LayeringRules rules;

  // Duplicate exact annotation within the same device
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(Ann1, Ann1)", rules).IsOK());

  // Duplicate exact annotation across different devices
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(Ann1); EP2(Ann1)", rules).IsOK());

  // Duplicate prefix annotation within the same device
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(=Ann1, =Ann1)", rules).IsOK());

  // Duplicate prefix annotation across different devices
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(=Ann1); EP2(=Ann1)", rules).IsOK());

  // Same annotation but different match types (exact vs prefix) should be OK
  ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Ann1, =Ann1)", rules));
  ASSERT_EQ(rules.rules.size(), 2u);
  EXPECT_FALSE(rules.rules[0].prefix_match);
  EXPECT_TRUE(rules.rules[1].prefix_match);
}
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)