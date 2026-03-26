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
    EXPECT_TRUE(rules.rules[0].prefix_match);
  }

  // Test multiple annotations for one device
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Annotation1, Annotation2)", rules));
    ASSERT_EQ(rules.rules.size(), 2u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_TRUE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP1");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_TRUE(rules.rules[1].prefix_match);
  }

  // Test multiple devices
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Annotation1); EP2(Annotation2)", rules));
    ASSERT_EQ(rules.rules.size(), 2u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_TRUE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP2");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_TRUE(rules.rules[1].prefix_match);
  }

  // Test exact match
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(=Annotation1)", rules));
    ASSERT_EQ(rules.rules.size(), 1u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_FALSE(rules.rules[0].prefix_match);
  }

  // Test trimming whitespace
  {
    LayeringRules rules;
    ASSERT_STATUS_OK(LayeringRules::FromConfigString("  EP1  (  Annotation1  ,  =Annotation2  )  ;  EP2  (  Annotation3  )  ", rules));
    ASSERT_EQ(rules.rules.size(), 3u);
    EXPECT_EQ(rules.rules[0].device, "EP1");
    EXPECT_EQ(rules.rules[0].annotation, "Annotation1");
    EXPECT_TRUE(rules.rules[0].prefix_match);
    EXPECT_EQ(rules.rules[1].device, "EP1");
    EXPECT_EQ(rules.rules[1].annotation, "Annotation2");
    EXPECT_FALSE(rules.rules[1].prefix_match);
    EXPECT_EQ(rules.rules[2].device, "EP2");
    EXPECT_EQ(rules.rules[2].annotation, "Annotation3");
    EXPECT_TRUE(rules.rules[2].prefix_match);
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

  // Duplicate prefix annotation within the same device
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(Ann1, Ann1)", rules).IsOK());

  // Duplicate prefix annotation across different devices
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(Ann1); EP2(Ann1)", rules).IsOK());

  // Duplicate exact annotation within the same device
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(=Ann1, =Ann1)", rules).IsOK());

  // Duplicate exact annotation across different devices
  EXPECT_FALSE(LayeringRules::FromConfigString("EP1(=Ann1); EP2(=Ann1)", rules).IsOK());

  // Same annotation but different match types (prefix vs exact) should be OK
  ASSERT_STATUS_OK(LayeringRules::FromConfigString("EP1(Ann1, =Ann1)", rules));
  ASSERT_EQ(rules.rules.size(), 2u);
  EXPECT_TRUE(rules.rules[0].prefix_match);
  EXPECT_FALSE(rules.rules[1].prefix_match);
}

TEST(LayeringIndexTest, MakeNodeUnassigned_PreservesEpRuleMapping) {
  // Scenario: All nodes for a rule are unassigned in one graph.
  // ep_name_to_layering_indices_ must still contain the rule so that
  // sibling subgraphs (or the same graph on a subsequent pass) can still
  // use it for filtering.

  // 1. Setup Graph with two nodes, both annotated with the same rule
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
  NodeArg* mid_arg = &graph.GetOrCreateNodeArg("mid", &type_proto);
  NodeArg* output_arg = &graph.GetOrCreateNodeArg("output", &type_proto);

  Node& node0 = graph.AddNode("node0", "Abs", "Node 0", {input_arg}, {mid_arg});
  node0.SetLayeringAnnotation("RuleA");
  Node& node1 = graph.AddNode("node1", "Abs", "Node 1", {mid_arg}, {output_arg});
  node1.SetLayeringAnnotation("RuleA");

  ASSERT_STATUS_OK(graph.Resolve());

  // 2. Setup Rules: RuleA -> DeviceA
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  // 3. Create Index
  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // Both nodes should be assigned
  ASSERT_TRUE(index.GetNodeAssignment(graph, node0.Index()).has_value());
  ASSERT_TRUE(index.GetNodeAssignment(graph, node1.Index()).has_value());

  // 3. Unassign both nodes (simulating EP failing to claim them)
  index.MakeNodeUnassigned(graph, node0.Index());
  index.MakeNodeUnassigned(graph, node1.Index());

  // Nodes should be unassigned
  EXPECT_FALSE(index.GetNodeAssignment(graph, node0.Index()).has_value());
  EXPECT_FALSE(index.GetNodeAssignment(graph, node1.Index()).has_value());

  // 4. CRITICAL: ep_name_to_layering_indices_ must still map DeviceA -> {0}
  // so that other graphs/passes can still use this rule for filtering.
  auto rules_opt = index.GetLayeringRulesForThisEp("DeviceA");
  ASSERT_TRUE(rules_opt.has_value()) << "EP-to-rule mapping should not be erased when nodes are unassigned";
  EXPECT_EQ(rules_opt->get().count(0), 1u);
}

TEST(LayeringIndexTest, UpdateAfterFullUnassignment_RestoresVisibility) {
  // Scenario: All nodes for a rule are unassigned, then Update() adds
  // a new node matching the same rule. The new node must be visible
  // to the EP via GetLayeringRulesForThisEp.

  // 1. Setup Graph with one annotated node
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

  Node& node0 = graph.AddNode("node0", "Abs", "Node 0", {input_arg}, {output_arg});
  node0.SetLayeringAnnotation("RuleA");
  ASSERT_STATUS_OK(graph.Resolve());

  // 2. Setup Rules: RuleA -> DeviceA
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));
  ASSERT_TRUE(index.GetNodeAssignment(graph, node0.Index()).has_value());

  // 3. Unassign the only node
  index.MakeNodeUnassigned(graph, node0.Index());
  EXPECT_FALSE(index.GetNodeAssignment(graph, node0.Index()).has_value());

  // 4. Simulate layout transform adding a new node with inherited annotation
  NodeArg* new_output_arg = &graph.GetOrCreateNodeArg("new_output", &type_proto);
  Node& new_node = graph.AddNode("new_node", "Abs", "Node with inherited assignment",
                                 {output_arg}, {new_output_arg});
  new_node.SetLayeringAnnotation("RuleA");  // Inherits parent's annotation
  ASSERT_STATUS_OK(graph.Resolve());

  // Record the new node index
  NodeIndex new_node_index = new_node.Index();

  // 5. Update index with the new node
  std::vector<NodeIndex> new_nodes = {new_node_index};
  index.Update(graph, new_nodes);

  // 6. New node should be assigned to rule 0
  auto assign = index.GetNodeAssignment(graph, new_node.Index());
  ASSERT_TRUE(assign.has_value());
  EXPECT_EQ(*assign, 0u);

  // 7. CRITICAL: The rule must still be visible for DeviceA
  auto rules_opt = index.GetLayeringRulesForThisEp("DeviceA");
  ASSERT_TRUE(rules_opt.has_value()) << "EP-to-rule mapping must be intact for Update to be effective";
  EXPECT_EQ(rules_opt->get().count(0), 1u);
}

// ============================================================================
// Tests for graph_partitioner.cc LayeringIndex integration
// These tests exercise behaviors from GetCapabilityForEP, InlineNodes, and
// the partitioning pipeline when a LayeringIndex is present.
// ============================================================================

// Helper to create a simple linear graph: input -> node0 -> node1 -> ... -> output
namespace {

struct SimpleGraphHelper {
  std::unique_ptr<Model> model;
  Graph* graph = nullptr;
  std::vector<NodeIndex> node_indices;

  static SimpleGraphHelper Create(int num_nodes, const std::string& op_type = "Abs") {
    SimpleGraphHelper h;
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 12;
    h.model = std::make_unique<Model>("test_model", false, ModelMetaData(), PathString(),
                                      IOnnxRuntimeOpSchemaRegistryList(),
                                      domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                      DefaultLoggingManager().DefaultLogger());
    h.graph = &h.model->MainGraph();

    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    NodeArg* prev_arg = &h.graph->GetOrCreateNodeArg("input", &type_proto);

    for (int i = 0; i < num_nodes; ++i) {
      std::string out_name = (i == num_nodes - 1) ? "output" : "mid_" + std::to_string(i);
      NodeArg* out_arg = &h.graph->GetOrCreateNodeArg(out_name, &type_proto);
      Node& node = h.graph->AddNode("node_" + std::to_string(i), op_type,
                                    "Node " + std::to_string(i), {prev_arg}, {out_arg});
      h.node_indices.push_back(node.Index());
      prev_arg = out_arg;
    }
    return h;
  }
};

LayeringIndex CreateTwoEpIndex(const Graph& graph,
                               const std::string& ep_a, const std::string& annotation_a,
                               const std::string& ep_b, const std::string& annotation_b) {
  LayeringRules rules;
  rules.rules.push_back({ep_a, annotation_a, false});  // Index 0
  rules.rules.push_back({ep_b, annotation_b, false});  // Index 1

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map[ep_a].insert(0);
  ep_map[ep_b].insert(1);

  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = ep_a;
  rule_map[1] = ep_b;

  return LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));
}

}  // namespace

TEST(LayeringIndexPartitionerTest, FilteredGraphViewerExcludesOtherEpNodes) {
  // Validates the filtering logic in create_graph_viewer (GetCapabilityForEP):
  // When layering_index is present, nodes assigned to other EPs should be excluded
  // from the GraphViewer presented to the current EP.

  // Setup: 3-node chain, node0 -> RuleA (DeviceA), node1 -> unannotated, node2 -> RuleB (DeviceB)
  auto h = SimpleGraphHelper::Create(3);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);
  node0->SetLayeringAnnotation("RuleA");
  node2->SetLayeringAnnotation("RuleB");
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  // Verify: From DeviceA's perspective, node2 should be excluded
  auto rules_a = index.GetLayeringRulesForThisEp("DeviceA");
  ASSERT_TRUE(rules_a.has_value());

  // node0 should be assigned to rule 0 (DeviceA)
  auto assign0 = index.GetNodeAssignment(*h.graph, h.node_indices[0]);
  ASSERT_TRUE(assign0.has_value());
  EXPECT_EQ(*assign0, 0u);

  // node1 should be unassigned (available to any EP)
  auto assign1 = index.GetNodeAssignment(*h.graph, h.node_indices[1]);
  EXPECT_FALSE(assign1.has_value());

  // node2 should be assigned to rule 1 (DeviceB)
  auto assign2 = index.GetNodeAssignment(*h.graph, h.node_indices[2]);
  ASSERT_TRUE(assign2.has_value());
  EXPECT_EQ(*assign2, 1u);

  // Simulate the filtering logic from create_graph_viewer:
  // For DeviceA: include nodes with no assignment OR assignment in DeviceA's rules
  InlinedVector<const Node*> filtered_for_device_a;
  for (auto& node : h.graph->Nodes()) {
    auto rule_idx_opt = index.GetNodeAssignment(*h.graph, node.Index());
    bool include = true;
    if (rule_idx_opt) {
      // Node has assignment - include only if it belongs to DeviceA
      if (rules_a->get().count(*rule_idx_opt) == 0) {
        include = false;
      }
    }
    if (include) {
      filtered_for_device_a.push_back(&node);
    }
  }

  // DeviceA should see node0 (assigned to it) and node1 (unassigned), but NOT node2
  EXPECT_EQ(filtered_for_device_a.size(), 2u);
  bool found_node0 = false, found_node1 = false, found_node2 = false;
  for (const auto* n : filtered_for_device_a) {
    if (n->Index() == h.node_indices[0]) found_node0 = true;
    if (n->Index() == h.node_indices[1]) found_node1 = true;
    if (n->Index() == h.node_indices[2]) found_node2 = true;
  }
  EXPECT_TRUE(found_node0) << "DeviceA's assigned node should be included";
  EXPECT_TRUE(found_node1) << "Unassigned node should be included for any EP";
  EXPECT_FALSE(found_node2) << "DeviceB's assigned node should be excluded from DeviceA's view";
}

TEST(LayeringIndexPartitionerTest, FilteredGraphViewerForDeviceBExcludesDeviceANodes) {
  // Mirror of the above test but from DeviceB's perspective.

  auto h = SimpleGraphHelper::Create(3);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);
  node0->SetLayeringAnnotation("RuleA");
  node2->SetLayeringAnnotation("RuleB");
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  auto rules_b = index.GetLayeringRulesForThisEp("DeviceB");
  ASSERT_TRUE(rules_b.has_value());

  // Simulate filtering for DeviceB
  InlinedVector<const Node*> filtered_for_device_b;
  for (auto& node : h.graph->Nodes()) {
    auto rule_idx_opt = index.GetNodeAssignment(*h.graph, node.Index());
    bool include = true;
    if (rule_idx_opt) {
      if (rules_b->get().count(*rule_idx_opt) == 0) {
        include = false;
      }
    }
    if (include) {
      filtered_for_device_b.push_back(&node);
    }
  }

  // DeviceB should see node1 (unassigned) and node2 (assigned to it), but NOT node0
  EXPECT_EQ(filtered_for_device_b.size(), 2u);
  bool found_node0 = false, found_node1 = false, found_node2 = false;
  for (const auto* n : filtered_for_device_b) {
    if (n->Index() == h.node_indices[0]) found_node0 = true;
    if (n->Index() == h.node_indices[1]) found_node1 = true;
    if (n->Index() == h.node_indices[2]) found_node2 = true;
  }
  EXPECT_FALSE(found_node0) << "DeviceA's assigned node should be excluded from DeviceB's view";
  EXPECT_TRUE(found_node1) << "Unassigned node should be included for any EP";
  EXPECT_TRUE(found_node2) << "DeviceB's assigned node should be included";
}

TEST(LayeringIndexPartitionerTest, ResetUnclaimedNodesRemovesAssignment) {
  // Validates the reset_assignment_unclaimed_nodes logic:
  // Nodes that were pre-assigned to an EP via layering but NOT claimed in capabilities
  // should be unassigned so subsequent EPs can pick them up.

  auto h = SimpleGraphHelper::Create(4);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node1 = h.graph->GetNode(h.node_indices[1]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);

  node0->SetLayeringAnnotation("RuleA");
  node1->SetLayeringAnnotation("RuleA");
  node2->SetLayeringAnnotation("RuleA");
  ASSERT_STATUS_OK(h.graph->Resolve());

  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  auto index = LayeringIndex::Create(*h.graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // All 3 nodes should be assigned initially
  ASSERT_TRUE(index.GetNodeAssignment(*h.graph, h.node_indices[0]).has_value());
  ASSERT_TRUE(index.GetNodeAssignment(*h.graph, h.node_indices[1]).has_value());
  ASSERT_TRUE(index.GetNodeAssignment(*h.graph, h.node_indices[2]).has_value());

  // Simulate: EP only claims node0 and node2 (not node1)
  InlinedHashSet<NodeIndex> claimed;
  claimed.insert(h.node_indices[0]);
  claimed.insert(h.node_indices[2]);

  auto ep_rules_opt = index.GetLayeringRulesForThisEp("DeviceA");
  ASSERT_TRUE(ep_rules_opt.has_value());
  const auto& ep_rules = ep_rules_opt->get();

  // Replicate reset_assignment_unclaimed_nodes logic:
  // For each assigned-filtered-in node, if not claimed, unassign it
  std::vector<NodeIndex> assigned_filtered_in = {h.node_indices[0], h.node_indices[1], h.node_indices[2]};
  for (auto node_index : assigned_filtered_in) {
    if (claimed.count(node_index) == 0) {
      auto rule_idx_opt = index.GetNodeAssignment(*h.graph, node_index);
      if (rule_idx_opt && ep_rules.count(*rule_idx_opt) > 0) {
        index.MakeNodeUnassigned(*h.graph, node_index);
      }
    }
  }

  // node0 and node2 should still be assigned
  EXPECT_TRUE(index.GetNodeAssignment(*h.graph, h.node_indices[0]).has_value());
  EXPECT_TRUE(index.GetNodeAssignment(*h.graph, h.node_indices[2]).has_value());
  // node1 should be unassigned (not claimed by EP)
  EXPECT_FALSE(index.GetNodeAssignment(*h.graph, h.node_indices[1]).has_value());
}

TEST(LayeringIndexPartitionerTest, UpdateAfterLayoutTransformAddsNewNodes) {
  // Validates the LayeringIndex update after layout transformation creates new nodes.
  // In GetCapabilityForEP, after layout transform, new nodes with inherited annotations
  // are added and the index is updated.

  auto h = SimpleGraphHelper::Create(1);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  node0->SetLayeringAnnotation("RuleA");
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  // Record the max node index before "layout transformation"
  const NodeIndex first_new_node = h.graph->MaxNodeIndex();

  // Simulate layout transformation adding new nodes with inherited annotation
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  NodeArg* extra_out = &h.graph->GetOrCreateNodeArg("extra_output", &type_proto);
  NodeArg* output_arg = &h.graph->GetOrCreateNodeArg("output", nullptr);  // reuse existing
  Node& new_node = h.graph->AddNode("new_node", "Abs", "Node with inherited annotation",
                                    {output_arg}, {extra_out});
  new_node.SetLayeringAnnotation("RuleA");  // Inherits parent's annotation
  ASSERT_STATUS_OK(h.graph->Resolve());

  const NodeIndex end_node = h.graph->MaxNodeIndex();

  // Collect new node indices (as done in graph_partitioner.cc)
  InlinedVector<NodeIndex> new_node_indices;
  for (NodeIndex idx = first_new_node; idx < end_node; ++idx) {
    if (h.graph->GetNode(idx) != nullptr) {
      new_node_indices.push_back(idx);
    }
  }

  // Update index
  ASSERT_FALSE(new_node_indices.empty());
  index.Update(*h.graph, new_node_indices);

  // New node should be assigned to rule 0 (DeviceA)
  auto assign = index.GetNodeAssignment(*h.graph, new_node.Index());
  ASSERT_TRUE(assign.has_value());
  EXPECT_EQ(*assign, 0u);

  // And the annotation string should be on the node
  EXPECT_EQ(new_node.GetLayeringAnnotation(), "RuleA");
}

TEST(LayeringIndexPartitionerTest, UpdateWithUnannotatedNewNodeRemainsUnassigned) {
  // New nodes created by layout transform that do NOT have annotations
  // should remain unassigned after Update.

  auto h = SimpleGraphHelper::Create(1);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  node0->SetLayeringAnnotation("RuleA");
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  // Add a new node WITHOUT annotation
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  NodeArg* extra_out = &h.graph->GetOrCreateNodeArg("extra_output", &type_proto);
  NodeArg* output_arg = &h.graph->GetOrCreateNodeArg("output", nullptr);
  Node& new_node = h.graph->AddNode("unannotated_node", "Abs", "No annotation",
                                    {output_arg}, {extra_out});
  // Deliberately NOT setting annotation
  ASSERT_STATUS_OK(h.graph->Resolve());

  std::vector<NodeIndex> new_nodes = {new_node.Index()};
  index.Update(*h.graph, new_nodes);

  // New node should remain unassigned
  auto assign = index.GetNodeAssignment(*h.graph, new_node.Index());
  EXPECT_FALSE(assign.has_value());
}

TEST(LayeringIndexPartitionerTest, InlineAnnotationMaterialization) {
  // Validates the InlineNodes logic where a node has an inherited-only assignment
  // (no explicit annotation string) and the annotation is materialized before inlining.
  // This tests the code path:
  //   if (layering_index != nullptr && !has_explicit_annotation) {
  //     auto rule_idx = layering_index->GetNodeAssignment(graph, node->Index());
  //     if (rule_idx) { ... node->SetLayeringAnnotation(rules.rules[*rule_idx].annotation); }
  //   }

  // Setup: A graph where a node is assigned via inheritance (subgraph scenario)
  // but has no explicit annotation string on it.
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

  // Create a node without explicit annotation
  Node& node = graph.AddNode("inherited_node", "Abs", "Node with inherited assignment",
                             {input_arg}, {output_arg});
  ASSERT_STATUS_OK(graph.Resolve());

  // Create index where the node is somehow assigned (e.g., through inheritance)
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA", false});  // Index 0

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";

  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // The node has no annotation, so it shouldn't be assigned yet
  ASSERT_TRUE(node.GetLayeringAnnotation().empty());
  EXPECT_FALSE(index.GetNodeAssignment(graph, node.Index()).has_value());

  // Now simulate what InlineNodes does: manually annotate and update
  // This simulates the case where GetNodeAssignment returns a value
  // for a node in a subgraph that inherited its parent's assignment.
  node.SetLayeringAnnotation("RuleA");
  std::vector<NodeIndex> updated = {node.Index()};
  index.Update(graph, updated);

  // After materialization + update, the node should be properly assigned
  auto assign = index.GetNodeAssignment(graph, node.Index());
  ASSERT_TRUE(assign.has_value());
  EXPECT_EQ(*assign, 0u);

  // And the annotation string should be on the node
  EXPECT_EQ(node.GetLayeringAnnotation(), "RuleA");
}

TEST(LayeringIndexPartitionerTest, UpdateBatchMultipleNewAnnotatedNodes) {
  // Tests that Update correctly handles a batch of multiple new nodes,
  // some annotated with different rules. This mirrors the behavior after
  // layout transformation creates several new nodes.

  auto h = SimpleGraphHelper::Create(1);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  node0->SetLayeringAnnotation("RuleA");
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  // Add 3 new nodes: one for RuleA, one for RuleB, one unannotated
  NodeArg* out1 = &h.graph->GetOrCreateNodeArg("new_out1", &type_proto);
  NodeArg* out2 = &h.graph->GetOrCreateNodeArg("new_out2", &type_proto);
  NodeArg* out3 = &h.graph->GetOrCreateNodeArg("new_out3", &type_proto);
  NodeArg* output = &h.graph->GetOrCreateNodeArg("output", nullptr);

  Node& new_a = h.graph->AddNode("new_a", "Abs", "", {output}, {out1});
  new_a.SetLayeringAnnotation("RuleA");

  Node& new_b = h.graph->AddNode("new_b", "Abs", "", {out1}, {out2});
  new_b.SetLayeringAnnotation("RuleB");

  Node& new_none = h.graph->AddNode("new_none", "Abs", "", {out2}, {out3});
  // No annotation

  ASSERT_STATUS_OK(h.graph->Resolve());

  std::vector<NodeIndex> new_nodes = {new_a.Index(), new_b.Index(), new_none.Index()};
  index.Update(*h.graph, new_nodes);

  // new_a -> RuleA -> rule index 0
  auto assign_a = index.GetNodeAssignment(*h.graph, new_a.Index());
  ASSERT_TRUE(assign_a.has_value());
  EXPECT_EQ(*assign_a, 0u);

  // new_b -> RuleB -> rule index 1
  auto assign_b = index.GetNodeAssignment(*h.graph, new_b.Index());
  ASSERT_TRUE(assign_b.has_value());
  EXPECT_EQ(*assign_b, 1u);

  // new_none -> unassigned
  auto assign_none = index.GetNodeAssignment(*h.graph, new_none.Index());
  EXPECT_FALSE(assign_none.has_value());
}

TEST(LayeringIndexPartitionerTest, MakeUnassignedThenReassignViaPrefixRule) {
  // Test that prefix rules work correctly after unassign+update cycle.
  // This covers the interaction between MakeNodeUnassigned, prefix matching,
  // and Update.

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
  node.SetLayeringAnnotation("Layer_GPU_Compute");
  ASSERT_STATUS_OK(graph.Resolve());

  // Prefix rule: "Layer_GPU" matches "Layer_GPU_Compute"
  LayeringRules rules;
  rules.rules.push_back({"GPUDevice", "Layer_GPU", true});  // Index 0, prefix match

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["GPUDevice"].insert(0);
  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "GPUDevice";

  auto index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  // Node should be assigned via prefix match
  auto assign = index.GetNodeAssignment(graph, node.Index());
  ASSERT_TRUE(assign.has_value());
  EXPECT_EQ(*assign, 0u);

  // Unassign the node
  index.MakeNodeUnassigned(graph, node.Index());
  EXPECT_FALSE(index.GetNodeAssignment(graph, node.Index()).has_value());

  // Add a new node with a different annotation that also matches the prefix
  NodeArg* new_out = &graph.GetOrCreateNodeArg("new_output", &type_proto);
  Node& new_node = graph.AddNode("new_node", "Abs", "Node with inherited annotation",
                                 {output_arg}, {new_out});
  new_node.SetLayeringAnnotation("Layer_GPU_Memory");
  ASSERT_STATUS_OK(graph.Resolve());

  std::vector<NodeIndex> new_nodes = {new_node.Index()};
  index.Update(graph, new_nodes);

  // New node should also be assigned via prefix match
  auto new_assign = index.GetNodeAssignment(graph, new_node.Index());
  ASSERT_TRUE(new_assign.has_value());
  EXPECT_EQ(*new_assign, 0u);
}

TEST(LayeringIndexPartitionerTest, NoLayeringIndexAllNodesVisible) {
  // When layering_index is nullptr (no layering configuration),
  // all nodes should be visible to all EPs. This verifies the baseline
  // behavior that the filtering code path is only active when layering is enabled.

  auto h = SimpleGraphHelper::Create(3);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);

  // Even if nodes have annotations, without a LayeringIndex, everything is visible
  node0->SetLayeringAnnotation("RuleA");
  node2->SetLayeringAnnotation("RuleB");
  ASSERT_STATUS_OK(h.graph->Resolve());

  // Without LayeringIndex, a standard GraphViewer should see all nodes
  GraphViewer viewer(*h.graph);
  EXPECT_EQ(viewer.NumberOfNodes(), 3);

  // All nodes accessible
  EXPECT_NE(viewer.GetNode(h.node_indices[0]), nullptr);
  EXPECT_NE(viewer.GetNode(h.node_indices[1]), nullptr);
  EXPECT_NE(viewer.GetNode(h.node_indices[2]), nullptr);
}

TEST(LayeringIndexPartitionerTest, EpWithNoLayeringRulesSeesAllUnassignedNodes) {
  // An EP that has no rules in the LayeringIndex (i.e., GetLayeringRulesForThisEp returns nullopt)
  // should still see unassigned nodes, but nodes assigned to other EPs are excluded.
  // This is the behavior for a CPU fallback EP not mentioned in layering config,
  // as implemented in graph_partitioner.cc create_graph_viewer:
  //   if (!rules_opt || rules_opt->get().count(*rule_idx_opt) == 0) { include = false; }

  auto h = SimpleGraphHelper::Create(4);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);
  node0->SetLayeringAnnotation("RuleA");
  node2->SetLayeringAnnotation("RuleB");
  // node1 and node3 are unannotated
  ASSERT_STATUS_OK(h.graph->Resolve());

  auto index = CreateTwoEpIndex(*h.graph, "DeviceA", "RuleA", "DeviceB", "RuleB");

  // "CPUDevice" has no rules in the index
  auto rules_cpu = index.GetLayeringRulesForThisEp("CPUDevice");
  EXPECT_FALSE(rules_cpu.has_value());

  // Replicate create_graph_viewer filtering logic for an EP with no rules.
  // When rules_opt is nullopt, any node with an assignment is excluded:
  //   if (!rules_opt || ...) { include = false; }
  // Unassigned nodes remain included.
  InlinedVector<const Node*> filtered_for_cpu;
  for (auto& node : h.graph->Nodes()) {
    auto rule_idx_opt = index.GetNodeAssignment(*h.graph, node.Index());
    bool include = true;
    if (rule_idx_opt) {
      if (!rules_cpu || rules_cpu->get().count(*rule_idx_opt) == 0) {
        include = false;
      }
    }
    if (include) {
      filtered_for_cpu.push_back(&node);
    }
  }

  // CPUDevice should see only the 2 unassigned nodes (node1, node3).
  // node0 (RuleA/DeviceA) and node2 (RuleB/DeviceB) are excluded.
  EXPECT_EQ(filtered_for_cpu.size(), 2u);

  bool found[4] = {};
  for (const auto* n : filtered_for_cpu) {
    for (size_t i = 0; i < std::size(found); ++i) {
      if (n->Index() == h.node_indices[i]) found[i] = true;
    }
  }
  EXPECT_FALSE(found[0]) << "node0 assigned to DeviceA should be excluded";
  EXPECT_TRUE(found[1]) << "node1 unassigned should be included";
  EXPECT_FALSE(found[2]) << "node2 assigned to DeviceB should be excluded";
  EXPECT_TRUE(found[3]) << "node3 unassigned should be included";
}
TEST(LayeringIndexPartitionerTest, MultipleRulesForSameEp) {
  // An EP can have multiple rules assigned to it. All nodes matching any of its
  // rules should be visible to it, while nodes matching other EP rules should not.

  auto h = SimpleGraphHelper::Create(4);
  auto* node0 = h.graph->GetNode(h.node_indices[0]);
  auto* node1 = h.graph->GetNode(h.node_indices[1]);
  auto* node2 = h.graph->GetNode(h.node_indices[2]);

  node0->SetLayeringAnnotation("RuleA1");
  node1->SetLayeringAnnotation("RuleA2");
  node2->SetLayeringAnnotation("RuleB");
  // node3 unannotated
  ASSERT_STATUS_OK(h.graph->Resolve());

  // DeviceA has two rules: RuleA1 (index 0) and RuleA2 (index 1)
  // DeviceB has one rule: RuleB (index 2)
  LayeringRules rules;
  rules.rules.push_back({"DeviceA", "RuleA1", false});  // Index 0
  rules.rules.push_back({"DeviceA", "RuleA2", false});  // Index 1
  rules.rules.push_back({"DeviceB", "RuleB", false});   // Index 2

  LayeringIndex::EpNameToLayeringIndices ep_map;
  ep_map["DeviceA"].insert(0);
  ep_map["DeviceA"].insert(1);
  ep_map["DeviceB"].insert(2);

  LayeringIndex::LayeringIndexToEpName rule_map;
  rule_map[0] = "DeviceA";
  rule_map[1] = "DeviceA";
  rule_map[2] = "DeviceB";

  auto index = LayeringIndex::Create(*h.graph, std::move(ep_map), std::move(rule_map), std::move(rules));

  auto rules_a = index.GetLayeringRulesForThisEp("DeviceA");
  ASSERT_TRUE(rules_a.has_value());
  EXPECT_EQ(rules_a->get().size(), 2u);  // Both rule indices 0 and 1

  // Simulate filtering for DeviceA
  InlinedVector<const Node*> filtered_for_a;
  for (auto& node : h.graph->Nodes()) {
    auto rule_idx_opt = index.GetNodeAssignment(*h.graph, node.Index());
    bool include = true;
    if (rule_idx_opt) {
      if (rules_a->get().count(*rule_idx_opt) == 0) {
        include = false;
      }
    }
    if (include) {
      filtered_for_a.push_back(&node);
    }
  }

  // DeviceA should see node0, node1 (both its rules), and node3 (unassigned) = 3 nodes
  // node2 (RuleB/DeviceB) should be excluded
  EXPECT_EQ(filtered_for_a.size(), 3u);

  bool found[4] = {};
  for (const auto* n : filtered_for_a) {
    for (int i = 0; i < 4; ++i) {
      if (n->Index() == h.node_indices[i]) found[i] = true;
    }
  }
  EXPECT_TRUE(found[0]);   // node0 - RuleA1
  EXPECT_TRUE(found[1]);   // node1 - RuleA2
  EXPECT_FALSE(found[2]);  // node2 - RuleB (excluded)
  EXPECT_TRUE(found[3]);   // node3 - unassigned
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)