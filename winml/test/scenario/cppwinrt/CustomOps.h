#include "test.h"
using VoidTest = void(*)();
using SetupTest = VoidTest;
struct CustomOpsTestApi
{
    SetupTest CustomOpsScenarioTestSetup;
    SetupTest CustomOpsScenarioGpuTestSetup;
    VoidTest CustomOperatorFusion;
    VoidTest CustomKernelWithBuiltInSchema;
    VoidTest CustomKernelWithCustomSchema;
};
const CustomOpsTestApi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(CustomOpsScenarioTest, CustomOpsScenarioTestSetup)
WINML_TEST(CustomOpsScenarioTest, CustomKernelWithBuiltInSchema)
WINML_TEST(CustomOpsScenarioTest, CustomKernelWithCustomSchema)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN_WITH_SETUP(CustomOpsScenarioGpuTest, CustomOpsScenarioGpuTestSetup)
WINML_TEST(CustomOpsScenarioGpuTest, CustomOperatorFusion)
WINML_TEST_CLASS_END()