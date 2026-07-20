// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/pool.h"
#include "default_providers.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
namespace onnxruntime {
namespace test {

// Execution providers excluded from the four opset-18 AveragePool ceil_mode +
// count_include_pad parity tests below. Shared by all four tests to prevent scoping
// drift. Two rationales:
//   - Most EPs (kTensorrt, kNvTensorRTRTX, kAcl, kOpenVINO, kDml, kWebGpu, kDnnl,
//     kCoreML, kQnn) do not implement the clamped-window divisor (PyTorch #183528),
//     so they return the pre-fix full-kernel-size average and would fail these tests.
//   - kCuda / kCudaNHWC DO support the semantics, but these opset-18 cases are
//     CPU-reference gate tests (the CUDA path has its own parity tests) and cuDNN-NHWC
//     can flap on the 2D case, so they are excluded here too.
//
// NOTE: do not confuse this with kPoolingEpsExcludedFromCeilCipTests (defined ~L1150), which
// has the OPPOSITE kCuda membership. This set is the CPU-reference GATE and therefore INCLUDES
// kCuda/kCudaNHWC in the exclusion list (CPU is the oracle here); that other set EXCLUDES
// kCuda/kCudaNHWC because there CUDA is the tested target. Pick the one matching your intent.
static const std::unordered_set<std::string> kPoolingEpsExcludedFromCeilCountIncludePadTests = {
    kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider,
    kNvTensorRTRTXExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider,
    kDmlExecutionProvider, kWebGpuExecutionProvider, kDnnlExecutionProvider,
    kCoreMLExecutionProvider, kQnnExecutionProvider};

template <typename T>
class PoolTest : public ::testing::Test {
};

using PoolTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(PoolTest, PoolTestTypes);

// Disable TensorRT on some of the tests because "pads" attribute is not supported

TEST(PoolTest, MaxPool) {
  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});

  std::vector<float> x_vals = {
      0.19151945412158966, 0.6221087574958801, 0.43772774934768677, 0.7853586077690125, 0.7799758315086365, 0.27259260416030884, 0.2764642536640167, 0.801872193813324,
      0.9581393599510193, 0.8759326338768005, 0.35781726241111755, 0.5009950995445251, 0.683462917804718, 0.7127020359039307, 0.37025076150894165, 0.5611962080001831,
      0.5030831694602966, 0.013768449425697327, 0.772826611995697, 0.8826411962509155, 0.36488598585128784, 0.6153962016105652, 0.07538124173879623, 0.3688240051269531,
      0.9331400990486145, 0.6513781547546387, 0.39720258116722107, 0.7887301445007324, 0.3168361186981201, 0.5680986642837524, 0.8691273927688599, 0.4361734092235565,
      0.802147626876831, 0.14376682043075562, 0.7042609453201294, 0.7045813202857971, 0.2187921106815338, 0.9248676300048828, 0.44214075803756714, 0.9093159437179565,
      0.05980922281742096, 0.18428708612918854, 0.047355279326438904, 0.6748809218406677, 0.5946247577667236, 0.5333101749420166, 0.043324064463377, 0.5614330768585205,
      0.32966843247413635, 0.5029668211936951, 0.11189431697130203, 0.6071937084197998, 0.5659446716308594, 0.006764062214642763, 0.617441713809967, 0.912122905254364,
      0.7905241250991821, 0.9920814633369446, 0.9588017463684082, 0.7919641137123108, 0.2852509617805481, 0.6249167323112488, 0.47809380292892456, 0.19567517936229706,

      0.382317453622818, 0.053873684257268906, 0.45164841413497925, 0.9820047616958618, 0.12394270300865173, 0.1193808987736702, 0.7385230660438538, 0.587303638458252,
      0.47163254022598267, 0.10712681710720062, 0.22921857237815857, 0.8999651670455933, 0.41675353050231934, 0.5358516573905945, 0.0062085166573524475, 0.3006417155265808,
      0.43689316511154175, 0.6121490001678467, 0.9181980490684509, 0.625736653804779, 0.7059975862503052, 0.14983370900154114, 0.7460634112358093, 0.8310070037841797,
      0.6337257623672485, 0.4383098781108856, 0.15257278084754944, 0.5684096217155457, 0.5282242894172668, 0.9514287710189819, 0.48035916686058044, 0.5025595426559448,
      0.5368781685829163, 0.8192020654678345, 0.05711563676595688, 0.6694217324256897, 0.7671166062355042, 0.7081153392791748, 0.7968671917915344, 0.5577608346939087,
      0.9658365249633789, 0.14715689420700073, 0.02964700013399124, 0.5938934683799744, 0.11406569927930832, 0.9508098363876343, 0.32570740580558777, 0.19361868500709534,
      0.4578116536140442, 0.9204025864601135, 0.8790691494941711, 0.252615749835968, 0.34800878167152405, 0.18258872628211975, 0.9017960429191589, 0.7065281867980957,
      0.7266584634780884, 0.900087833404541, 0.7791637778282166, 0.5991547703742981, 0.29112523794174194, 0.1513952612876892, 0.33517464995384216, 0.6575517654418945,

      0.07334254682064056, 0.055006396025419235, 0.32319480180740356, 0.5904818177223206, 0.8538985848426819, 0.2870624363422394, 0.17306722700595856, 0.13402120769023895,
      0.9946538209915161, 0.1794978678226471, 0.3175468146800995, 0.568291425704956, 0.009348574094474316, 0.9006485939025879, 0.9772414565086365, 0.5568946599960327,
      0.08477384597063065, 0.3330024778842926, 0.7284286618232727, 0.14243537187576294, 0.5524689555168152, 0.2730432450771332, 0.9744951128959656, 0.6677868962287903,
      0.2556532919406891, 0.1083114966750145, 0.7761807441711426, 0.7824779748916626, 0.7616038918495178, 0.9144031405448914, 0.6586228013038635, 0.568367600440979,
      0.20175568759441376, 0.6982963681221008, 0.952195405960083, 0.8899632692337036, 0.9935673475265503, 0.8187035322189331, 0.5451221466064453, 0.45125406980514526,
      0.8905571699142456, 0.9732648134231567, 0.5934113264083862, 0.36607450246810913, 0.3230946958065033, 0.8714232444763184, 0.2156340628862381, 0.7349451780319214,
      0.36561909317970276, 0.8016026020050049, 0.7827355861663818, 0.7013553977012634, 0.6227765679359436, 0.4936826527118683, 0.8405377268791199, 0.7120969891548157,
      0.4439089894294739, 0.031034860759973526, 0.36323976516723633, 0.7307217717170715, 0.475566565990448, 0.3444169759750366, 0.6408804059028625, 0.12620532512664795};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.9920814633369446, 0.9820047616958618, 0.9946538209915161};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TensorRT: result differs
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDmlExecutionProvider});
}

// Only CUDA kernel has float 16 support
// Disable for now, still investigating the issue with cudnn lib
#if defined(USE_CUDA) || defined(USE_COREML)
TEST(PoolTest, MaxPool_F16) {
#if defined(USE_CUDA)
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});

  std::vector<float> x_vals = {
      0.19151945412158966, 0.6221087574958801, 0.43772774934768677, 0.7853586077690125, 0.7799758315086365, 0.27259260416030884, 0.2764642536640167, 0.801872193813324,
      0.9581393599510193, 0.8759326338768005, 0.35781726241111755, 0.5009950995445251, 0.683462917804718, 0.7127020359039307, 0.37025076150894165, 0.5611962080001831,
      0.5030831694602966, 0.013768449425697327, 0.772826611995697, 0.8826411962509155, 0.36488598585128784, 0.6153962016105652, 0.07538124173879623, 0.3688240051269531,
      0.9331400990486145, 0.6513781547546387, 0.39720258116722107, 0.7887301445007324, 0.3168361186981201, 0.5680986642837524, 0.8691273927688599, 0.4361734092235565,
      0.802147626876831, 0.14376682043075562, 0.7042609453201294, 0.7045813202857971, 0.2187921106815338, 0.9248676300048828, 0.44214075803756714, 0.9093159437179565,
      0.05980922281742096, 0.18428708612918854, 0.047355279326438904, 0.6748809218406677, 0.5946247577667236, 0.5333101749420166, 0.043324064463377, 0.5614330768585205,
      0.32966843247413635, 0.5029668211936951, 0.11189431697130203, 0.6071937084197998, 0.5659446716308594, 0.006764062214642763, 0.617441713809967, 0.912122905254364,
      0.7905241250991821, 0.9920814633369446, 0.9588017463684082, 0.7919641137123108, 0.2852509617805481, 0.6249167323112488, 0.47809380292892456, 0.19567517936229706,

      0.382317453622818, 0.053873684257268906, 0.45164841413497925, 0.9820047616958618, 0.12394270300865173, 0.1193808987736702, 0.7385230660438538, 0.587303638458252,
      0.47163254022598267, 0.10712681710720062, 0.22921857237815857, 0.8999651670455933, 0.41675353050231934, 0.5358516573905945, 0.0062085166573524475, 0.3006417155265808,
      0.43689316511154175, 0.6121490001678467, 0.9181980490684509, 0.625736653804779, 0.7059975862503052, 0.14983370900154114, 0.7460634112358093, 0.8310070037841797,
      0.6337257623672485, 0.4383098781108856, 0.15257278084754944, 0.5684096217155457, 0.5282242894172668, 0.9514287710189819, 0.48035916686058044, 0.5025595426559448,
      0.5368781685829163, 0.8192020654678345, 0.05711563676595688, 0.6694217324256897, 0.7671166062355042, 0.7081153392791748, 0.7968671917915344, 0.5577608346939087,
      0.9658365249633789, 0.14715689420700073, 0.02964700013399124, 0.5938934683799744, 0.11406569927930832, 0.9508098363876343, 0.32570740580558777, 0.19361868500709534,
      0.4578116536140442, 0.9204025864601135, 0.8790691494941711, 0.252615749835968, 0.34800878167152405, 0.18258872628211975, 0.9017960429191589, 0.7065281867980957,
      0.7266584634780884, 0.900087833404541, 0.7791637778282166, 0.5991547703742981, 0.29112523794174194, 0.1513952612876892, 0.33517464995384216, 0.6575517654418945,

      0.07334254682064056, 0.055006396025419235, 0.32319480180740356, 0.5904818177223206, 0.8538985848426819, 0.2870624363422394, 0.17306722700595856, 0.13402120769023895,
      0.9946538209915161, 0.1794978678226471, 0.3175468146800995, 0.568291425704956, 0.009348574094474316, 0.9006485939025879, 0.9772414565086365, 0.5568946599960327,
      0.08477384597063065, 0.3330024778842926, 0.7284286618232727, 0.14243537187576294, 0.5524689555168152, 0.2730432450771332, 0.9744951128959656, 0.6677868962287903,
      0.2556532919406891, 0.1083114966750145, 0.7761807441711426, 0.7824779748916626, 0.7616038918495178, 0.9144031405448914, 0.6586228013038635, 0.568367600440979,
      0.20175568759441376, 0.6982963681221008, 0.952195405960083, 0.8899632692337036, 0.9935673475265503, 0.8187035322189331, 0.5451221466064453, 0.45125406980514526,
      0.8905571699142456, 0.9732648134231567, 0.5934113264083862, 0.36607450246810913, 0.3230946958065033, 0.8714232444763184, 0.2156340628862381, 0.7349451780319214,
      0.36561909317970276, 0.8016026020050049, 0.7827355861663818, 0.7013553977012634, 0.6227765679359436, 0.4936826527118683, 0.8405377268791199, 0.7120969891548157,
      0.4439089894294739, 0.031034860759973526, 0.36323976516723633, 0.7307217717170715, 0.475566565990448, 0.3444169759750366, 0.6408804059028625, 0.12620532512664795};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  int x_size = 1 * 3 * 8 * 8;
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.9920814633369446, 0.9820047616958618, 0.9946538209915161};

  std::vector<MLFloat16> f_X(x_size);
  std::vector<MLFloat16> f_Y(3);
  ConvertFloatToMLFloat16(x_vals.data(), f_X.data(), x_size);
  ConvertFloatToMLFloat16(expected_vals.data(), f_Y.data(), 3);

  test.AddInput<MLFloat16>("X", x_dims, f_X);
  test.AddOutput<MLFloat16>("Y", expected_dims, f_Y);
  // TensorRT: Assertion `!attrs.count("pads")' failed
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

static void MaxPool_8_WithIndexTest(bool has_index, int64_t storage_order = 0) {
  OpTester test("MaxPool", 8);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});
  test.AddAttribute("storage_order", storage_order);

  std::vector<float> x_vals = {
      0.19151945412158966, 0.6221087574958801, 0.43772774934768677, 0.7853586077690125, 0.7799758315086365, 0.27259260416030884, 0.2764642536640167, 0.801872193813324,
      0.9581393599510193, 0.8759326338768005, 0.35781726241111755, 0.5009950995445251, 0.683462917804718, 0.7127020359039307, 0.37025076150894165, 0.5611962080001831,
      0.5030831694602966, 0.013768449425697327, 0.772826611995697, 0.8826411962509155, 0.36488598585128784, 0.6153962016105652, 0.07538124173879623, 0.3688240051269531,
      0.9331400990486145, 0.6513781547546387, 0.39720258116722107, 0.7887301445007324, 0.3168361186981201, 0.5680986642837524, 0.8691273927688599, 0.4361734092235565,
      0.802147626876831, 0.14376682043075562, 0.7042609453201294, 0.7045813202857971, 0.2187921106815338, 0.9248676300048828, 0.44214075803756714, 0.9093159437179565,
      0.05980922281742096, 0.18428708612918854, 0.047355279326438904, 0.6748809218406677, 0.5946247577667236, 0.5333101749420166, 0.043324064463377, 0.5614330768585205,
      0.32966843247413635, 0.5029668211936951, 0.11189431697130203, 0.6071937084197998, 0.5659446716308594, 0.006764062214642763, 0.617441713809967, 0.912122905254364,
      0.7905241250991821, 0.9920814633369446, 0.9588017463684082, 0.7919641137123108, 0.2852509617805481, 0.6249167323112488, 0.47809380292892456, 0.19567517936229706,

      0.382317453622818, 0.053873684257268906, 0.45164841413497925, 0.9820047616958618, 0.12394270300865173, 0.1193808987736702, 0.7385230660438538, 0.587303638458252,
      0.47163254022598267, 0.10712681710720062, 0.22921857237815857, 0.8999651670455933, 0.41675353050231934, 0.5358516573905945, 0.0062085166573524475, 0.3006417155265808,
      0.43689316511154175, 0.6121490001678467, 0.9181980490684509, 0.625736653804779, 0.7059975862503052, 0.14983370900154114, 0.7460634112358093, 0.8310070037841797,
      0.6337257623672485, 0.4383098781108856, 0.15257278084754944, 0.5684096217155457, 0.5282242894172668, 0.9514287710189819, 0.48035916686058044, 0.5025595426559448,
      0.5368781685829163, 0.8192020654678345, 0.05711563676595688, 0.6694217324256897, 0.7671166062355042, 0.7081153392791748, 0.7968671917915344, 0.5577608346939087,
      0.9658365249633789, 0.14715689420700073, 0.02964700013399124, 0.5938934683799744, 0.11406569927930832, 0.9508098363876343, 0.32570740580558777, 0.19361868500709534,
      0.4578116536140442, 0.9204025864601135, 0.8790691494941711, 0.252615749835968, 0.34800878167152405, 0.18258872628211975, 0.9017960429191589, 0.7065281867980957,
      0.7266584634780884, 0.900087833404541, 0.7791637778282166, 0.5991547703742981, 0.29112523794174194, 0.1513952612876892, 0.33517464995384216, 0.6575517654418945,

      0.07334254682064056, 0.055006396025419235, 0.32319480180740356, 0.5904818177223206, 0.8538985848426819, 0.2870624363422394, 0.17306722700595856, 0.13402120769023895,
      0.9946538209915161, 0.1794978678226471, 0.3175468146800995, 0.568291425704956, 0.009348574094474316, 0.9006485939025879, 0.9772414565086365, 0.5568946599960327,
      0.08477384597063065, 0.3330024778842926, 0.7284286618232727, 0.14243537187576294, 0.5524689555168152, 0.2730432450771332, 0.9744951128959656, 0.6677868962287903,
      0.2556532919406891, 0.1083114966750145, 0.7761807441711426, 0.7824779748916626, 0.7616038918495178, 0.9144031405448914, 0.6586228013038635, 0.568367600440979,
      0.20175568759441376, 0.6982963681221008, 0.952195405960083, 0.8899632692337036, 0.9935673475265503, 0.8187035322189331, 0.5451221466064453, 0.45125406980514526,
      0.8905571699142456, 0.9732648134231567, 0.5934113264083862, 0.36607450246810913, 0.3230946958065033, 0.8714232444763184, 0.2156340628862381, 0.7349451780319214,
      0.36561909317970276, 0.8016026020050049, 0.7827355861663818, 0.7013553977012634, 0.6227765679359436, 0.4936826527118683, 0.8405377268791199, 0.7120969891548157,
      0.4439089894294739, 0.031034860759973526, 0.36323976516723633, 0.7307217717170715, 0.475566565990448, 0.3444169759750366, 0.6408804059028625, 0.12620532512664795};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.9920814633369446, 0.9820047616958618, 0.9946538209915161};
  std::vector<int64_t> expected_indices_row = {57, 67, 136};
  std::vector<int64_t> expected_indices_col = {15, 88, 129};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  if (has_index) {
    storage_order == 0 ? test.AddOutput<int64_t>("Indices", expected_dims, expected_indices_row)
                       : test.AddOutput<int64_t>("Indices", expected_dims, expected_indices_col);
  }
  // TODO: Enable the case for WebGPU once WGSL can support int64.
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kAclExecutionProvider,
            kOpenVINOExecutionProvider, kWebGpuExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_8_With_Index) {
  MaxPool_8_WithIndexTest(false);                      // row major
  MaxPool_8_WithIndexTest(true, 0 /*storage_order*/);  // row major
  MaxPool_8_WithIndexTest(true, 1 /*storage_order*/);  // col major
}

TEST(PoolTest, MaxPool1D_case1) {
  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<float> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<float> expected_vals = {2, 4, 6, 8};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool1D_case2) {
  OpTester test("MaxPool");
  // no padding
  test.AddAttribute("auto_pad", "VALID");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<float> x_vals = {1, 2, 3, 4, 5};
  std::vector<int64_t> x_dims = {1, 1, 5};
  // The last dim is (5-2+1)/1 = 4
  std::vector<int64_t> expected_dims = {1, 1, 4};
  std::vector<float> expected_vals = {2, 3, 4, 5};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool1D_case3) {
  OpTester test("MaxPool");
  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  // Pad one element
  test.AddAttribute("pads", std::vector<int64_t>{0, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<float> x_vals = {1, 2, 3, 4, 5};
  std::vector<int64_t> x_dims = {1, 1, 5};
  // Since we padded it, the last dim is larger compared to the case above
  std::vector<int64_t> expected_dims = {1, 1, 5};
  std::vector<float> expected_vals = {2, 3, 4, 5, 5};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

static void MaxPool1D_8_WithIndexTest(int64_t storage_order) {
  OpTester test("MaxPool", 8);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<float> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<float> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);

  // TODO: Enable the case for WebGPU once WGSL can support int64.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider, kWebGpuExecutionProvider});
}

TEST(PoolTest, MaxPool1D_8_With_Index) {
  MaxPool1D_8_WithIndexTest(0 /*storage_order*/);
  MaxPool1D_8_WithIndexTest(1 /*storage_order*/);
}

static void MaxPool1D_12_WithIndexTest_int8(int64_t storage_order) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<int8_t> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<int8_t> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<int8_t>("X", x_dims, x_vals);
  test.AddOutput<int8_t>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider});
}

static void MaxPool1D_12_WithIndexTest_uint8(int64_t storage_order) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<uint8_t> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<uint8_t> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<uint8_t>("X", x_dims, x_vals);
  test.AddOutput<uint8_t>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool1D_12_With_Index_8bits) {
  MaxPool1D_12_WithIndexTest_int8(0 /*storage_order*/);
  MaxPool1D_12_WithIndexTest_int8(1 /*storage_order*/);
  MaxPool1D_12_WithIndexTest_uint8(0 /*storage_order*/);
  MaxPool1D_12_WithIndexTest_uint8(1 /*storage_order*/);
}

// Used by MaxPool2D_uint8
template <typename InputIter>
void print_vector(std::ostream& os, const std::string& txt, InputIter begin, InputIter end) {
  os << txt;
  while (begin != end) {
    std::cout << uint16_t(*begin) << ", ";
    ++begin;
  }
  os << std::endl;
}

TEST(PoolTest, MaxPool2D_uint8) {
  OpTester test("MaxPool", 12);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{5, 5});
  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});

  std::vector<int64_t> output_shape = {1, 1, 5, 5};
  std::vector<uint8_t> output = {
      13, 14, 15, 15, 15,
      18, 19, 20, 20, 20,
      23, 24, 25, 25, 25,
      23, 24, 25, 25, 25,
      23, 24, 25, 25, 25};

  test.AddInput<uint8_t>("Input", {1, 1, 5, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});

  test.AddOutput<uint8_t>("Output", output_shape, output);
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
#endif
}

TEST(PoolTest, MaxPool_10_Dilation_1d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("dilations", std::vector<int64_t>{3});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1, -3, -2, -4, -6, -5, -4, -2};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 6};
  std::vector<float> expected_vals = {4, 3, 2, 4, -1, -2};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_DefaultDilations) {
  OpTester test("MaxPool");

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<float> x_vals = {0.f, 1.f, 2.f,
                               3.f, 4.f, 5.f,
                               6.f, 7.f, 8.f};

  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<float> expected_vals = {1.f, 2.f,
                                      4.f, 5.f,
                                      7.f, 8.f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_DefaultDilations_int8) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<int8_t> x_vals = {0, 1, 2,
                                3, 4, 5,
                                6, 7, 8};

  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<int8_t> expected_vals = {1, 2,
                                       4, 5,
                                       7, 8};

  test.AddInput<int8_t>("X", x_dims, x_vals);
  test.AddOutput<int8_t>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_DefaultDilations_uint8) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<uint8_t> x_vals = {0, 1, 2,
                                 3, 4, 5,
                                 6, 7, 8};

  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<uint8_t> expected_vals = {1, 2,
                                        4, 5,
                                        7, 8};

  test.AddInput<uint8_t>("X", x_dims, x_vals);
  test.AddOutput<uint8_t>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_10_DilationPadding_1d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("dilations", std::vector<int64_t>{3});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1, -3, -2, -4, -6, -5, -4, -2};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals = {2, 4, 3, 2, 4, -1, -2, -2};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider,
            kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_2d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<float> expected_vals = {10, 12, 10, 14, 16, 14};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_2d_int8) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<int8_t> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<int8_t> expected_vals = {10, 12, 10, 14, 16, 14};

  test.AddInput<int8_t>("X", x_dims, x_vals);
  test.AddOutput<int8_t>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_10_DilationPadding_2d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 4, 5};
  std::vector<float> expected_vals = {
      7, 6, 8, 6, 8,
      11, 10, 12, 10, 12,
      15, 14, 16, 14, 16,
      11, 10, 12, 10, 12};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_Ceil0_2d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 1, 3};
  std::vector<float> expected_vals = {10, 12, 10};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_12_Dilation_Ceil0_2d_int8) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  std::vector<int8_t> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 1, 3};
  std::vector<int8_t> expected_vals = {10, 12, 10};

  test.AddInput<int8_t>("X", x_dims, x_vals);
  test.AddOutput<int8_t>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_Ceil1_2d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<float> expected_vals = {10, 12, 10, 10, 12, 10};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_10_DilationPadding_3d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2, 2});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4,
      1, 3, 2, 4, -1,
      5, 7, 6, 8, -2,
      9, 11, 10, 12, -3,
      13, 15, 14, 16, -4};
  std::vector<int64_t> x_dims = {1, 1, 2, 4, 5};
  std::vector<int64_t> expected_dims = {1, 1, 2, 4, 5};
  std::vector<float> expected_vals = {
      7, 6, 8, 6, 8,
      11, 10, 12, 10, 12,
      15, 14, 16, 14, 16,
      11, 10, 12, 10, 12,
      7, 6, 8, 6, 8,
      11, 10, 12, 10, 12,
      15, 14, 16, 14, 16,
      11, 10, 12, 10, 12};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider});
}

TYPED_TEST(PoolTest, GlobalMaxPool) {
  OpTester test("GlobalMaxPool");

  std::vector<float> x_vals = {0.19151945412158966, 0.6221087574958801, 0.43772774934768677,
                               0.7853586077690125, 0.7799758315086365, 0.27259260416030884,
                               0.2764642536640167, 0.801872193813324, 0.9581393599510193,
                               0.8759326338768005, 0.35781726241111755, 0.5009950995445251,
                               0.683462917804718, 0.7127020359039307, 0.37025076150894165,
                               0.5611962080001831, 0.5030831694602966, 0.013768449425697327,
                               0.772826611995697, 0.8826411962509155, 0.36488598585128784,
                               0.6153962016105652, 0.07538124173879623, 0.3688240051269531,
                               0.9331400990486145, 0.6513781547546387, 0.39720258116722107,
                               0.7887301445007324, 0.3168361186981201, 0.5680986642837524,
                               0.8691273927688599, 0.4361734092235565, 0.802147626876831,
                               0.14376682043075562, 0.7042609453201294, 0.7045813202857971,
                               0.2187921106815338, 0.9248676300048828, 0.44214075803756714,
                               0.9093159437179565, 0.05980922281742096, 0.18428708612918854,
                               0.047355279326438904, 0.6748809218406677, 0.5946247577667236,
                               0.5333101749420166, 0.043324064463377, 0.5614330768585205,
                               0.32966843247413635, 0.5029668211936951, 0.11189431697130203,
                               0.6071937084197998, 0.5659446716308594, 0.006764062214642763,
                               0.617441713809967, 0.912122905254364, 0.7905241250991821,
                               0.9920814633369446, 0.9588017463684082, 0.7919641137123108,
                               0.2852509617805481, 0.6249167323112488, 0.47809380292892456,
                               0.19567517936229706, 0.382317453622818, 0.053873684257268906,
                               0.45164841413497925, 0.9820047616958618, 0.12394270300865173,
                               0.1193808987736702, 0.7385230660438538, 0.587303638458252,
                               0.47163254022598267, 0.10712681710720062, 0.22921857237815857,
                               0.8999651670455933, 0.41675353050231934, 0.5358516573905945,
                               0.0062085166573524475, 0.3006417155265808, 0.43689316511154175,
                               0.6121490001678467, 0.9181980490684509, 0.625736653804779,
                               0.7059975862503052, 0.14983370900154114, 0.7460634112358093,
                               0.8310070037841797, 0.6337257623672485, 0.4383098781108856,
                               0.15257278084754944, 0.5684096217155457, 0.5282242894172668,
                               0.9514287710189819, 0.48035916686058044, 0.5025595426559448,
                               0.5368781685829163, 0.8192020654678345, 0.05711563676595688,
                               0.6694217324256897, 0.7671166062355042, 0.7081153392791748,
                               0.7968671917915344, 0.5577608346939087, 0.9658365249633789,
                               0.14715689420700073, 0.02964700013399124, 0.5938934683799744,
                               0.11406569927930832, 0.9508098363876343, 0.32570740580558777,
                               0.19361868500709534, 0.4578116536140442, 0.9204025864601135,
                               0.8790691494941711, 0.252615749835968, 0.34800878167152405,
                               0.18258872628211975, 0.9017960429191589, 0.7065281867980957,
                               0.7266584634780884, 0.900087833404541, 0.7791637778282166,
                               0.5991547703742981, 0.29112523794174194, 0.1513952612876892,
                               0.33517464995384216, 0.6575517654418945, 0.07334254682064056,
                               0.055006396025419235, 0.32319480180740356, 0.5904818177223206,
                               0.8538985848426819, 0.2870624363422394, 0.17306722700595856,
                               0.13402120769023895, 0.9946538209915161, 0.1794978678226471,
                               0.3175468146800995, 0.568291425704956, 0.009348574094474316,
                               0.9006485939025879, 0.9772414565086365, 0.5568946599960327,
                               0.08477384597063065, 0.3330024778842926, 0.7284286618232727,
                               0.14243537187576294, 0.5524689555168152, 0.2730432450771332,
                               0.9744951128959656, 0.6677868962287903, 0.2556532919406891,
                               0.1083114966750145, 0.7761807441711426, 0.7824779748916626,
                               0.7616038918495178, 0.9144031405448914, 0.6586228013038635,
                               0.568367600440979, 0.20175568759441376, 0.6982963681221008,
                               0.952195405960083, 0.8899632692337036, 0.9935673475265503,
                               0.8187035322189331, 0.5451221466064453, 0.45125406980514526,
                               0.8905571699142456, 0.9732648134231567, 0.5934113264083862,
                               0.36607450246810913, 0.3230946958065033, 0.8714232444763184,
                               0.2156340628862381, 0.7349451780319214, 0.36561909317970276,
                               0.8016026020050049, 0.7827355861663818, 0.7013553977012634,
                               0.6227765679359436, 0.4936826527118683, 0.8405377268791199,
                               0.7120969891548157, 0.4439089894294739, 0.031034860759973526,
                               0.36323976516723633, 0.7307217717170715, 0.475566565990448,
                               0.3444169759750366, 0.6408804059028625, 0.12620532512664795};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.9920814633369446, 0.9820047616958618, 0.9946538209915161};

  if constexpr (std::is_same<TypeParam, float>::value) {
    test.AddInput<float>("X", x_dims, x_vals);
    test.AddOutput<float>("Y", expected_dims, expected_vals);
  } else {
    std::vector<TypeParam> x_vals_fp16(x_vals.size());
    std::vector<TypeParam> expected_vals_fp16(expected_vals.size());

    ConvertFloatToMLFloat16(x_vals.data(), x_vals_fp16.data(), x_vals.size());
    ConvertFloatToMLFloat16(expected_vals.data(), expected_vals_fp16.data(), expected_vals.size());
    test.AddInput<TypeParam>("X", x_dims, x_vals_fp16);
    test.AddOutput<TypeParam>("Y", expected_dims, expected_vals_fp16);
  }

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TYPED_TEST(PoolTest, GlobalMaxPool3D) {
  OpTester test("GlobalMaxPool");

  std::vector<float> x_vals = {0.19151945412158966, 0.6221087574958801, 0.43772774934768677,
                               0.7853586077690125, 0.7799758315086365, 0.27259260416030884,
                               0.2764642536640167, 0.801872193813324, 0.9581393599510193,
                               0.8759326338768005, 0.35781726241111755, 0.5009950995445251,
                               0.683462917804718, 0.7127020359039307, 0.37025076150894165,
                               0.5611962080001831, 0.5030831694602966, 0.013768449425697327,
                               0.772826611995697, 0.8826411962509155, 0.36488598585128784,
                               0.6153962016105652, 0.07538124173879623, 0.3688240051269531,
                               0.9331400990486145, 0.6513781547546387, 0.39720258116722107,
                               0.7887301445007324, 0.3168361186981201, 0.5680986642837524,
                               0.8691273927688599, 0.4361734092235565, 0.802147626876831,
                               0.14376682043075562, 0.7042609453201294, 0.7045813202857971,
                               0.2187921106815338, 0.9248676300048828, 0.44214075803756714,
                               0.9093159437179565, 0.05980922281742096, 0.18428708612918854,
                               0.047355279326438904, 0.6748809218406677, 0.5946247577667236,
                               0.5333101749420166, 0.043324064463377, 0.5614330768585205,
                               0.32966843247413635, 0.5029668211936951, 0.11189431697130203,
                               0.6071937084197998, 0.5659446716308594, 0.006764062214642763,
                               0.617441713809967, 0.912122905254364, 0.7905241250991821,
                               0.9920814633369446, 0.9588017463684082, 0.7919641137123108,
                               0.2852509617805481, 0.6249167323112488, 0.47809380292892456,
                               0.19567517936229706, 0.382317453622818, 0.053873684257268906,
                               0.45164841413497925, 0.9820047616958618, 0.12394270300865173,
                               0.1193808987736702, 0.7385230660438538, 0.587303638458252,
                               0.47163254022598267, 0.10712681710720062, 0.22921857237815857,
                               0.8999651670455933, 0.41675353050231934, 0.5358516573905945,
                               0.0062085166573524475, 0.3006417155265808, 0.43689316511154175,
                               0.6121490001678467, 0.9181980490684509, 0.625736653804779,
                               0.7059975862503052, 0.14983370900154114, 0.7460634112358093,
                               0.8310070037841797, 0.6337257623672485, 0.4383098781108856,
                               0.15257278084754944, 0.5684096217155457, 0.5282242894172668,
                               0.9514287710189819, 0.48035916686058044, 0.5025595426559448,
                               0.5368781685829163, 0.8192020654678345, 0.05711563676595688,
                               0.6694217324256897, 0.7671166062355042, 0.7081153392791748,
                               0.7968671917915344, 0.5577608346939087, 0.9658365249633789,
                               0.14715689420700073, 0.02964700013399124, 0.5938934683799744,
                               0.11406569927930832, 0.9508098363876343, 0.32570740580558777,
                               0.19361868500709534, 0.4578116536140442, 0.9204025864601135,
                               0.8790691494941711, 0.252615749835968, 0.34800878167152405,
                               0.18258872628211975, 0.9017960429191589, 0.7065281867980957,
                               0.7266584634780884, 0.900087833404541, 0.7791637778282166,
                               0.5991547703742981, 0.29112523794174194, 0.1513952612876892,
                               0.33517464995384216, 0.6575517654418945, 0.07334254682064056,
                               0.055006396025419235, 0.32319480180740356, 0.5904818177223206,
                               0.8538985848426819, 0.2870624363422394, 0.17306722700595856,
                               0.13402120769023895, 0.9946538209915161, 0.1794978678226471,
                               0.3175468146800995, 0.568291425704956, 0.009348574094474316,
                               0.9006485939025879, 0.9772414565086365, 0.5568946599960327,
                               0.08477384597063065, 0.3330024778842926, 0.7284286618232727,
                               0.14243537187576294, 0.5524689555168152, 0.2730432450771332,
                               0.9744951128959656, 0.6677868962287903, 0.2556532919406891,
                               0.1083114966750145, 0.7761807441711426, 0.7824779748916626,
                               0.7616038918495178, 0.9144031405448914, 0.6586228013038635,
                               0.568367600440979, 0.20175568759441376, 0.6982963681221008,
                               0.952195405960083, 0.8899632692337036, 0.9935673475265503,
                               0.8187035322189331, 0.5451221466064453, 0.45125406980514526,
                               0.8905571699142456, 0.9732648134231567, 0.5934113264083862,
                               0.36607450246810913, 0.3230946958065033, 0.8714232444763184,
                               0.2156340628862381, 0.7349451780319214, 0.36561909317970276,
                               0.8016026020050049, 0.7827355861663818, 0.7013553977012634,
                               0.6227765679359436, 0.4936826527118683, 0.8405377268791199,
                               0.7120969891548157, 0.4439089894294739, 0.031034860759973526,
                               0.36323976516723633, 0.7307217717170715, 0.475566565990448,
                               0.3444169759750366, 0.6408804059028625, 0.12620532512664795};
  std::vector<int64_t> x_dims = {1, 3, 8, 4, 2};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1, 1};
  std::vector<float> expected_vals = {0.9920814633369446, 0.9820047616958618, 0.9946538209915161};

  if constexpr (std::is_same<TypeParam, float>::value) {
    test.AddInput<float>("X", x_dims, x_vals);
    test.AddOutput<float>("Y", expected_dims, expected_vals);
  } else {
    std::vector<TypeParam> x_vals_fp16(x_vals.size());
    std::vector<TypeParam> expected_vals_fp16(expected_vals.size());

    ConvertFloatToMLFloat16(x_vals.data(), x_vals_fp16.data(), x_vals.size());
    ConvertFloatToMLFloat16(expected_vals.data(), expected_vals_fp16.data(), expected_vals.size());
    test.AddInput<TypeParam>("X", x_dims, x_vals_fp16);
    test.AddOutput<TypeParam>("Y", expected_dims, expected_vals_fp16);
  }

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, AveragePool) {
  OpTester test("AveragePool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{8, 8});

  std::vector<float> x_vals = {0.3337382376194, 0.8794041872024536, 0.33745908737182617,
                               0.666634202003479, 0.44255536794662476, 0.6473854184150696,
                               0.7674617171287537, 0.8822641968727112, 0.8852233290672302,
                               0.7453723549842834, 0.2818361520767212, 0.8706393241882324,
                               0.5406734347343445, 0.02016347087919712, 0.8047968745231628,
                               0.6037390828132629, 0.6242085099220276, 0.15702469646930695,
                               0.19581079483032227, 0.7122684717178345, 0.06907976418733597,
                               0.5234333872795105, 0.9091887474060059, 0.4319673180580139,
                               0.8100792169570923, 0.10371053218841553, 0.3888828456401825,
                               0.27514228224754333, 0.2670423686504364, 0.04306316748261452,
                               0.36913928389549255, 0.8686641454696655, 0.6307396292686462,
                               0.7112566232681274, 0.25298961997032166, 0.5131869316101074,
                               0.11016560345888138, 0.20159587264060974, 0.43771353363990784,
                               0.7566956877708435, 0.10168474912643433, 0.7238786220550537,
                               0.4961036741733551, 0.9173188209533691, 0.6056748032569885,
                               0.250592976808548, 0.4755987823009491, 0.904503583908081,
                               0.4725301265716553, 0.8506938219070435, 0.13940207660198212,
                               0.9848986864089966, 0.6715511083602905, 0.8943559527397156,
                               0.40052708983421326, 0.0880642905831337, 0.8935731649398804,
                               0.3453705310821533, 0.8090538382530212, 0.19269756972789764,
                               0.03951506316661835, 0.027226323261857033, 0.8117656111717224,
                               0.7711597084999084, 0.8593372702598572, 0.20363913476467133,
                               0.7842649817466736, 0.29195329546928406, 0.5064213871955872,
                               0.7418627142906189, 0.1069103255867958, 0.5893736481666565,
                               0.2143796980381012, 0.15637169778347015, 0.1684667021036148,
                               0.7528857588768005, 0.5846885442733765, 0.9133154153823853,
                               0.6781020760536194, 0.21141840517520905, 0.05769576504826546,
                               0.49993178248405457, 0.2309824675321579, 0.05175522714853287,
                               0.6969341039657593, 0.47234174609184265, 0.11310867220163345,
                               0.6184650659561157, 0.896835207939148, 0.6077945232391357,
                               0.3074592649936676, 0.07904505729675293, 0.048881493508815765,
                               0.24833321571350098, 0.9844338893890381, 0.4520559012889862,
                               0.26799046993255615, 0.7592704892158508, 0.37819114327430725,
                               0.30964234471321106, 0.8839467167854309, 0.0934458002448082,
                               0.379569411277771, 0.09841523319482803, 0.6000676155090332,
                               0.7950544357299805, 0.45938217639923096, 0.5537487864494324,
                               0.38861554861068726, 0.4074040949344635, 0.38612639904022217,
                               0.89164137840271, 0.21732182800769806, 0.6711451411247253,
                               0.5769082307815552, 0.9865275621414185, 0.03840707615017891,
                               0.1573856621980667, 0.09340689331293106, 0.9288106560707092,
                               0.16059239208698273, 0.8247162103652954, 0.422741562128067,
                               0.987165629863739, 0.9476590752601624, 0.9242128133773804,
                               0.9987634420394897, 0.32882997393608093, 0.011870949529111385,
                               0.984099805355072, 0.09365611523389816, 0.33463314175605774,
                               0.6386845111846924, 0.9860017895698547, 0.4672822654247284,
                               0.9529699683189392, 0.15891511738300323, 0.7175184488296509,
                               0.024524977430701256, 0.8217390179634094, 0.14411452412605286,
                               0.45218998193740845, 0.4429023861885071, 0.9931989312171936,
                               0.8507111072540283, 0.13051295280456543, 0.07811085134744644,
                               0.943297803401947, 0.030969098210334778, 0.21649038791656494,
                               0.9491124749183655, 0.5731316804885864, 0.5927708745002747,
                               0.7653813362121582, 0.5627018809318542, 0.01101104449480772,
                               0.7299126982688904, 0.3900069296360016, 0.0853394865989685,
                               0.43255582451820374, 0.8431127071380615, 0.5303983092308044,
                               0.6451488137245178, 0.16481569409370422, 0.35921016335487366,
                               0.036783039569854736, 0.5699883103370667, 0.5847001075744629,
                               0.9650961756706238, 0.9053892493247986, 0.2933308482170105,
                               0.2615077495574951, 0.48302537202835083, 0.5642899870872498,
                               0.20961439609527588, 0.37418732047080994, 0.4921484887599945,
                               0.7827269434928894, 0.28503814339637756, 0.4663805067539215,
                               0.1988927721977234, 0.20202897489070892, 0.3183555603027344,
                               0.4528728425502777, 0.2815922796726227, 0.820142388343811,
                               0.4963360130786896, 0.46687841415405273, 0.7405545115470886,
                               0.40191709995269775, 0.21238186955451965, 0.46927347779273987};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.5146896243095398, 0.4851023256778717, 0.4756942689418793};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, AveragePool_IncludePadPixel) {
  OpTester test("AveragePool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("count_include_pad", (int64_t)1);
  std::vector<float> x_vals = {0.3337f, 0.8794f, 0.3375f,
                               0.6666f, 0.4426f, 0.6474f,
                               0.7675f, 0.8823f, 0.8852f};

  std::vector<int64_t> x_dims = {1, 1, 3, 3};
  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals = {0.0834f, 0.3033f, 0.3042f, 0.0844f,
                                      0.2501f, 0.5806f, 0.5767f, 0.2462f,
                                      0.3585f, 0.6897f, 0.7144f, 0.3832f,
                                      0.1919f, 0.4124f, 0.4419f, 0.2213f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.SetOutputTolerance(0.0001f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Regression test for https://github.com/microsoft/onnxruntime/issues/26708
// AveragePool with count_include_pad=1 and asymmetric pads (only bottom/right)
// was using incorrect pad index for hend, producing wrong results.
TEST(PoolTest, AveragePool_CountIncludePad_AsymmetricPads) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 1, 1});  // no top/left, 1 bottom, 1 right
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("count_include_pad", (int64_t)1);

  // Input: 2x2 all ones
  std::vector<float> x_vals = {1.0f, 1.0f,
                               1.0f, 1.0f};
  std::vector<int64_t> x_dims = {1, 1, 2, 2};

  // Output: 2x2
  // Top-left:     (1+1+1+1)/4 = 1.0
  // Top-right:    (1+0+1+0)/4 = 0.5
  // Bottom-left:  (1+1+0+0)/4 = 0.5
  // Bottom-right: (1+0+0+0)/4 = 0.25
  std::vector<int64_t> expected_dims = {1, 1, 2, 2};
  std::vector<float> expected_vals = {1.0f, 0.5f,
                                      0.5f, 0.25f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // The CUDA custom AveragePoolWithPad kernel now honors per-side (asymmetric) pads, so the
  // CUDA (NCHW) leg is un-excluded here to lock in that fix. kCudaNHWCExecutionProvider is
  // excluded here only to avoid redundant coverage: the asymmetric NHWC-CUDA decode branch is
  // now exercised (and passing) by the 1D/2D AveragePool_CUDA_* parity tests below. The remaining
  // exclusions are EPs whose external libraries (CoreML, etc.) still produce wrong results here.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaNHWCExecutionProvider,
            kTensorrtExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider,
            kDnnlExecutionProvider, kCoreMLExecutionProvider, kQnnExecutionProvider,
            kDmlExecutionProvider});
}

// AveragePool3D with count_include_pad=1 and asymmetric pads (only back/bottom)
// Regression test for 3D path of the pad-index bug
TEST(PoolTest, AveragePool3D_CountIncludePad_AsymmetricPads) {
  OpTester test("AveragePool", 19);
  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 1, 1, 0});  // no front/top/left, 1 back, 1 bottom
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("count_include_pad", (int64_t)1);
  // Input: 2x2x2 all ones (N=1, C=1, D=2, H=2, W=2)
  std::vector<float> x3d_vals = {
      1.0f, 1.0f,
      1.0f, 1.0f,
      1.0f, 1.0f,
      1.0f, 1.0f};
  std::vector<int64_t> x3d_dims = {1, 1, 2, 2, 2};
  // Output: 2x2x1 (D x H x W)
  // D=0,H=0,W=0: (8 ones)/8 = 1.0
  // D=1,H=0,W=0: (4 ones + 4 padded zeros)/8 = 0.5
  // D=0,H=1,W=0: (4 ones + 4 padded zeros)/8 = 0.5
  // D=1,H=1,W=0: (2 ones + 6 padded zeros)/8 = 0.25
  std::vector<int64_t> expected3d_dims = {1, 1, 2, 2, 1};
  std::vector<float> expected3d_vals = {1.0f, 0.5f,
                                        0.5f, 0.25f};
  test.AddInput<float>("X", x3d_dims, x3d_vals);
  test.AddOutput<float>("Y", expected3d_dims, expected3d_vals);
  // The CUDA custom AveragePoolWithPad kernel now honors per-side (asymmetric) pads in 3D, so the
  // CUDA (NCHW) leg is un-excluded here to lock in that fix. kCudaNHWCExecutionProvider stays
  // excluded here: the asymmetric NHWC-CUDA decode branch is now exercised (and passing) by the
  // 1D/2D AveragePool_CUDA_* parity tests below, but 3D (NDHWC) NHWC pooling is not among them, so
  // it remains a follow-up. The remaining exclusions are EPs whose external libraries (CoreML,
  // etc.) still produce wrong results here.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaNHWCExecutionProvider,
            kTensorrtExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider,
            kDnnlExecutionProvider, kCoreMLExecutionProvider, kQnnExecutionProvider,
            kDmlExecutionProvider});
}

// test 'strides' attribute not specified
TEST(PoolTest, AveragePool_DefaultStrides) {
  OpTester test("AveragePool");
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});
  std::vector<float> x_vals = {0.f, 1.f, 2.f,
                               3.f, 4.f, 5.f,
                               6.f, 7.f, 8.f};

  std::vector<int64_t> x_dims = {1, 3, 3};
  std::vector<int64_t> expected_dims = {1, 3, 2};
  std::vector<float> expected_vals = {0.5f, 1.5f,
                                      3.5f, 4.5f,
                                      6.5f, 7.5f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, AveragePool_10_ceil1_2d) {
  OpTester test("AveragePool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<float> x_vals = {
      1, 3, 2, 4,
      5, 7, 6, 8,
      9, 11, 10, 12,
      13, 15, 14, 16};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 2, 3};
  std::vector<float> expected_vals = {4.0f, 4.5f, 5.0f, 14.0f, 14.5f, 15.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, AveragePool_19_dilation_2d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<float> x_vals = {
      1, 3, 2, 4,
      5, 7, 6, 8,
      9, 11, 10, 12,
      13, 15, 14, 16};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 2, 2};
  std::vector<float> expected_vals = {5.5f, 7.5f, 9.5f, 11.5f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider,
            kTensorrtExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(PoolTest, AveragePool_19_ceil_count_include_pad_1d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {2.0903f, 4.6493f, 1.6320f, -3.2051f, 4.6975f, 4.7296f, 3.3653f, -1.5815f, -2.3832f, 0.9628f, -1.5899f, -2.6820f, 5.7529f, 7.7346f, -0.8910f, -2.0151f, 0.1313f, -0.5374f};
  std::vector<int64_t> x_dims = {1, 2, 9};
  std::vector<int64_t> expected_dims = {1, 2, 4};
  std::vector<float> expected_vals = {0.73807144f, 2.5655572f, 0.8032287f, -0.09990001f, 0.34911433f, 1.0389f, 1.4536142f, -0.40353334f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // TODO: Re-enable DML when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

// ---------------------------------------------------------------------------
// CUDA AveragePool asymmetric-padding parity tests.
//
// cuDNN's pooling descriptor stores one symmetric pad per axis, so it silently drops the
// ONNX end pad when pad_begin != pad_end, producing wrong averages on CUDA (e.g. for a 1D
// pad of (0,3) cuDNN yields [4, 6.5, 8] while the CPU reference yields [4, 5.571, 4], and a
// 2D pad of (0,0,3,3) diverges by up to 53.25). The custom AveragePoolWithPad CUDA kernel
// fixes this. These cases keep the CUDA EP UN-excluded so the CUDA leg actually runs and must
// match the CPU reference oracle. Expected values are the CPU reference outputs.
//
// ceil_mode + count_include_pad cases use opset 19 so the CPU leg runs the already-correct v19
// reference functor and validates the CUDA kernel independently of the separate CPU opset-7..18
// MLAS fix (PR #29629); the CUDA routing is opset-independent, so this still exercises the fix.
//
// Exception: the fp16 case below runs CUDA-only (CPU has no fp16 AveragePool kernel on x64 and
// the Arm64 NEON fp16 pooling kernel mishandles the ceil_mode + count_include_pad divisor), so
// the "CPU leg runs the v19 reference oracle" statement above does not apply to it — see its own
// comment for how it validates the CUDA half accumulate-in-float path without a CPU oracle.
//
// The float cases share a single exclusion set (kPoolingEpsExcludedFromCeilCipTests) so the list cannot
// drift test-to-test. It names every EP whose pooling does NOT implement ONNX's asymmetric-pad /
// dilated / ceil_mode + count_include_pad clamped-divisor semantics (they would produce wrong
// values and cannot serve as an oracle). The CPU EP (correct v19 reference) stays un-excluded as
// the float oracle, and the CUDA + CUDA-NHWC EPs stay un-excluded as the tested targets — the
// asymmetric NHWC-CUDA path is intentionally exercised here (USE_CUDA_NHWC_OPS defaults ON) and
// passes. kWebGpuExecutionProvider is listed defensively: it auto-skips in a CUDA-only build
// (DefaultWebGpuExecutionProvider returns nullptr), but naming it keeps a future WebGPU build leg
// from re-triggering the CI failure this list fixes.
//
// NOTE: do not confuse this with kPoolingEpsExcludedFromCeilCountIncludePadTests (top of file,
// ~L21), which has the OPPOSITE kCuda membership. That set is a CPU-reference GATE and INCLUDES
// kCuda/kCudaNHWC in its exclusions; this set EXCLUDES them because here CUDA/CUDA-NHWC ARE the
// tested targets. Opposite kCuda intent — pick the one matching your test.
// ---------------------------------------------------------------------------
const std::unordered_set<std::string> kPoolingEpsExcludedFromCeilCipTests = {
    kTensorrtExecutionProvider, kNvTensorRTRTXExecutionProvider, kDnnlExecutionProvider,
    kOpenVINOExecutionProvider, kAclExecutionProvider, kCoreMLExecutionProvider,
    kQnnExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider};

TEST(PoolTest, AveragePool_CUDA_asymmetric_tail_pad_1d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3};
  std::vector<float> expected_vals = {4.0f, 5.5714283f, 4.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

TEST(PoolTest, AveragePool_CUDA_asymmetric_tail_pad_1d_exclude_pad) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)0);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3};
  // exclude-pad divides by in-bounds cells only.
  std::vector<float> expected_vals = {4.0f, 6.5f, 8.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

TEST(PoolTest, AveragePool_CUDA_asymmetric_tail_pad_2d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3, 3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7, 7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals(81);
  for (int i = 0; i < 81; ++i) {
    x_vals[i] = static_cast<float>(i + 1);
  }
  std::vector<int64_t> x_dims = {1, 1, 9, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3, 3};
  std::vector<float> expected_vals = {31.0f, 28.714287f, 17.5f,
                                      45.857143f, 41.142857f, 24.642858f,
                                      33.5f, 29.785715f, 17.75f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

TEST(PoolTest, AveragePool_CUDA_asymmetric_tail_pad_2d_exclude_pad) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3, 3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7, 7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)0);

  std::vector<float> x_vals(81);
  for (int i = 0; i < 81; ++i) {
    x_vals[i] = static_cast<float>(i + 1);
  }
  std::vector<int64_t> x_dims = {1, 1, 9, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3, 3};
  std::vector<float> expected_vals = {31.0f, 33.5f, 35.0f,
                                      53.5f, 56.0f, 57.5f,
                                      67.0f, 69.5f, 71.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// auto_pad=SAME_UPPER produces naturally asymmetric pads (here pad(0,1)); proves the latent
// SAME-pad bug on CUDA is also fixed by the same kernel.
TEST(PoolTest, AveragePool_CUDA_same_upper_asymmetric_1d) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "SAME_UPPER");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  std::vector<int64_t> x_dims = {1, 1, 10};
  std::vector<int64_t> expected_dims = {1, 1, 5};
  std::vector<float> expected_vals = {2.0f, 4.0f, 6.0f, 8.0f, 6.3333335f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// Regression guard: symmetric pads must STAY on the fast cuDNN path and remain correct.
TEST(PoolTest, AveragePool_CUDA_symmetric_pad_regression_1d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 4};
  std::vector<float> expected_vals = {1.4285715f, 4.0f, 5.5714283f, 4.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// MaxPool asymmetric-pad probe/regression on CUDA (verify-then-decide per design). MaxPool
// ignores pad cells (no divisor); asymmetric tail pad only changes output size, computed
// correctly upstream. CUDA un-excluded to confirm parity with the CPU reference.
TEST(PoolTest, MaxPool_CUDA_asymmetric_tail_pad_1d) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3};
  std::vector<float> expected_vals = {7.0f, 9.0f, 9.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// fp16 asymmetric-pad AveragePool. Runs on the CUDA EP ONLY (via an explicit provider list):
// the CPU AveragePool has no fp16 kernel on x64, and the Arm64 NEON fp16 pooling kernel does
// not honor the ceil_mode + count_include_pad divisor rule (a separate, pre-existing CPU
// limitation), so it cannot serve as the fp16 oracle. This test validates that the CUDA
// AveragePoolWithPad kernel's half accumulate-in-float path matches the reference values.
TEST(PoolTest, AveragePool_CUDA_asymmetric_tail_pad_1d_fp16) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<MLFloat16> x_vals = {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
                                   MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f),
                                   MLFloat16(7.0f), MLFloat16(8.0f), MLFloat16(9.0f)};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 3};
  std::vector<MLFloat16> expected_vals = {MLFloat16(4.0f), MLFloat16(5.5714283f), MLFloat16(4.0f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.SetOutputTolerance(0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Symmetric pads BUT dilation > 1. cuDNN's pooling descriptor has no dilation parameter, so the
// old (asymmetric-pads-only) guard let this fall through to cuDNN, which silently ignored the
// dilation and produced the wrong result. The dilation guard (!default_dilations) now routes this
// to the custom kernel. opset 19 so the CPU AveragePoolV19 reference (which honors dilation) also
// runs and must match. Expected values come from that CPU reference.
TEST(PoolTest, AveragePool_CUDA_symmetric_pad_dilation_1d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("dilations", std::vector<int64_t>{2});
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> x_dims = {1, 1, 9};
  std::vector<int64_t> expected_dims = {1, 1, 9};
  std::vector<float> expected_vals = {1.3333334f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                      4.6666665f, 5.3333335f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// auto_pad=SAME_LOWER produces naturally asymmetric pads with the extra pad on the LOW side
// (here pad(1,0)); companion to the SAME_UPPER case. opset 19 so the CPU reference also runs.
TEST(PoolTest, AveragePool_CUDA_same_lower_asymmetric_1d) {
  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "SAME_LOWER");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  std::vector<int64_t> x_dims = {1, 1, 10};
  std::vector<int64_t> expected_dims = {1, 1, 5};
  std::vector<float> expected_vals = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCipTests);
}

// bf16 AveragePool is intentionally NOT tested here: although the CUDA kernel instantiates
// BFloat16 (compile-checked) and AveragePoolWithPad accumulates it in float like fp16, the ONNX
// AveragePool schema type constraint does not include tensor(bfloat16), so OpTester's model
// type-checker rejects such a graph at load. The fp16 case above already exercises the
// accumulate-in-float path.

// (a)-gate regression test for the CPU/MLAS AvgPool ceil_mode + count_include_pad bug
// (PyTorch #183528). This is the opset-18 clone of AveragePool_19_ceil_count_include_pad_1d:
// same X and same expected_vals, but at opset 18 the float path routes through MLAS (which
// divided by the full kernel size and produced a wrong average) instead of the v19 reference
// loop. GPU / other EPs are excluded so the test is green the moment the CPU fix lands; the
// CUDA leg is tracked separately as the (b) probe.
TEST(PoolTest, AveragePool_18_ceil_count_include_pad_1d) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3});
  test.AddAttribute("pads", std::vector<int64_t>{3, 3});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{7});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {2.0903f, 4.6493f, 1.6320f, -3.2051f, 4.6975f, 4.7296f, 3.3653f, -1.5815f, -2.3832f, 0.9628f, -1.5899f, -2.6820f, 5.7529f, 7.7346f, -0.8910f, -2.0151f, 0.1313f, -0.5374f};
  std::vector<int64_t> x_dims = {1, 2, 9};
  std::vector<int64_t> expected_dims = {1, 2, 4};
  std::vector<float> expected_vals = {0.73807144f, 2.5655572f, 0.8032287f, -0.09990001f, 0.34911433f, 1.0389f, 1.4536142f, -0.40353334f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCountIncludePadTests);
}

// 2D opset-18 case for the same bug. Input is the PyTorch #183528 repro:
// x = arange(1, 17).reshape(1, 1, 4, 4), kernel=3, stride=2, pad=1, ceil_mode=1,
// count_include_pad=1. The ceil-mode trailing window ends past input+pad_tail, so MLAS's
// full-kernel divisor gave a wrong average; the reference loop divides by the clamped
// window (in-bounds + real pad cells only).
TEST(PoolTest, AveragePool_18_ceil_count_include_pad_2d) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 3, 3};
  std::vector<float> expected_vals = {1.5555556f, 3.3333333f, 2.0f,
                                      6.3333335f, 11.0f, 6.0f,
                                      4.5f, 7.5f, 4.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCountIncludePadTests);
}

// 3D opset-18 case for the same bug, exercising the AveragePool3DTask path.
TEST(PoolTest, AveragePool_18_ceil_count_include_pad_3d) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)1);

  std::vector<float> x_vals(27);
  for (int i = 0; i < 27; ++i) {
    x_vals[i] = static_cast<float>(i + 1);
  }
  std::vector<int64_t> x_dims = {1, 1, 3, 3, 3};
  std::vector<int64_t> expected_dims = {1, 1, 2, 2, 2};
  // Ground truth from the CPU v19 reference loop (window clamped to input + real pad).
  std::vector<float> expected_vals = {2.2222223f, 2.5185184f, 3.1111112f, 3.4074075f,
                                      4.888889f, 5.185185f, 5.7777777f, 6.074074f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCountIncludePadTests);
}

// No-regression guard: with count_include_pad=0 the divisor already counts only in-bounds
// cells, so this combo stays on the MLAS fast path and must remain correct.
TEST(PoolTest, AveragePool_18_ceil_count_exclude_pad_2d) {
  OpTester test("AveragePool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("ceil_mode", (int64_t)1);
  test.AddAttribute("count_include_pad", (int64_t)0);

  std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 3, 3};
  // count_include_pad=0: each output divides by the number of in-bounds cells only.
  std::vector<float> expected_vals = {3.5f, 5.0f, 6.0f,
                                      9.5f, 11.0f, 12.0f,
                                      13.5f, 15.0f, 16.0f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kPoolingEpsExcludedFromCeilCountIncludePadTests);
}

TEST(PoolTest, GlobalAveragePool) {
  OpTester test("GlobalAveragePool");

  std::vector<float> x_vals = {0.3337382376194, 0.8794041872024536, 0.33745908737182617,
                               0.666634202003479, 0.44255536794662476, 0.6473854184150696,
                               0.7674617171287537, 0.8822641968727112, 0.8852233290672302,
                               0.7453723549842834, 0.2818361520767212, 0.8706393241882324,
                               0.5406734347343445, 0.02016347087919712, 0.8047968745231628,
                               0.6037390828132629, 0.6242085099220276, 0.15702469646930695,
                               0.19581079483032227, 0.7122684717178345, 0.06907976418733597,
                               0.5234333872795105, 0.9091887474060059, 0.4319673180580139,
                               0.8100792169570923, 0.10371053218841553, 0.3888828456401825,
                               0.27514228224754333, 0.2670423686504364, 0.04306316748261452,
                               0.36913928389549255, 0.8686641454696655, 0.6307396292686462,
                               0.7112566232681274, 0.25298961997032166, 0.5131869316101074,
                               0.11016560345888138, 0.20159587264060974, 0.43771353363990784,
                               0.7566956877708435, 0.10168474912643433, 0.7238786220550537,
                               0.4961036741733551, 0.9173188209533691, 0.6056748032569885,
                               0.250592976808548, 0.4755987823009491, 0.904503583908081,
                               0.4725301265716553, 0.8506938219070435, 0.13940207660198212,
                               0.9848986864089966, 0.6715511083602905, 0.8943559527397156,
                               0.40052708983421326, 0.0880642905831337, 0.8935731649398804,
                               0.3453705310821533, 0.8090538382530212, 0.19269756972789764,
                               0.03951506316661835, 0.027226323261857033, 0.8117656111717224,
                               0.7711597084999084, 0.8593372702598572, 0.20363913476467133,
                               0.7842649817466736, 0.29195329546928406, 0.5064213871955872,
                               0.7418627142906189, 0.1069103255867958, 0.5893736481666565,
                               0.2143796980381012, 0.15637169778347015, 0.1684667021036148,
                               0.7528857588768005, 0.5846885442733765, 0.9133154153823853,
                               0.6781020760536194, 0.21141840517520905, 0.05769576504826546,
                               0.49993178248405457, 0.2309824675321579, 0.05175522714853287,
                               0.6969341039657593, 0.47234174609184265, 0.11310867220163345,
                               0.6184650659561157, 0.896835207939148, 0.6077945232391357,
                               0.3074592649936676, 0.07904505729675293, 0.048881493508815765,
                               0.24833321571350098, 0.9844338893890381, 0.4520559012889862,
                               0.26799046993255615, 0.7592704892158508, 0.37819114327430725,
                               0.30964234471321106, 0.8839467167854309, 0.0934458002448082,
                               0.379569411277771, 0.09841523319482803, 0.6000676155090332,
                               0.7950544357299805, 0.45938217639923096, 0.5537487864494324,
                               0.38861554861068726, 0.4074040949344635, 0.38612639904022217,
                               0.89164137840271, 0.21732182800769806, 0.6711451411247253,
                               0.5769082307815552, 0.9865275621414185, 0.03840707615017891,
                               0.1573856621980667, 0.09340689331293106, 0.9288106560707092,
                               0.16059239208698273, 0.8247162103652954, 0.422741562128067,
                               0.987165629863739, 0.9476590752601624, 0.9242128133773804,
                               0.9987634420394897, 0.32882997393608093, 0.011870949529111385,
                               0.984099805355072, 0.09365611523389816, 0.33463314175605774,
                               0.6386845111846924, 0.9860017895698547, 0.4672822654247284,
                               0.9529699683189392, 0.15891511738300323, 0.7175184488296509,
                               0.024524977430701256, 0.8217390179634094, 0.14411452412605286,
                               0.45218998193740845, 0.4429023861885071, 0.9931989312171936,
                               0.8507111072540283, 0.13051295280456543, 0.07811085134744644,
                               0.943297803401947, 0.030969098210334778, 0.21649038791656494,
                               0.9491124749183655, 0.5731316804885864, 0.5927708745002747,
                               0.7653813362121582, 0.5627018809318542, 0.01101104449480772,
                               0.7299126982688904, 0.3900069296360016, 0.0853394865989685,
                               0.43255582451820374, 0.8431127071380615, 0.5303983092308044,
                               0.6451488137245178, 0.16481569409370422, 0.35921016335487366,
                               0.036783039569854736, 0.5699883103370667, 0.5847001075744629,
                               0.9650961756706238, 0.9053892493247986, 0.2933308482170105,
                               0.2615077495574951, 0.48302537202835083, 0.5642899870872498,
                               0.20961439609527588, 0.37418732047080994, 0.4921484887599945,
                               0.7827269434928894, 0.28503814339637756, 0.4663805067539215,
                               0.1988927721977234, 0.20202897489070892, 0.3183555603027344,
                               0.4528728425502777, 0.2815922796726227, 0.820142388343811,
                               0.4963360130786896, 0.46687841415405273, 0.7405545115470886,
                               0.40191709995269775, 0.21238186955451965, 0.46927347779273987};
  std::vector<int64_t> x_dims = {1, 3, 8, 8};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {0.5146896243095398, 0.4851023256778717, 0.4756942689418793};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TEST(PoolTest, GlobalAveragePool_22_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("GlobalAveragePool", 22);

  std::vector<float> x_vals = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<int64_t> x_dims = {1, 1, 4, 4};
  std::vector<int64_t> expected_dims = {1, 1, 1, 1};
  std::vector<float> expected_vals = {8.5f};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(PoolTest, GlobalAveragePool_Large_128) {
  OpTester test("GlobalAveragePool");

  std::vector<float> x_vals(1 * 1 * 128 * 128, 2.71828f);
  std::vector<int64_t> x_dims = {1, 1, 128, 128};
  std::vector<int64_t> expected_dims = {1, 1, 1, 1};
  std::vector<float> expected_vals = {2.71828f};
  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals,
                        /*sort_output=*/false, /*rel_error=*/1e-3f, /*abs_error=*/1e-2f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TEST(PoolTest, GlobalAveragePool_Large_256) {
  OpTester test("GlobalAveragePool");

  std::vector<float> x_vals(1 * 1 * 256 * 256, 3.14159f);
  std::vector<int64_t> x_dims = {1, 1, 256, 256};
  std::vector<int64_t> expected_dims = {1, 1, 1, 1};
  std::vector<float> expected_vals = {3.14159f};
  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals,
                        /*sort_output=*/false, /*rel_error=*/1e-3f, /*abs_error=*/1e-2f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TEST(PoolTest, LpPool) {
  OpTester test("LpPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

  std::vector<float> x_vals = {0.688458621501922607421875,
                               0.8835647106170654296875,
                               0.782541573047637939453125,
                               0.300049364566802978515625,
                               0.8066387176513671875,
                               0.4520850479602813720703125,
                               0.598959147930145263671875,
                               0.4597113132476806640625,
                               0.8161861896514892578125,
                               0.7667262554168701171875,
                               0.840198040008544921875,
                               0.583297073841094970703125,
                               0.708858668804168701171875,
                               0.4728293716907501220703125,
                               0.4992314875125885009765625,
                               0.2504110038280487060546875,
                               0.3881411850452423095703125,
                               0.517398893833160400390625,
                               0.657192409038543701171875,
                               0.7325098514556884765625,
                               0.10206781327724456787109375,
                               0.2179393768310546875,
                               0.0616043470799922943115234375,
                               0.475992143154144287109375,
                               0.737536609172821044921875,
                               0.9689886569976806640625,
                               0.4474093914031982421875,
                               0.4323260486125946044921875,
                               0.648917853832244873046875,
                               0.701454102993011474609375,
                               0.107639573514461517333984375,
                               0.811198413372039794921875,
                               0.725269258022308349609375,
                               0.7497208118438720703125,
                               0.3340204060077667236328125,
                               0.87611293792724609375,
                               0.6691205501556396484375,
                               0.87189638614654541015625,
                               0.971237838268280029296875,
                               0.11620916426181793212890625,
                               0.0249019563198089599609375,
                               0.752140820026397705078125,
                               0.865541160106658935546875,
                               0.015474068932235240936279296875,
                               0.5126011371612548828125,
                               0.45315420627593994140625,
                               0.1925573647022247314453125,
                               0.98408544063568115234375,
                               0.14754636585712432861328125,
                               0.54990971088409423828125,
                               0.903382003307342529296875,
                               0.2905881404876708984375,
                               0.33750665187835693359375,
                               0.3232279717922210693359375,
                               0.07346880435943603515625,
                               0.3991589844226837158203125,
                               0.903037011623382568359375,
                               0.083290748298168182373046875,
                               0.20850212872028350830078125,
                               0.05971308052539825439453125,
                               0.4810305535793304443359375,
                               0.087783016264438629150390625,
                               0.2952007353305816650390625,
                               0.2153458297252655029296875,
                               0.4049233496189117431640625,
                               0.7175214290618896484375,
                               0.872620165348052978515625,
                               0.522788941860198974609375,
                               0.43519175052642822265625,
                               0.0193175189197063446044921875,
                               0.846780240535736083984375,
                               0.5219886302947998046875,
                               0.242856085300445556640625,
                               0.2003507316112518310546875,
                               0.8327982425689697265625,
                               0.18934874236583709716796875,
                               0.917275846004486083984375,
                               0.658357441425323486328125,
                               0.847428977489471435546875,
                               0.81426322460174560546875,
                               0.036692313849925994873046875,
                               0.132266581058502197265625,
                               0.357086241245269775390625,
                               0.4745192825794219970703125,
                               0.821886956691741943359375,
                               0.2454545795917510986328125,
                               0.1065533459186553955078125,
                               0.791345179080963134765625,
                               0.545370578765869140625,
                               0.3979628086090087890625,
                               0.49180948734283447265625,
                               0.1297818124294281005859375,
                               0.36476039886474609375,
                               0.3085542619228363037109375,
                               0.899958193302154541015625,
                               0.4159581661224365234375,
                               0.675307571887969970703125,
                               0.829472124576568603515625,
                               0.2064842283725738525390625,
                               0.64016926288604736328125,
                               0.20317254960536956787109375,
                               0.61657464504241943359375,
                               0.290811240673065185546875,
                               0.26665222644805908203125,
                               0.3393469750881195068359375,
                               0.2539980709552764892578125,
                               0.791014850139617919921875,
                               0.940179288387298583984375,
                               0.827880084514617919921875,
                               0.460959732532501220703125,
                               0.63165509700775146484375,
                               0.1342843472957611083984375,
                               0.583048880100250244140625,
                               0.4310896396636962890625,
                               0.070260427892208099365234375,
                               0.518509685993194580078125,
                               0.255076229572296142578125,
                               0.588839232921600341796875,
                               0.13979454338550567626953125,
                               0.816810190677642822265625,
                               0.506142139434814453125,
                               0.780538499355316162109375,
                               0.70891857147216796875,
                               0.775202929973602294921875,
                               0.33364391326904296875,
                               0.21829630434513092041015625,
                               0.794861137866973876953125,
                               0.440593779087066650390625,
                               0.51086711883544921875,
                               0.059619002044200897216796875,
                               0.626003265380859375,
                               0.831237018108367919921875,
                               0.775263965129852294921875,
                               0.48013699054718017578125,
                               0.98830425739288330078125,
                               0.5461161136627197265625,
                               0.0545087419450283050537109375,
                               0.067873962223529815673828125,
                               0.334798395633697509765625,
                               0.083531044423580169677734375,
                               0.1419331729412078857421875,
                               0.62124884128570556640625,
                               0.4215275943279266357421875,
                               0.349430382251739501953125,
                               0.645228683948516845703125,
                               0.15098969638347625732421875,
                               0.789717197418212890625,
                               0.59648799896240234375,
                               0.3775124251842498779296875,
                               0.2767163217067718505859375,
                               0.558230340480804443359375,
                               0.991863429546356201171875,
                               0.813561499118804931640625,
                               0.79598820209503173828125,
                               0.567295074462890625,
                               0.4774146378040313720703125,
                               0.3510249555110931396484375,
                               0.681096494197845458984375,
                               0.745837032794952392578125,
                               0.681192934513092041015625,
                               0.88084888458251953125,
                               0.52995645999908447265625,
                               0.087239809334278106689453125,
                               0.414192855358123779296875,
                               0.539312899112701416015625,
                               0.23079840838909149169921875,
                               0.548077642917633056640625,
                               0.3750600516796112060546875,
                               0.3628396093845367431640625,
                               0.078880332410335540771484375,
                               0.95263445377349853515625,
                               0.41051447391510009765625,
                               0.820193827152252197265625,
                               0.4604322016239166259765625,
                               0.3603973090648651123046875,
                               0.5672309398651123046875,
                               0.685865581035614013671875,
                               0.7147781848907470703125,
                               0.772135257720947265625,
                               0.623492062091827392578125,
                               0.7632234096527099609375,
                               0.877109348773956298828125,
                               0.096309013664722442626953125,
                               0.21554203331470489501953125,
                               0.254471242427825927734375,
                               0.58027327060699462890625,
                               0.3754498958587646484375,
                               0.717136919498443603515625,
                               0.2995398044586181640625,
                               0.931284368038177490234375,
                               0.011751591227948665618896484375,
                               0.07255984842777252197265625,
                               0.87918460369110107421875,
                               0.02955267764627933502197265625,
                               0.889126598834991455078125,
                               0.0329551957547664642333984375,
                               0.23701806366443634033203125,
                               0.5436298847198486328125,
                               0.4716108739376068115234375,
                               0.1311373412609100341796875,
                               0.983278572559356689453125,
                               0.571916878223419189453125,
                               0.739863812923431396484375,
                               0.28372323513031005859375,
                               0.18242438137531280517578125,
                               0.522270500659942626953125,
                               0.880189239978790283203125,
                               0.530347883701324462890625,
                               0.3022750318050384521484375,
                               0.02125177718698978424072265625,
                               0.76706016063690185546875,
                               0.666437804698944091796875,
                               0.5887668132781982421875,
                               0.3817012608051300048828125,
                               0.069761075079441070556640625,
                               0.13000230491161346435546875,
                               0.3799968063831329345703125,
                               0.92774105072021484375,
                               0.2970103323459625244140625,
                               0.2885017096996307373046875,
                               0.644755303859710693359375,
                               0.4826243221759796142578125,
                               0.02549990825355052947998046875,
                               0.845977962017059326171875,
                               0.1354812681674957275390625,
                               0.59001064300537109375,
                               0.786619603633880615234375,
                               0.808787405490875244140625,
                               0.850969374179840087890625,
                               0.864635884761810302734375,
                               0.9816544055938720703125,
                               0.704220354557037353515625,
                               0.406329214572906494140625,
                               0.4230716228485107421875,
                               0.410357534885406494140625,
                               0.7462520599365234375,
                               0.251948177814483642578125,
                               0.3785230815410614013671875,
                               0.704321324825286865234375,
                               0.0714503824710845947265625,
                               0.906627714633941650390625,
                               0.0333719812333583831787109375, 0.654077053070068359375};
  std::vector<int64_t> x_dims = {1, 3, 9, 9};
  std::vector<int64_t> expected_dims = {1, 3, 7, 7};
  std::vector<float> expected_vals = {2.1165919303894043, 1.9042642116546631, 1.5751385688781738,
                                      1.4826388359069824, 1.5885931253433228, 1.7165449857711792,
                                      1.8440124988555908, 1.9269057512283325, 1.7515288591384888,
                                      1.5131627321243286, 1.5648597478866577, 1.7481330633163452,
                                      1.8362259864807129, 1.8987786769866943, 2.056734561920166,
                                      1.7989484071731567, 1.476754903793335, 1.4329502582550049,
                                      1.9585609436035156, 2.0552983283996582, 2.0338289737701416,
                                      2.1123726367950439, 1.9154638051986694, 1.8470758199691772,
                                      1.7075581550598145, 2.0650856494903564, 1.8786256313323975,
                                      1.6601848602294922, 2.0838139057159424, 1.9302912950515747,
                                      1.7651937007904053, 1.3319482803344727, 1.6723839044570923,
                                      1.6038172245025635, 1.281104564666748, 1.7076961994171143,
                                      1.8572235107421875, 1.9256408214569092, 1.5551244020462036,
                                      1.3944330215454102, 1.4710251092910767, 1.2723797559738159,
                                      1.5805213451385498, 1.786491870880127, 1.9965716600418091,
                                      1.6089824438095093, 1.6536226272583008, 1.7216441631317139,
                                      1.6427503824234009, 1.2622216939926147, 1.3339006900787354,
                                      1.5921475887298584, 1.4477853775024414, 1.5451828241348267,
                                      1.7485626935958862, 1.9603283405303955, 1.5874154567718506,
                                      1.174997091293335, 1.5267566442489624, 1.3757904767990112,
                                      1.4901281595230103, 1.6068876981735229, 1.7605991363525391,
                                      1.7780805826187134, 1.441672682762146, 1.6808938980102539,
                                      1.4773738384246826, 1.5793166160583496, 1.5747464895248413,
                                      1.6349068880081177, 1.8485732078552246, 1.4251554012298584,
                                      1.7163872718811035, 1.7315287590026855, 1.9817506074905396,
                                      1.7880076169967651, 1.7050145864486694, 1.557621955871582,
                                      1.2333823442459106, 1.5207540988922119, 1.6104618310928345,
                                      1.9518419504165649, 1.8223953247070312, 1.8038734197616577,
                                      1.567004919052124, 1.2572110891342163, 1.3791522979736328,
                                      1.3418225049972534, 1.6210030317306519, 1.8650168180465698,
                                      2.1098208427429199, 1.5974785089492798, 1.3397328853607178,
                                      1.435505747795105, 1.3628946542739868, 1.558194637298584,
                                      1.9369972944259644, 2.0405406951904297, 1.69834303855896,
                                      1.5347113609313965, 1.1952571868896484, 1.36539626121521,
                                      1.5550618171691895, 1.6876083612442017, 1.8127884864807129,
                                      1.8130189180374146, 1.6180311441421509, 1.2502912282943726,
                                      1.7129987478256226, 1.6241954565048218, 1.848590612411499,
                                      1.6104695796966553, 1.8547911643981934, 1.7072041034698486,
                                      1.6555715799331665, 1.722585916519165, 1.4128400087356567,
                                      1.4920854568481445, 1.4759902954101562, 1.5659612417221069,
                                      1.664239764213562, 1.9113870859146118, 1.9744715690612793,
                                      1.5460153818130493, 1.315888524055481, 1.2653214931488037,
                                      1.6348875761032104, 1.7388149499893188, 1.8604984283447266,
                                      1.751006007194519, 1.4989807605743408, 1.3538862466812134,
                                      1.4081637859344482, 1.7571074962615967, 1.9259833097457886,
                                      1.9354615211486816, 2.1322348117828369, 1.9475457668304443,
                                      1.7524666786193848, 1.3199115991592407, 1.8716570138931274,
                                      1.9475260972976685, 1.8482059240341187, 1.9523605108261108,
                                      2.0444071292877197, 1.8444844484329224, 1.6809544563293457};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider});
}

// test data generated with lp_pool_test_generator.py
TEST(PoolTest, LpPool1d) {
  std::vector<int64_t> kernel_sizes[2] = {{2}, {3}};
  std::vector<int64_t> strides[2] = {{1}, {2}};
  std::vector<float> ys[4] = {
      {2.2361f, 3.6056f, 5.0000f},
      {2.2361f, 5.0000f},
      {3.7417f, 5.3852f},
      {3.7417f}};
  std::vector<int64_t> y_sizes[4] = {
      {1, 1, 3},
      {1, 1, 2},
      {1, 1, 2},
      {1, 1, 1},
  };
  int y_count = 0;
  for (int kernel_size_count = 0; kernel_size_count < 2; kernel_size_count++)
    for (int stride_count = 0; stride_count < 2; stride_count++) {
      OpTester test("LpPool", 18);
      test.AddAttribute("auto_pad", "");
      test.AddAttribute("p", static_cast<int64_t>(2));
      test.AddInput<float>("X", {1, 1, 4}, {1, 2, 3, 4});
      test.AddAttribute("strides", strides[stride_count]);
      test.AddAttribute("kernel_shape", kernel_sizes[kernel_size_count]);

      test.AddOutput<float>("Y", y_sizes[y_count], ys[y_count]);

      // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a94f434942252e6d98ac17705c06ce060
      // TensorRT does not support 1d pooling
      test.Run(OpTester::ExpectResult::kExpectSuccess, "",
               {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider});
      y_count++;
    }
}

// test data generated with lp_pool_test_generator.py
TEST(PoolTest, LpPool2d) {
  std::vector<int64_t> kernel_sizes[2] = {{2, 2}, {3, 3}};
  std::vector<int64_t> strides[2] = {{1, 1}, {2, 2}};
  std::vector<float> ys[4] = {
      {8.1240f, 9.8995f, 11.7473f, 15.5563f, 17.4929f, 19.4422f, 23.3666f, 25.3377f, 27.3130f},
      {8.1240f, 11.7473f, 23.3666f, 27.3130f},
      {20.6398f, 23.3024f, 31.6544f, 34.5109f},
      {20.6398f}};
  std::vector<int64_t> y_sizes[4] = {
      {1, 1, 3, 3},
      {1, 1, 2, 2},
      {1, 1, 2, 2},
      {1, 1, 1, 1},
  };
  int y_count = 0;
  for (int kernel_size_count = 0; kernel_size_count < 2; kernel_size_count++)
    for (int stride_count = 0; stride_count < 2; stride_count++) {
      OpTester test("LpPool", 18);
      test.AddAttribute("auto_pad", "");
      test.AddAttribute("p", static_cast<int64_t>(2));
      test.AddInput<float>("X", {1, 1, 4, 4},
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
      test.AddAttribute("strides", strides[stride_count]);
      test.AddAttribute("kernel_shape", kernel_sizes[kernel_size_count]);

      test.AddOutput<float>("Y", y_sizes[y_count], ys[y_count]);
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider});
      y_count++;
    }
}

TEST(PoolTest, LpPoolCeilMode) {
  OpTester test("LpPool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("ceil_mode", static_cast<int64_t>(1));
  test.AddAttribute("p", static_cast<int64_t>(1));
  test.AddInput<float>("X", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("Y", {1, 1, 2}, {6, 7});

  // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a94f434942252e6d98ac17705c06ce060
  // TensorRT does not support 1d pooling
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider});
}

TEST(PoolTest, GlobalLpPool) {
  OpTester test("GlobalLpPool");
  test.AddAttribute("p", static_cast<int64_t>(3));
  std::vector<float> x_vals = {0.688458621501922607421875,
                               0.8835647106170654296875,
                               0.782541573047637939453125,
                               0.300049364566802978515625,
                               0.8066387176513671875,
                               0.4520850479602813720703125,
                               0.598959147930145263671875,
                               0.4597113132476806640625,
                               0.8161861896514892578125,
                               0.7667262554168701171875,
                               0.840198040008544921875,
                               0.583297073841094970703125,
                               0.708858668804168701171875,
                               0.4728293716907501220703125,
                               0.4992314875125885009765625,
                               0.2504110038280487060546875,
                               0.3881411850452423095703125,
                               0.517398893833160400390625,
                               0.657192409038543701171875,
                               0.7325098514556884765625,
                               0.10206781327724456787109375,
                               0.2179393768310546875,
                               0.0616043470799922943115234375,
                               0.475992143154144287109375,
                               0.737536609172821044921875,
                               0.9689886569976806640625,
                               0.4474093914031982421875,
                               0.4323260486125946044921875,
                               0.648917853832244873046875,
                               0.701454102993011474609375,
                               0.107639573514461517333984375,
                               0.811198413372039794921875,
                               0.725269258022308349609375,
                               0.7497208118438720703125,
                               0.3340204060077667236328125,
                               0.87611293792724609375,
                               0.6691205501556396484375,
                               0.87189638614654541015625,
                               0.971237838268280029296875,
                               0.11620916426181793212890625,
                               0.0249019563198089599609375,
                               0.752140820026397705078125,
                               0.865541160106658935546875,
                               0.015474068932235240936279296875,
                               0.5126011371612548828125,
                               0.45315420627593994140625,
                               0.1925573647022247314453125,
                               0.98408544063568115234375,
                               0.14754636585712432861328125,
                               0.54990971088409423828125,
                               0.903382003307342529296875,
                               0.2905881404876708984375,
                               0.33750665187835693359375,
                               0.3232279717922210693359375,
                               0.07346880435943603515625,
                               0.3991589844226837158203125,
                               0.903037011623382568359375,
                               0.083290748298168182373046875,
                               0.20850212872028350830078125,
                               0.05971308052539825439453125,
                               0.4810305535793304443359375,
                               0.087783016264438629150390625,
                               0.2952007353305816650390625,
                               0.2153458297252655029296875,
                               0.4049233496189117431640625,
                               0.7175214290618896484375,
                               0.872620165348052978515625,
                               0.522788941860198974609375,
                               0.43519175052642822265625,
                               0.0193175189197063446044921875,
                               0.846780240535736083984375,
                               0.5219886302947998046875,
                               0.242856085300445556640625,
                               0.2003507316112518310546875,
                               0.8327982425689697265625,
                               0.18934874236583709716796875,
                               0.917275846004486083984375,
                               0.658357441425323486328125,
                               0.847428977489471435546875,
                               0.81426322460174560546875,
                               0.036692313849925994873046875,
                               0.132266581058502197265625,
                               0.357086241245269775390625,
                               0.4745192825794219970703125,
                               0.821886956691741943359375,
                               0.2454545795917510986328125,
                               0.1065533459186553955078125,
                               0.791345179080963134765625,
                               0.545370578765869140625,
                               0.3979628086090087890625,
                               0.49180948734283447265625,
                               0.1297818124294281005859375,
                               0.36476039886474609375,
                               0.3085542619228363037109375,
                               0.899958193302154541015625,
                               0.4159581661224365234375,
                               0.675307571887969970703125,
                               0.829472124576568603515625,
                               0.2064842283725738525390625,
                               0.64016926288604736328125,
                               0.20317254960536956787109375,
                               0.61657464504241943359375,
                               0.290811240673065185546875,
                               0.26665222644805908203125,
                               0.3393469750881195068359375,
                               0.2539980709552764892578125,
                               0.791014850139617919921875,
                               0.940179288387298583984375,
                               0.827880084514617919921875,
                               0.460959732532501220703125,
                               0.63165509700775146484375,
                               0.1342843472957611083984375,
                               0.583048880100250244140625,
                               0.4310896396636962890625,
                               0.070260427892208099365234375,
                               0.518509685993194580078125,
                               0.255076229572296142578125,
                               0.588839232921600341796875,
                               0.13979454338550567626953125,
                               0.816810190677642822265625,
                               0.506142139434814453125,
                               0.780538499355316162109375,
                               0.70891857147216796875,
                               0.775202929973602294921875,
                               0.33364391326904296875,
                               0.21829630434513092041015625,
                               0.794861137866973876953125,
                               0.440593779087066650390625,
                               0.51086711883544921875,
                               0.059619002044200897216796875,
                               0.626003265380859375,
                               0.831237018108367919921875,
                               0.775263965129852294921875,
                               0.48013699054718017578125,
                               0.98830425739288330078125,
                               0.5461161136627197265625,
                               0.0545087419450283050537109375,
                               0.067873962223529815673828125,
                               0.334798395633697509765625,
                               0.083531044423580169677734375,
                               0.1419331729412078857421875,
                               0.62124884128570556640625,
                               0.4215275943279266357421875,
                               0.349430382251739501953125,
                               0.645228683948516845703125,
                               0.15098969638347625732421875,
                               0.789717197418212890625,
                               0.59648799896240234375,
                               0.3775124251842498779296875,
                               0.2767163217067718505859375,
                               0.558230340480804443359375,
                               0.991863429546356201171875,
                               0.813561499118804931640625,
                               0.79598820209503173828125,
                               0.567295074462890625,
                               0.4774146378040313720703125,
                               0.3510249555110931396484375,
                               0.681096494197845458984375,
                               0.745837032794952392578125,
                               0.681192934513092041015625,
                               0.88084888458251953125,
                               0.52995645999908447265625,
                               0.087239809334278106689453125,
                               0.414192855358123779296875,
                               0.539312899112701416015625,
                               0.23079840838909149169921875,
                               0.548077642917633056640625,
                               0.3750600516796112060546875,
                               0.3628396093845367431640625,
                               0.078880332410335540771484375,
                               0.95263445377349853515625,
                               0.41051447391510009765625,
                               0.820193827152252197265625,
                               0.4604322016239166259765625,
                               0.3603973090648651123046875,
                               0.5672309398651123046875,
                               0.685865581035614013671875,
                               0.7147781848907470703125,
                               0.772135257720947265625,
                               0.623492062091827392578125,
                               0.7632234096527099609375,
                               0.877109348773956298828125,
                               0.096309013664722442626953125,
                               0.21554203331470489501953125,
                               0.254471242427825927734375,
                               0.58027327060699462890625,
                               0.3754498958587646484375,
                               0.717136919498443603515625,
                               0.2995398044586181640625,
                               0.931284368038177490234375,
                               0.011751591227948665618896484375,
                               0.07255984842777252197265625,
                               0.87918460369110107421875,
                               0.02955267764627933502197265625,
                               0.889126598834991455078125,
                               0.0329551957547664642333984375,
                               0.23701806366443634033203125,
                               0.5436298847198486328125,
                               0.4716108739376068115234375,
                               0.1311373412609100341796875,
                               0.983278572559356689453125,
                               0.571916878223419189453125,
                               0.739863812923431396484375,
                               0.28372323513031005859375,
                               0.18242438137531280517578125,
                               0.522270500659942626953125,
                               0.880189239978790283203125,
                               0.530347883701324462890625,
                               0.3022750318050384521484375,
                               0.02125177718698978424072265625,
                               0.76706016063690185546875,
                               0.666437804698944091796875,
                               0.5887668132781982421875,
                               0.3817012608051300048828125,
                               0.069761075079441070556640625,
                               0.13000230491161346435546875,
                               0.3799968063831329345703125,
                               0.92774105072021484375,
                               0.2970103323459625244140625,
                               0.2885017096996307373046875,
                               0.644755303859710693359375,
                               0.4826243221759796142578125,
                               0.02549990825355052947998046875,
                               0.845977962017059326171875,
                               0.1354812681674957275390625,
                               0.59001064300537109375,
                               0.786619603633880615234375,
                               0.808787405490875244140625,
                               0.850969374179840087890625,
                               0.864635884761810302734375,
                               0.9816544055938720703125,
                               0.704220354557037353515625,
                               0.406329214572906494140625,
                               0.4230716228485107421875,
                               0.410357534885406494140625,
                               0.7462520599365234375,
                               0.251948177814483642578125,
                               0.3785230815410614013671875,
                               0.704321324825286865234375,
                               0.0714503824710845947265625,
                               0.906627714633941650390625,
                               0.0333719812333583831787109375, 0.654077053070068359375};
  std::vector<int64_t> x_dims = {1, 3, 9, 9};
  std::vector<int64_t> expected_dims = {1, 3, 1, 1};
  std::vector<float> expected_vals = {2.7576668262481689453125, 2.6182243824005126953125,
                                      2.682276248931884765625};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider});
}

TEST(PoolTest, MaxPoolDimWithZeroForN) {
  OpTester test("MaxPool", 10);
  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2});

  std::vector<float> x_vals = {};
  std::vector<int64_t> x_dims = {0, 2, 4};  // N of 0 should be handled
  std::vector<int64_t> expected_dims = {0, 2, 2};
  std::vector<float> expected_vals = {};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  // TODO: Fix WebGPU Transpose error: "Invalid dispatch group size (0, 1, 1)".
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kWebGpuExecutionProvider});
}

// Verify that non-positive stride/dilation values are rejected by PoolAttributes kernel validation.
// AddShapeToTensorData(false) omits input shape from the graph so ONNX shape inference is bypassed
// (convPoolShapeInference returns early when hasInputShape is false). This lets the model pass
// Graph::Resolve() and reach kernel construction where our ORT_ENFORCE checks fire.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, MaxPool_ZeroStride) {
  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{0, 0});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "All stride values must be positive",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, AveragePool_ZeroStride) {
  OpTester test("AveragePool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{0, 0});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("count_include_pad", static_cast<int64_t>(0));

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "All stride values must be positive",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, LpPool_ZeroStride) {
  OpTester test("LpPool", 18);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{0, 0});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "All stride values must be positive",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, MaxPool_ZeroDilation) {
  OpTester test("MaxPool", 10);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("dilations", std::vector<int64_t>{0, 0});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "All dilation values must be positive",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a 'pads' attribute shorter than 2 * kernel_shape rank is rejected by PoolAttributes
// kernel validation. AddShapeToTensorData(false) omits input shape from the graph so ONNX shape
// inference is bypassed and the model reaches kernel construction where the ORT_ENFORCE fires.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, MaxPool_PadsTooShort) {
  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "twice the kernel_shape rank",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, AveragePool_PadsTooShort) {
  OpTester test("AveragePool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("count_include_pad", static_cast<int64_t>(0));

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "twice the kernel_shape rank",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

TEST(PoolTest, LpPool_PadsTooShort) {
  OpTester test("LpPool", 18);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "twice the kernel_shape rank",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a kernel_shape whose rank does not match the input spatial rank is rejected by
// PoolAttributes::InferOutputSize. LpPool (opset < 18) uses the generic Pool::Compute path, which
// has no earlier rank guard, so the request reaches the InferOutputSize validation directly.
// AddShapeToTensorData(false) omits the input shape so ONNX shape inference is bypassed.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, LpPool_KernelRankMismatch) {
  OpTester test("LpPool", 11);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  // Input spatial rank is 3 while kernel_shape rank is 2.
  std::vector<float> x_vals(1 * 1 * 8 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "input spatial rank",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that attribute values producing a non-positive output dimension are rejected by
// PoolAttributes::ComputeOutputSize. A 3x3 kernel over a 1x1 spatial input with no padding makes
// the padded input smaller than the dilated kernel, so the computed output dimension is negative.
// AddShapeToTensorData(false) omits the input shape so ONNX shape inference is bypassed.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, AveragePool_NegativeOutputDim) {
  OpTester test("AveragePool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("count_include_pad", static_cast<int64_t>(0));

  std::vector<float> x_vals(1 * 1 * 1 * 1, 1.0f);
  test.AddInput<float>("X", {1, 1, 1, 1}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "output dimension is negative",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a 'pads' attribute longer than 2 * kernel_shape rank is rejected by the
// PoolAttributes constructor. AddShapeToTensorData(false) omits the input shape so ONNX shape
// inference is bypassed and the model reaches kernel construction where the ORT_ENFORCE fires.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, MaxPool_PadsTooLong) {
  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "twice the kernel_shape rank",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a 'strides' attribute whose length does not match the kernel_shape rank is rejected
// by the PoolAttributes constructor. The assertion pins the explicit strides length message
// ("Strides dimensions should match kernel shape"). AddShapeToTensorData(false) bypasses shape
// inference. Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, MaxPool_StridesLengthMismatch) {
  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Strides dimensions should match kernel shape",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a 'dilations' attribute whose length does not match the kernel_shape rank is
// rejected by the PoolAttributes constructor. AddShapeToTensorData(false) bypasses shape inference.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML) that produce
// different error messages.
TEST(PoolTest, MaxPool_DilationsLengthMismatch) {
  // 'dilations' is only a valid MaxPool attribute from opset 10 onward, so target that version.
  OpTester test("MaxPool", 10);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1, 1});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Dilations dimensions should match kernel shape",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a low-rank input is rejected before pooling. The rank check in SetOutputSize
// (NumDimensions >= 2) is defense-in-depth for execution providers and direct callers, but the CPU
// pooling kernels reject inputs with rank < 3 earlier in Compute, so the tested path fires that
// earlier guard and this test pins its message ("Input dimension cannot be less than 3").
// AddShapeToTensorData(false) bypasses shape inference so the rank-2 input reaches the kernel.
// Exclude compiling EPs (TRT, QNN) and EPs with their own validation (DML).
TEST(PoolTest, MaxPool_InputRankTooLow) {
  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> x_vals(4 * 4, 1.0f);
  test.AddInput<float>("X", {4, 4}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Input dimension cannot be less than 3",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Verify that a MaxPool 'storage_order' outside the valid set {0, 1} is rejected by the
// PoolAttributes constructor. storage_order was introduced in MaxPool opset 8, so target that
// version. AddShapeToTensorData(false) bypasses shape inference so the model reaches kernel
// construction where the ORT_ENFORCE fires. Exclude compiling EPs (TRT, QNN) and EPs with their
// own validation (DML) that produce different error messages.
TEST(PoolTest, MaxPool_InvalidStorageOrder) {
  OpTester test("MaxPool", 8);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("storage_order", static_cast<int64_t>(2));

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.Run(OpTester::ExpectResult::kExpectFailure, "storage_order must be 0",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// DML EP validates stride/dilation in OperatorHelper.cpp (KernelHelper constructor) via
// ML_CHECK_VALID_ARGUMENT_MSG, but the descriptive message is lost when the exception crosses
// the COM/HRESULT boundary (CATCH_RETURN strips the message, THROW_IF_FAILED re-throws with
// just E_INVALIDARG). We still verify that DML rejects the invalid values by matching the
// Win32 text for E_INVALIDARG (0x80070057).
TEST(PoolTest, MaxPool_ZeroStride_Dml) {
  if (DefaultDmlExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "DML EP not available";
  }

  OpTester test("MaxPool");
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{0, 0});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.ConfigEp(DefaultDmlExecutionProvider())
      .Config(OpTester::ExpectResult::kExpectFailure, "The parameter is incorrect")
      .RunWithConfig();
}

TEST(PoolTest, MaxPool_ZeroDilation_Dml) {
  if (DefaultDmlExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "DML EP not available";
  }

  OpTester test("MaxPool", 10);
  test.AddShapeToTensorData(false);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("dilations", std::vector<int64_t>{0, 0});

  std::vector<float> x_vals(1 * 1 * 8 * 8, 1.0f);
  test.AddInput<float>("X", {1, 1, 8, 8}, x_vals);
  test.AddOutput<float>("Y", {0}, {});

  test.ConfigEp(DefaultDmlExecutionProvider())
      .Config(OpTester::ExpectResult::kExpectFailure, "The parameter is incorrect")
      .RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
