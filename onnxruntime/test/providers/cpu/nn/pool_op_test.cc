// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/pool.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because "pads" attribute is not supported

TEST(PoolTest, MaxPool) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{8, 8});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: result differs
}

// Only CUDA kernel has float 16 support
// Disable for now, still investigating the issue with cudnn lib
#ifdef USE_CUDA
TEST(PoolTest, MaxPool_F16) {
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{8, 8});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Assertion `!attrs.count("pads")' failed
}
#endif

static void MaxPool_8_WithIndexTest(bool has_index, int64_t storage_order = 0) {
  OpTester test("MaxPool", 8);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{8, 8});
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDnnlExecutionProvider, kTensorrtExecutionProvider, kAclExecutionProvider, kArmNNExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(PoolTest, MaxPool_8_With_Index) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  MaxPool_8_WithIndexTest(false);                      // row major
  MaxPool_8_WithIndexTest(true, 0 /*storage_order*/);  // row major
  MaxPool_8_WithIndexTest(true, 1 /*storage_order*/);  // col major
}

TEST(PoolTest, MaxPool1D) {
  OpTester test("MaxPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<float> expected_vals = {2, 4, 6, 8};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

static void MaxPool1D_8_WithIndexTest(int64_t storage_order) {
  OpTester test("MaxPool", 8);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<float> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<float> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool1D_8_With_Index) {
  MaxPool1D_8_WithIndexTest(0 /*storage_order*/);
  MaxPool1D_8_WithIndexTest(1 /*storage_order*/);
}

static void MaxPool1D_12_WithIndexTest_int8(int64_t storage_order) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<int8_t> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<int8_t> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<int8_t>("X", x_dims, x_vals);
  test.AddOutput<int8_t>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

static void MaxPool1D_12_WithIndexTest_uint8(int64_t storage_order) {
  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});
  test.AddAttribute("storage_order", storage_order);

  std::vector<uint8_t> x_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> x_dims = {1, 2, 4};
  std::vector<int64_t> expected_dims = {1, 2, 2};
  std::vector<uint8_t> expected_vals = {2, 4, 6, 8};
  std::vector<int64_t> expected_indices = {1, 3, 5, 7};

  test.AddInput<uint8_t>("X", x_dims, x_vals);
  test.AddOutput<uint8_t>("Y", expected_dims, expected_vals);
  test.AddOutput<int64_t>("Indices", expected_dims, expected_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
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
#if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
#else
  test.Run();
#endif
}

TEST(PoolTest, MaxPool_10_Dilation_1d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{3});
  test.AddAttribute("dilations", vector<int64_t>{3});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1, -3, -2, -4, -6, -5, -4, -2};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 6};
  std::vector<float> expected_vals = {4, 3, 2, 4, -1, -2};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", vector<int64_t>{1, 1});
  test.AddAttribute("kernel_shape", vector<int64_t>{3});
  test.AddAttribute("dilations", vector<int64_t>{3});

  std::vector<float> x_vals = {
      1, 3, 2, 4, -1, -3, -2, -4, -6, -5, -4, -2};
  std::vector<int64_t> x_dims = {1, 1, 12};
  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals = {2, 4, 3, 2, 4, -1, -2, -2};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_2d_int8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, MaxPool_10_DilationPadding_2d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});

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
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_Ceil0_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool_12_Dilation_Ceil0_2d_int8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 12);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool_10_Dilation_Ceil1_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2});
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, MaxPool_10_DilationPadding_3d) {
  OpTester test("MaxPool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1, 1});
  test.AddAttribute("pads", vector<int64_t>{1, 1, 1, 1, 1, 1});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});
  test.AddAttribute("dilations", vector<int64_t>{2, 2, 2});

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
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(PoolTest, GlobalMaxPool) {
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

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(PoolTest, GlobalMaxPool3D) {
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

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(PoolTest, AveragePool) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{8, 8});

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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// test 'strides' attribute not specified
TEST(PoolTest, AveragePool_DefaultStrides) {
  OpTester test("AveragePool");
  test.AddAttribute("kernel_shape", vector<int64_t>{2});
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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool", 10);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{3, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(PoolTest, AveragePool_19_dilation_2d) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("AveragePool", 19);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kAclExecutionProvider, kOpenVINOExecutionProvider});
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
  test.Run();
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
  test.Run();
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
  test.Run();
}

TEST(PoolTest, LpPool) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("LpPool");

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{3, 3});

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
  test.Run();
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
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
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
      test.Run();
      y_count++;
    }
}

TEST(PoolTest, LpPoolCeilMode) {
  OpTester test("LpPool", 18);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{3});
  test.AddAttribute("ceil_mode", static_cast<int64_t>(1));
  test.AddAttribute("p", static_cast<int64_t>(1));
  test.AddInput<float>("X", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("Y", {1, 1, 2}, {6, 7});

  // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a94f434942252e6d98ac17705c06ce060
  // TensorRT does not support 1d pooling
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
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
  test.Run();
}

TEST(PoolTest, MaxPoolDimWithZeroForN) {
  OpTester test("MaxPool", 10);
  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> x_vals = {};
  std::vector<int64_t> x_dims = {0, 2, 4};  // N of 0 should be handled
  std::vector<int64_t> expected_dims = {0, 2, 2};
  std::vector<float> expected_vals = {};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(PoolTest, MaxPoolOutputCeilModeSizeReduceByOne) {
  OpTester test("MaxPool", 12);
  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{0, 0});
  test.AddAttribute("kernel_shape", vector<int64_t>{1});
  test.AddAttribute("ceil_mode", static_cast<int64_t>(1));

  std::vector<float> x_vals = {1, 2,};
  std::vector<int64_t> x_dims = {1, 1, 2};  // N of 0 should be handled
  std::vector<int64_t> expected_dims = {1, 1, 1};
  std::vector<float> expected_vals = {2};

  test.AddInput<float>("X", x_dims, x_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
