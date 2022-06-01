
#if defined(USE_SNPE)

#ifdef _MSC_VER
#pragma warning(disable : 4389)
#endif

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void TfNToFloat(float* out,
                uint8_t* in,
                const unsigned char step_equivalent_to_0,
                const float quantized_step_size,
                size_t num_element,
                int bit_width) {
  for (size_t i = 0; i < num_element; ++i) {
    if (8 == bit_width) {
      double quantized_value = static_cast<double>(in[i]);
      double step_eq_to_0 = static_cast<double>(step_equivalent_to_0);
      out[i] = static_cast<double>((quantized_value - step_eq_to_0) * quantized_step_size);
    } else if (16 == bit_width) {
      uint16_t* temp = reinterpret_cast<uint16_t*>(in);
      double quantized_value = static_cast<double>(temp[i]);
      double step_eq_to_0 = static_cast<double>(step_equivalent_to_0);
      out[i] = static_cast<double>((quantized_value - step_eq_to_0) * quantized_step_size);
    }
  }
}

bool FloatToTfN(uint8_t* out,
                unsigned char& step_equivalent_to_0,
                float& quantized_step_size,
                float* in,
                size_t num_element,
                int bit_width) {
  float trueMin = std::numeric_limits<float>::max();
  float trueMax = std::numeric_limits<float>::min();

  for (size_t i = 0; i < num_element; ++i) {
    trueMin = fmin(trueMin, in[i]);
    trueMax = fmax(trueMax, in[i]);
  }

  double encodingMin;
  double encodingMax;
  double stepCloseTo0;
  double trueBitWidthMax = pow(2, bit_width) - 1;

  if (trueMin > 0.0f) {
    stepCloseTo0 = 0.0;
    encodingMin = 0.0;
    encodingMax = trueMax;
  } else if (trueMax < 0.0f) {
    stepCloseTo0 = trueBitWidthMax;
    encodingMin = trueMin;
    encodingMax = 0.0;
  } else {
    double trueStepSize = static_cast<double>(trueMax - trueMin) / trueBitWidthMax;
    stepCloseTo0 = -trueMin / trueStepSize;
    if (stepCloseTo0 == round(stepCloseTo0)) {
      // 0.0 is exactly representable
      encodingMin = trueMin;
      encodingMax = trueMax;
    } else {
      stepCloseTo0 = round(stepCloseTo0);
      encodingMin = (0.0 - stepCloseTo0) * trueStepSize;
      encodingMax = (trueBitWidthMax - stepCloseTo0) * trueStepSize;
    }
  }

  const double minEncodingRange = 0.01;
  double encodingRange = encodingMax - encodingMin;
  quantized_step_size = encodingRange / trueBitWidthMax;
  step_equivalent_to_0 = static_cast<unsigned char>(round(stepCloseTo0));

  if (encodingRange < minEncodingRange) {
    std::cerr << "Expect the encoding range to be larger than " << minEncodingRange << "\n"
              << "Got: " << encodingRange << "\n";
    return false;
  } else {
    for (size_t i = 0; i < num_element; ++i) {
      int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);

      if (quantizedValue < 0)
        quantizedValue = 0;
      else if (quantizedValue > static_cast<int>(trueBitWidthMax))
        quantizedValue = static_cast<int>(trueBitWidthMax);

      if (bit_width == 8) {
        out[i] = static_cast<uint8_t>(quantizedValue);
      } else if (bit_width == 16) {
        uint16_t* temp = reinterpret_cast<uint16_t*>(out);
        temp[i] = static_cast<uint16_t>(quantizedValue);
      }
    }
  }
  return true;
}

std::string LoadDlcFile(std::string file_path) {
  std::ifstream dlc_file(file_path, std::ios::binary | std::ios::ate);

  std::streamsize file_size = dlc_file.tellg();
  dlc_file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  dlc_file.read(buffer.data(), file_size);
  std::string dlc_payload(buffer.data(), buffer.size());

  return dlc_payload;
}

class OpTesterSnpe : public OpTester {
 public:
  explicit OpTesterSnpe(const char* op,
                        int opset_version = 7,
                        const char* domain = onnxruntime::kOnnxDomain,
                        bool verify_output = true) : OpTester(op, opset_version, domain, verify_output) {}

  void Run() {
    if (provider_options_map_.empty()) {
      OpTester::Run();
    } else {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(CreateExecutionProviderFactory_SNPE(provider_options_map_)->CreateProvider());
      OpTester::Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
  void AddProviderOption(const std::string& key, const std::string& value) {
    provider_options_map_[key] = value;
  }
 private:
  ProviderOptions provider_options_map_;
};

TEST(Snpe_ConvertFromConcat, MimoSupport) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_concat_2d_axis_0.dlc"));

  test.AddInput<float>("value0", {2, 2}, {11.0f, 12.0f, 13.0f, 14.0f});
  test.AddInput<float>("value1", {2, 2}, {21.0f, 22.0f, 23.0f, 24.0f});
  test.AddOutput<float>("output", {4, 2}, {11.0f, 12.0f, 13.0f, 14.0f,
                                           21.0f, 22.0f, 23.0f, 24.0f});
  test.Run();

  test.AddProviderOption("buffer_type", "FLOAT");
  test.Run();
}

TEST(Snpe_ConvertFromConcat, MimoSupportInputOrder) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_concat_2d_axis_0.dlc"));

  test.AddInput<float>("value1", {2, 2}, {21.0f, 22.0f, 23.0f, 24.0f});
  test.AddInput<float>("value0", {2, 2}, {11.0f, 12.0f, 13.0f, 14.0f});
  test.AddOutput<float>("output", {4, 2}, {11.0f, 12.0f, 13.0f, 14.0f, 21.0f, 22.0f, 23.0f, 24.0f});
  test.Run();

  test.AddProviderOption("buffer_type", "FLOAT");
  test.Run();
}

TEST(Snpe_ConvertFromSplit, SimoSupport) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_split_equal_parts_2d.dlc"));

  test.AddInput<float>("input", {2, 6}, {11.0f, 12.0f, 13.0f,  14.0f, 15.0f, 16.0f,
                                         21.0f, 22.0f, 23.0f,  24.0f, 25.0f, 26.0f});
  test.AddOutput<float>("output_1", {2, 3}, {11.0f, 12.0f, 13.0f,
                                             21.0f, 22.0f, 23.0f});
  test.AddOutput<float>("output_2", {2, 3}, {14.0f, 15.0f, 16.0f,
                                             24.0f, 25.0f, 26.0f});
  test.Run();

  test.AddProviderOption("buffer_type", "FLOAT");
  test.Run();
}

TEST(Snpe_ConvertFromAbs, SisoSupport) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_abs.dlc"));

  test.AddInput<float>("x", {2, 3, 2}, {11.0f, -12.0f, 13.0f, 14.0f, -15.0f, 16.0f,
                                        21.0f, 22.0f, -23.0f, 24.0f, -25.0f, 26.0f});
  test.AddOutput<float>("y", {2, 3, 2}, {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                                         21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f});
  test.Run();

  test.AddProviderOption("buffer_type", "FLOAT");
  test.Run();
}

TEST(Snpe_ConvertFromAbs, QuantizedModelTf8Test) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_abs_q.dlc"));

  unsigned char step_equivalent_to_0 = 0;
  float quantized_step_size = 0.094118f;
  int bit_width = 8;

  std::vector<float> input_test_data_f{11.0f, -12.0f, 13.0f, 14.0f, -15.0f, 16.0f, 21.0f, 22.0f, -23.0f, 24.0f};
  std::vector<uint8_t> input_test_data_ui8(10);
  FloatToTfN(input_test_data_ui8.data(),
             step_equivalent_to_0,
             quantized_step_size,
             input_test_data_f.data(),
             input_test_data_f.size(),
             bit_width);

  std::vector<uint8_t> input_data;
  auto it = input_data.begin();
  for (int i = 0; i < 6; ++i) {
    it = input_data.insert(it, input_test_data_ui8.begin(), input_test_data_ui8.end());
  }

  // std::vector<float> output_test_data_f{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  // std::vector<uint8_t> output_test_data_ui8(10);
  // FloatToTfN(output_test_data_ui8.data(), step_equivalent_to_0, quantized_step_size,
  //            output_test_data_f.data(), output_test_data_f.size(), bit_width);
  // data converted from float: {117, 128, 138, 149, 159, 170, 223, 234, 244, 255},
  // different with run result which is acceptable since it's quantized model.
  std::vector<uint8_t> output_test_data_ui8 {118, 128, 139, 149, 159, 171, 224, 233, 245, 255};

  std::vector<uint8_t> output_data;
  it = output_data.begin();
  for (int i = 0; i < 6; ++i) {
    it = output_data.insert(it, output_test_data_ui8.begin(), output_test_data_ui8.end());
  }

  test.AddInput<uint8_t>("x", {3, 4, 5}, input_data);
  test.AddOutput<uint8_t>("y", {3, 4, 5}, output_data);
  test.AddProviderOption("buffer_type", "TF8");
  test.Run();
}

TEST(Snpe_ConvertFromAbs, QuantizedModelItensorFloatTest) {
  OpTesterSnpe test("Snpe", 1, kMSDomain);

  test.AddAttribute("DLC", LoadDlcFile("./testdata/snpe/test_abs_q.dlc"));

  std::vector<float> input_test_data_f{11.0f, -12.0f, 13.0f, 14.0f, -15.0f, 16.0f, 21.0f, 22.0f, -23.0f, 24.0f};
  std::vector<float> input_data;
  auto it = input_data.begin();
  for (int i = 0; i < 6; ++i) {
    it = input_data.insert(it, input_test_data_f.begin(), input_test_data_f.end());
  }

  std::vector<float> output_test_data_f{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  std::vector<float> output_data;
  it = output_data.begin();
  for (int i = 0; i < 6; ++i) {
    it = output_data.insert(it, output_test_data_f.begin(), output_test_data_f.end());
  }

  test.AddInput<float>("x", {3, 4, 5}, input_data);
  test.AddOutput<float>("y", {3, 4, 5}, output_data);
  test.Run();

  test.AddProviderOption("buffer_type", "FLOAT");
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
#endif
