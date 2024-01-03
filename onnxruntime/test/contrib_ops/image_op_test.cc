// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/common/common.h"
#include "contrib_ops/cpu/crop.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/common/cuda_op_test_utils.h"

#include "test/shared_lib/test_fixture.h"
#include "test/util/include/asserts.h"

#include <fstream>
#include <memory>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::test;
namespace onnxruntime {
namespace test {

using ExpectResult = OpTester::ExpectResult;

// #if defined(USE_CUDA)
static const std::string IMAGE_URI_JPG = "testdata/images/img1.jpg";
static const std::string IMAGE_URI_RGB_RAW = "testdata/images/img1_rgb_3_640_480.raw";
static const std::string IMAGE_URI_BGR_RAW = "testdata/images/img1_bgr_3_640_480.raw";
static const std::string IMAGE_URI_Y_RAW = "testdata/images/img1_y_1_640_480.raw";
static const std::string IMAGE_URI_ENCODED_FROM_RGB = "testdata/images/encoder_test_rgb_encoded.jpg";
static const std::string IMAGE_URI_ENCODED_FROM_BGR = "testdata/images/encoder_test_bgr_encoded.jpg";
static const std::string IMAGE_URI_ENCODED_FROM_GRAYSCALE = "testdata/images/encoder_test_grayscale_encoded.jpg";

Status ReadImageData(const std::string& file_path, std::vector<uint8_t>& image_data, int64_t* width = nullptr, int64_t* height = nullptr, int64_t* channels = nullptr) {
  // read image data from file_path.
  // if file_path extension is .raw, then parse the filename to get image channel, width, and height.
  // return image data, and width, height, channels if parsed from filename.
  std::ifstream input_stream(file_path, std::ios::in | std::ios::binary | std::ios::ate);
  ORT_RETURN_IF_NOT(input_stream.is_open(), "Cannot open image: ", file_path);

  // Get the size.
  std::streamsize image_data_size = input_stream.tellg();
  input_stream.seekg(0, std::ios::beg);
  image_data.resize(image_data_size);

  if (input_stream.read(reinterpret_cast<char*>(image_data.data()), image_data_size)) {
    std::string filename = file_path.substr(file_path.find_last_of("/\\") + 1);
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    if (extension == "raw" && width && height && channels) {
      // Parse the filename to get width, height, and channels.
      std::string name = filename.substr(0, filename.find_last_of("."));
      std::vector<std::string> tokens;
      size_t pos = 0;
      std::string delimiter = "_";
      while ((pos = name.find(delimiter)) != std::string::npos) {
        tokens.push_back(name.substr(0, pos));
        name.erase(0, pos + delimiter.length());
      }
      if (!name.empty())
        tokens.push_back(name.substr(0, pos));

      ORT_RETURN_IF_NOT(tokens.size() >= 3, "can not parse filename to get image dimensions. ", filename);
      *channels = std::stoi(tokens[tokens.size() - 3]);
      *height = std::stoi(tokens[tokens.size() - 2]);
      *width = std::stoi(tokens[tokens.size() - 1]);

      ORT_RETURN_IF_NOT(image_data_size == (*channels) * (*width) * (*height), "image dimensions do not match image data size: ", file_path);
    }
  }
  return Status::OK();
}

TEST(ImageEncoderDecoderOpTest, EncodeRGB) {
  OpTester test("ImageEncoder", 1, kMSDomain);
  test.AddAttribute("pixel_format", "RGB");

  std::vector<uint8_t> input_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_RGB_RAW, input_image_data, &width, &height, &channels));
  std::vector<uint8_t> output_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_ENCODED_FROM_RGB, output_image_data));

  test.AddInput<uint8_t>("input", {1, 3, height, width}, input_image_data);
  test.AddOutput<uint8_t>("encoded_stream", {static_cast<int64_t>(output_image_data.size())}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ImageEncoderDecoderOpTest, EncodeBGR) {
  OpTester test("ImageEncoder", 1, kMSDomain);
  test.AddAttribute("pixel_format", "BGR");

  std::vector<uint8_t> input_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_BGR_RAW, input_image_data, &width, &height, &channels));
  std::vector<uint8_t> output_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_ENCODED_FROM_BGR, output_image_data));

  test.AddInput<uint8_t>("input", {1, 3, height, width}, input_image_data);
  test.AddOutput<uint8_t>("encoded_stream", {static_cast<int64_t>(output_image_data.size())}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ImageEncoderDecoderOpTest, EncodeGrayscale) {
  OpTester test("ImageEncoder", 1, kMSDomain);
  test.AddAttribute("pixel_format", "Grayscale");
  test.AddAttribute("subsampling", "400");

  std::vector<uint8_t> input_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_Y_RAW, input_image_data, &width, &height, &channels));
  std::vector<uint8_t> output_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_ENCODED_FROM_GRAYSCALE, output_image_data));

  test.AddInput<uint8_t>("input", {1, 1, height, width}, input_image_data);
  test.AddOutput<uint8_t>("encoded_stream", {static_cast<int64_t>(output_image_data.size())}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ImageEncoderDecoderOpTest, DecodeRGB) {
  OpTester test("ImageDecoder", 20, kOnnxDomain);
  test.AddAttribute("pixel_format", "RGB");

  std::vector<uint8_t> input_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_JPG, input_image_data));
  std::vector<uint8_t> output_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_RGB_RAW, output_image_data, &width, &height, &channels));

  test.AddInput<uint8_t>("encoded_stream", {static_cast<int64_t>(input_image_data.size()),}, input_image_data);
  test.AddOutput<uint8_t>("output", {channels, height, width}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
TEST(ImageEncoderDecoderOpTest, DecodeBGR) {
  OpTester test("ImageDecoder", 20, kOnnxDomain);
  test.AddAttribute("pixel_format", "BGR");

  std::vector<uint8_t> input_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_JPG, input_image_data));
  std::vector<uint8_t> output_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_BGR_RAW, output_image_data, &width, &height, &channels));

  test.AddInput<uint8_t>("encoded_stream", {static_cast<int64_t>(input_image_data.size()),}, input_image_data);
  test.AddOutput<uint8_t>("output", {channels, height, width}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ImageEncoderDecoderOpTest, DecodeGrayscale) {
  OpTester test("ImageDecoder", 20, kOnnxDomain);
  test.AddAttribute("pixel_format", "Grayscale");

  std::vector<uint8_t> input_image_data;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_JPG, input_image_data));
  std::vector<uint8_t> output_image_data;
  int64_t width, height, channels;
  EXPECT_STATUS_OK(ReadImageData(IMAGE_URI_Y_RAW, output_image_data, &width, &height, &channels));

  test.AddInput<uint8_t>("encoded_stream", {static_cast<int64_t>(input_image_data.size()),}, input_image_data);
  test.AddOutput<uint8_t>("output", {channels, height, width}, output_image_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  test.Run(ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
// #endif // USE_CUDA
}  // namespace test
}  // namespace onnxruntime
