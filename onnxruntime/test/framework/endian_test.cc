#include "core/framework/endian.h"
#include "core/framework/endian_utils.h"
#include "core/graph/onnx_protobuf.h"         // For TensorProto
#include "core/framework/tensorprotoutils.h"  // For ConvertRawDataInTensorProto

#include <vector>
#include <cstddef>  // For std::byte

#include "gtest/gtest.h"

namespace onnxruntime {
namespace utils {
namespace test {

TEST(EndianTest, EndiannessDetection) {
  constexpr uint16_t test_value = 0x1234;
  const unsigned char* test_value_first_byte = reinterpret_cast<const unsigned char*>(&test_value);
  if constexpr (endian::native == endian::little) {
    EXPECT_EQ(*test_value_first_byte, 0x34);
  } else if constexpr (endian::native == endian::big) {
    EXPECT_EQ(*test_value_first_byte, 0x12);
  }
}

TEST(EndianTest, SwapByteOrderCopy) {
  const auto src = std::vector<unsigned char>{
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};

  auto result = std::vector<unsigned char>(src.size());
  {
    SwapByteOrderCopy(3, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<unsigned char>{
        'c', 'b', 'a',
        'f', 'e', 'd',
        'i', 'h', 'g',
        'l', 'k', 'j'};
    EXPECT_EQ(result, expected);
  }

  {
    SwapByteOrderCopy(4, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<unsigned char>{
        'd', 'c', 'b', 'a',
        'h', 'g', 'f', 'e',
        'l', 'k', 'j', 'i'};
    EXPECT_EQ(result, expected);
  }
}

// Test fixture for SwapByteOrderInplace tests
class SwapByteOrderInplaceTest : public ::testing::Test {};

TEST_F(SwapByteOrderInplaceTest, ElementSize1) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};
  std::vector<std::byte> expected_data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(1, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize2_SingleElement) {
  std::vector<std::byte> data = {std::byte{0x01}, std::byte{0x02}};
  std::vector<std::byte> expected_data = {std::byte{0x02}, std::byte{0x01}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize2_MultipleElements) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}, std::byte{0x05}, std::byte{0x06}};
  std::vector<std::byte> expected_data = {
      std::byte{0x02}, std::byte{0x01}, std::byte{0x04}, std::byte{0x03}, std::byte{0x06}, std::byte{0x05}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize4_SingleElement) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};
  std::vector<std::byte> expected_data = {
      std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize4_MultipleElements) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};
  std::vector<std::byte> expected_data = {
      std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01},
      std::byte{0x08}, std::byte{0x07}, std::byte{0x06}, std::byte{0x05}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize8_SingleElement) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};
  std::vector<std::byte> expected_data = {
      std::byte{0x08}, std::byte{0x07}, std::byte{0x06}, std::byte{0x05},
      std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(8, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize8_MultipleElements) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08},
      std::byte{0x11}, std::byte{0x12}, std::byte{0x13}, std::byte{0x14},
      std::byte{0x15}, std::byte{0x16}, std::byte{0x17}, std::byte{0x18}};
  std::vector<std::byte> expected_data = {
      std::byte{0x08}, std::byte{0x07}, std::byte{0x06}, std::byte{0x05},
      std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01},
      std::byte{0x18}, std::byte{0x17}, std::byte{0x16}, std::byte{0x15},
      std::byte{0x14}, std::byte{0x13}, std::byte{0x12}, std::byte{0x11}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(8, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, EmptyBuffer) {
  std::vector<std::byte> data = {};
  std::vector<std::byte> expected_data = {};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  // Should not crash or throw for valid element sizes, e.g., 2 or 4
  // The ORT_ENFORCE checks will pass as 0 % element_size == 0
  // The loop for swapping will not execute.
  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize3_OddElementSize) {
  std::vector<std::byte> data = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03},
      std::byte{0x04}, std::byte{0x05}, std::byte{0x06}};
  std::vector<std::byte> expected_data = {
      std::byte{0x03}, std::byte{0x02}, std::byte{0x01},
      std::byte{0x06}, std::byte{0x05}, std::byte{0x04}};
  gsl::span<std::byte> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(3, data_span);
  EXPECT_EQ(data, expected_data);
}

// Test fixture for ConvertRawDataInTensorProto tests
class ConvertRawDataInTensorProtoTest : public ::testing::Test {
 protected:
  // Helper function to set up a TensorProto with float data
  void SetupFloatTensor(ONNX_NAMESPACE::TensorProto& tensor, const std::vector<float>& values) {
    tensor.Clear();
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (float value : values) {
      tensor.add_float_data(value);
    }
  }

  // Helper function to set up a TensorProto with int32 data
  void SetupInt32Tensor(ONNX_NAMESPACE::TensorProto& tensor, const std::vector<int32_t>& values) {
    tensor.Clear();
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    for (int32_t value : values) {
      tensor.add_int32_data(value);
    }
  }

  // Helper function to set up a TensorProto with int16 data (stored in int32 container)
  void SetupInt16Tensor(ONNX_NAMESPACE::TensorProto& tensor, const std::vector<int16_t>& values) {
    tensor.Clear();
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT16);
    for (int16_t value : values) {
      tensor.add_int32_data(value);
    }
  }

  // Helper function to set up a TensorProto with raw data
  template <typename T>
  void SetupRawDataTensor(ONNX_NAMESPACE::TensorProto& tensor, ONNX_NAMESPACE::TensorProto_DataType data_type,
                          const std::vector<T>& values) {
    tensor.Clear();
    tensor.set_data_type(data_type);
    tensor.set_raw_data(values.data(), values.size() * sizeof(T));
  }

  // Helper to compare float data before and after conversion
  void CompareFloatData(const ONNX_NAMESPACE::TensorProto& tensor, const std::vector<float>& expected_values) {
    ASSERT_EQ(tensor.float_data_size(), static_cast<int>(expected_values.size()));
    for (int i = 0; i < tensor.float_data_size(); i++) {
      // We swap bytes so the actual value might change if we're converting endianness
      // But a double swap should restore the original value
      if constexpr (endian::native == endian::little) {
        EXPECT_EQ(tensor.float_data(i), expected_values[i]);
      } else {
        // Just verify the value is different after one swap on big-endian
        // We can't predict the exact value without manual byte swapping
        if (expected_values[i] != 0) {  // Skip zero values as they're invariant to byte swapping
          EXPECT_EQ(tensor.float_data(i), expected_values[i]);
        }
      }
    }
  }

  // Helper to compare int32 data before and after conversion
  void CompareInt32Data(const ONNX_NAMESPACE::TensorProto& tensor, const std::vector<int32_t>& expected_values) {
    ASSERT_EQ(tensor.int32_data_size(), static_cast<int>(expected_values.size()));
    for (int i = 0; i < tensor.int32_data_size(); i++) {
      // Same logic as float comparison
      if constexpr (endian::native == endian::little) {
        EXPECT_EQ(tensor.int32_data(i), expected_values[i]);
      } else {
        if (expected_values[i] != 0) {
          EXPECT_EQ(tensor.int32_data(i), expected_values[i]);
        }
      }
    }
  }
};

TEST_F(ConvertRawDataInTensorProtoTest, FloatData) {
  ONNX_NAMESPACE::TensorProto tensor;
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
  SetupFloatTensor(tensor, values);

  // Save original values
  std::vector<float> original_values;
  for (int i = 0; i < tensor.float_data_size(); i++) {
    original_values.push_back(tensor.float_data(i));
  }

  // Convert once
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Convert back - should restore original values
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer
  CompareFloatData(tensor, original_values);
}

TEST_F(ConvertRawDataInTensorProtoTest, Int32Data) {
  ONNX_NAMESPACE::TensorProto tensor;
  std::vector<int32_t> values = {1, 2, 3, 4};
  SetupInt32Tensor(tensor, values);

  // Save original values
  std::vector<int32_t> original_values;
  for (int i = 0; i < tensor.int32_data_size(); i++) {
    original_values.push_back(tensor.int32_data(i));
  }

  // Convert once
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Convert back - should restore original values
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer
  CompareInt32Data(tensor, original_values);
}

TEST_F(ConvertRawDataInTensorProtoTest, Int16Data) {
  ONNX_NAMESPACE::TensorProto tensor;
  std::vector<int16_t> values = {1, 2, 3, 4};
  SetupInt16Tensor(tensor, values);

  // Save original values
  std::vector<int32_t> original_values;
  for (int i = 0; i < tensor.int32_data_size(); i++) {
    original_values.push_back(tensor.int32_data(i));
  }

  // Convert once
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Convert back - should restore original values
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // When we swap bytes on int16 values stored in int32 containers, the test should pass
  // on both little-endian and big-endian systems
  ASSERT_EQ(tensor.int32_data_size(), static_cast<int>(original_values.size()));
  for (int i = 0; i < tensor.int32_data_size(); i++) {
    EXPECT_EQ(tensor.int32_data(i), original_values[i]);
  }
}

TEST_F(ConvertRawDataInTensorProtoTest, RawFloatData) {
  ONNX_NAMESPACE::TensorProto tensor;
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
  SetupRawDataTensor(tensor, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, values);

  // Save original raw data
  std::string original_raw_data = tensor.raw_data();

  // Convert once
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Convert back - should restore original bytes
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  EXPECT_EQ(tensor.raw_data(), original_raw_data);
}

TEST_F(ConvertRawDataInTensorProtoTest, UInt8NoConversion) {
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  std::vector<uint8_t> values = {1, 2, 3, 4};
  for (auto val : values) {
    tensor.add_int32_data(val);
  }

  // Save original data
  std::vector<int32_t> original_values;
  for (int i = 0; i < tensor.int32_data_size(); i++) {
    original_values.push_back(tensor.int32_data(i));
  }

  // Convert - for 1-byte elements, no conversion should happen
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Verify no change occurred
  ASSERT_EQ(tensor.int32_data_size(), static_cast<int>(original_values.size()));
  for (int i = 0; i < tensor.int32_data_size(); i++) {
    EXPECT_EQ(tensor.int32_data(i), original_values[i]);
  }
}

TEST_F(ConvertRawDataInTensorProtoTest, DoubleConversionAndRestore) {
  // Test with double values
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  std::vector<double> values = {1.1, 2.2, 3.3, 4.4};
  for (auto val : values) {
    tensor.add_double_data(val);
  }

  // Save original data
  std::vector<double> original_values;
  for (int i = 0; i < tensor.double_data_size(); i++) {
    original_values.push_back(tensor.double_data(i));
  }

  // Convert once
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Convert again - this should restore original values
  onnxruntime::utils::ConvertRawDataInTensorProto(tensor);  // Pass by reference, not pointer

  // Verify restored values
  ASSERT_EQ(tensor.double_data_size(), static_cast<int>(original_values.size()));
  for (int i = 0; i < tensor.double_data_size(); i++) {
    EXPECT_EQ(tensor.double_data(i), original_values[i]);
  }
}

}  // namespace test
}  // namespace utils
}  // namespace onnxruntime
