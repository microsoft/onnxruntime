// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/path.h"
#include "core/graph/graph_flatbuffers_utils.h"

#include "test/ir/flatbuffers_utils_test_generated.h"

#include "test/util/include/asserts.h"

namespace onnxruntime {
using namespace fbs::utils;
namespace test {

namespace {
void CreateWriter(const std::string& filename, std::ofstream& external_data_stream, ExternalDataWriter& writer) {
  external_data_stream = std::ofstream(filename, std::ios::binary);

  ASSERT_FALSE(external_data_stream.fail()) << "Failed to create data file";

  // setup the data writer to write aligned data to external_data_stream
  // NOTE: this copies the logic from \orttraining\orttraining\training_api\checkpoint.cc to indirectly test that.
  writer = [&external_data_stream](int32_t data_type, gsl::span<const uint8_t> bytes, uint64_t& offset) {
    // for now align everything to 4 or 8 bytes. we can optimize this later if needed.
    int32_t alignment = 4;

    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
        data_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
      alignment = 8;
    }

    int64_t pos = external_data_stream.tellp();

    if (pos % alignment != 0) {
      // 8 bytes of 0's so we can pad to alignment 8 in a single `write`
      constexpr static const uint64_t zeros = 0;
      int64_t padding = alignment - (pos % alignment);
      // skipping validation of this write. doesn't matter if this or the 'real' write below fails. if this does the
      // other will as well as nothing will clear the failure bit in the ofstream in between the calls.
      external_data_stream.write(reinterpret_cast<const char*>(&zeros), padding);
      pos += padding;
    }

    external_data_stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    ORT_RETURN_IF(external_data_stream.fail(), "Failed writing external checkpoint data.");

    ORT_ENFORCE(pos + int64_t(bytes.size()) == external_data_stream.tellp());  // sanity check

    offset = pos;
    return Status::OK();
  };
}

void CreateReader(const std::string& filename, std::ifstream& external_data_stream, ExternalDataReader& reader) {
  external_data_stream = std::ifstream(filename, std::ios::binary);

  ASSERT_FALSE(external_data_stream.fail()) << "Failed to open data file.";

  reader = [&external_data_stream](uint64_t offset, gsl::span<uint8_t> output_buffer) {
    external_data_stream.seekg(offset);
    external_data_stream.read(reinterpret_cast<char*>(output_buffer.data()), output_buffer.size());

    ORT_RETURN_IF(external_data_stream.fail(),
                  "Failed to read external checkpoint data. Offset:", offset, " Bytes:", output_buffer.size());

    return Status::OK();
  };
}

template <typename T>
ONNX_NAMESPACE::TensorProto CreateInitializer(const std::string& name,
                                              ONNX_NAMESPACE::TensorProto_DataType data_type,
                                              const std::vector<int64_t>& dims,
                                              bool use_raw_data = false) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(data_type);

  int64_t num_elements = 1;
  for (auto dim : dims) {
    tp.add_dims(dim);
    num_elements *= dim;
  }

  std::vector<T> data(num_elements);
  std::iota(data.begin(), data.end(), T(1));  // fill with 1..num_elements

  if (use_raw_data) {
    tp.set_raw_data(data.data(), data.size() * sizeof(T));
  } else {
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        for (auto val : data) {
          tp.add_int64_data(val);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (auto val : data) {
          tp.add_float_data(val);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        for (auto val : data) {
          tp.add_int32_data(val);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        for (auto val : data) {
          tp.add_int32_data(val);
        }
        break;
      }
      default:
        ORT_THROW("Unsupported data type: ", data_type);
    }
  }

  return tp;
}

template <>
ONNX_NAMESPACE::TensorProto CreateInitializer<std::string>(const std::string& name,
                                                           ONNX_NAMESPACE::TensorProto_DataType data_type,
                                                           const std::vector<int64_t>& dims,
                                                           bool /*use_raw_data*/) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(data_type);

  int64_t num_elements = 1;
  for (auto dim : dims) {
    tp.add_dims(dim);
    num_elements *= dim;
  }

  for (int i = 0; i < num_elements; ++i) {
    tp.add_string_data("string_" + std::to_string(i));
  }

  return tp;
}

std::vector<ONNX_NAMESPACE::TensorProto> CreateInitializers() {
  std::vector<ONNX_NAMESPACE::TensorProto> initializers;
  // create data of various sizes. order is chosen to require padding between most but not all
  // assuming our writer aligns to 4 bytes unless it's 64-bit data (which is aligned to 8 bytes)
  // buffer: <16-bit><pad 2><32-bit><8-bit><pad 7><64-bit>
  // need 128 bytes to write to external data

  // 16-bit. 81 elements so we're 2 bytes past 4 byte alignment
  initializers.emplace_back(
      CreateInitializer<int16_t>("tensor_16", ONNX_NAMESPACE::TensorProto_DataType_INT16, {9, 9}));

  // string (should not use external)
  initializers.emplace_back(
      CreateInitializer<std::string>("tensor_string", ONNX_NAMESPACE::TensorProto_DataType_STRING, {2, 2}));

  // 32-bit
  initializers.emplace_back(
      CreateInitializer<float>("tensor_f32", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {8, 8}));

  // 8-bit. 129 elements so we're 1 byte past 4 or 8 byte alignment
  initializers.emplace_back(
      CreateInitializer<uint8_t>("tensor_8", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {3, 43}));

  // 64-bit
  initializers.emplace_back(
      CreateInitializer<int64_t>("tensor_64", ONNX_NAMESPACE::TensorProto_DataType_INT64, {6, 6}));

  // small (should not use external)
  initializers.emplace_back(
      CreateInitializer<int32_t>("tensor_32_small", ONNX_NAMESPACE::TensorProto_DataType_INT32, {2, 2}));

  return initializers;
}

}  // namespace

TEST(GraphUtilsTest, ExternalWriteReadBasic) {
  // create data
  auto initializers = CreateInitializers();

  flatbuffers::FlatBufferBuilder builder(1024);

  // write
  std::ofstream output_stream;
  ExternalDataWriter writer;
  CreateWriter("ExternalWriteReadBasicTest.bin", output_stream, writer);

  std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors;
  for (const auto& initializer : initializers) {
    flatbuffers::Offset<fbs::Tensor> fbs_tensor;
    ASSERT_STATUS_OK(SaveInitializerOrtFormat(builder, initializer, Path(), fbs_tensor, writer));
  }

  // TODO: This could more accurately check the expected data size
  ASSERT_GT(output_stream.tellp(), 0) << "No data written to external file.";
  output_stream.close();
  ASSERT_TRUE(output_stream.good()) << "Failed to close data file.";

  auto fbs_tensors_offset = builder.CreateVector(fbs_tensors);
  fbs::test::TestDataBuilder tdb(builder);
  tdb.add_initializers(fbs_tensors_offset);
  builder.Finish(tdb.Finish());
  auto fb_data = builder.GetBufferSpan();

  auto test_data = fbs::test::GetTestData(fb_data.data());
  auto fbs_tensors2 = test_data->initializers();

  // read
  std::ifstream input_stream;
  ExternalDataReader reader;
  CreateReader("ExternalWriteReadBasicTest.bin", input_stream, reader);

  std::vector<ONNX_NAMESPACE::TensorProto> loaded_initializers;
  OrtFormatLoadOptions options;

  for (const auto* fbs_tensor : *fbs_tensors2) {
    ONNX_NAMESPACE::TensorProto initializer;
    ASSERT_STATUS_OK(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader));
    loaded_initializers.emplace_back(std::move(initializer));
  }

  //
  // TODO: Validate the loaded data!
  //
  bool data_validated = false;
  ASSERT_TRUE(data_validated);
}
}  // namespace test
}  // namespace onnxruntime
