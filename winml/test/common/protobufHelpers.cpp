// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif

// LotusRT
#include "core/framework/allocator_utils.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "protobufHelpers.h"

#include <fstream>

using namespace wss;
using namespace wfc;
using namespace winml;

// Copy and pasted from LOTUS as is.    temporary code to load tensors from protobufs
int FdOpen(const std::string& name) {
  int fd = -1;
#ifdef _WIN32
  _sopen_s(&fd, name.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  fd = open(name.c_str(), O_RDONLY);
#endif
  return fd;
};

// Copy and pasted from LOTUS as is.    temporary code to load tensors from protobufs
void FdClose(int fd) {
  if (fd >= 0) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
  }
}

// Load Onnx TensorProto from Protobuf File
bool ProtobufHelpers::LoadOnnxTensorFromProtobufFile(onnx::TensorProto& tensor, std::wstring filePath) {
  // setup a string converter
  using convert_type = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_type, wchar_t> converter;

  // use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
  std::string file = converter.to_bytes(filePath.c_str());

  std::ifstream stream(file, std::ios::binary | std::ios::ate);
  std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  std::vector<char> buffer(static_cast<size_t>(size));
  if (stream.read(buffer.data(), size)) {
    return tensor.ParseFromArray(buffer.data(), static_cast<int>(size));
  } else {
    return false;
  }
}

template <typename DataType>
std::vector<DataType> GetTypeSpecificDataFromTensorProto(onnx::TensorProto /*tensorProto*/) {
  static_assert(false, "UNDEFINED! TensorProto methods aren't templated, so add a new template specialization.");
}
template <>
std::vector<float> GetTypeSpecificDataFromTensorProto(onnx::TensorProto tensorProto) {
  return std::vector<float>(std::begin(tensorProto.float_data()), std::end(tensorProto.float_data()));
}
template <>
std::vector<int32_t> GetTypeSpecificDataFromTensorProto(onnx::TensorProto tensorProto) {
  return std::vector<int32_t>(std::begin(tensorProto.int32_data()), std::end(tensorProto.int32_data()));
}
template <>
std::vector<int64_t> GetTypeSpecificDataFromTensorProto(onnx::TensorProto tensorProto) {
  return std::vector<int64_t>(std::begin(tensorProto.int64_data()), std::end(tensorProto.int64_data()));
}
template <>
std::vector<uint8_t> GetTypeSpecificDataFromTensorProto(onnx::TensorProto tensorProto) {
#pragma warning(push)
#pragma warning(disable : 4244)  // conversion with possible loss of data
  return std::vector<uint8_t>(std::begin(tensorProto.int32_data()), std::end(tensorProto.int32_data()));
#pragma warning(pop)
}
template <>
std::vector<double> GetTypeSpecificDataFromTensorProto(onnx::TensorProto tensorProto) {
  return std::vector<double>(std::begin(tensorProto.double_data()), std::end(tensorProto.double_data()));
}

template <typename DataType>
std::vector<DataType> GetTensorDataFromTensorProto(onnx::TensorProto tensorProto, uint64_t elementCount) {
  if (tensorProto.has_raw_data()) {
    std::vector<DataType> tensorData;
    auto& values = tensorProto.raw_data();
    if (elementCount != values.size() / sizeof(DataType)) {
      throw winrt::hresult_invalid_argument(L"TensorProto element count should match raw data buffer size in elements."
      );
    }

    tensorData = std::vector<DataType>(static_cast<size_t>(elementCount));
    memcpy(tensorData.data(), values.data(), values.size());
    return tensorData;
  } else {
    return GetTypeSpecificDataFromTensorProto<DataType>(tensorProto);
  }
}

static std::vector<winrt::hstring> GetTensorStringDataFromTensorProto(
  onnx::TensorProto tensorProto, uint64_t elementCount
) {
  if (tensorProto.string_data_size() != elementCount) {
    throw winrt::hresult_invalid_argument(L"Number of elements in TensorProto does not match expected element count.");
  }
  auto& values = tensorProto.string_data();
  auto returnVector = std::vector<winrt::hstring>(static_cast<size_t>(elementCount));
  std::transform(std::begin(values), std::end(values), std::begin(returnVector), [](auto& value) {
    return winrt::to_hstring(value);
  });
  return returnVector;
}

ITensor ProtobufHelpers::LoadTensorFromProtobufFile(const std::wstring& filePath, bool isFp16) {
  // load from the file path into the onnx format
  onnx::TensorProto tensorProto;
  if (LoadOnnxTensorFromProtobufFile(tensorProto, filePath)) {
    std::vector<int64_t> tensorShape = std::vector<int64_t>(tensorProto.dims().begin(), tensorProto.dims().end());
    int64_t initialValue = 1;
    int64_t elementCount =
      std::accumulate(tensorShape.begin(), tensorShape.end(), initialValue, std::multiplies<int64_t>());

    if (!tensorProto.has_data_type()) {
      std::cerr << "WARNING: Loading unknown TensorProto datatype.\n";
    }
    if (isFp16) {
      return TensorFloat16Bit::CreateFromIterable(
        tensorShape, GetTensorDataFromTensorProto<float>(tensorProto, elementCount)
      );
    }
    switch (tensorProto.data_type()) {
      case (onnx::TensorProto::DataType::TensorProto_DataType_FLOAT):
        return TensorFloat::CreateFromIterable(
          tensorShape, GetTensorDataFromTensorProto<float>(tensorProto, elementCount)
        );
      case (onnx::TensorProto::DataType::TensorProto_DataType_INT32):
        return TensorInt32Bit::CreateFromIterable(
          tensorShape, GetTensorDataFromTensorProto<int32_t>(tensorProto, elementCount)
        );
      case (onnx::TensorProto::DataType::TensorProto_DataType_INT64):
        return TensorInt64Bit::CreateFromIterable(
          tensorShape, GetTensorDataFromTensorProto<int64_t>(tensorProto, elementCount)
        );
      case (onnx::TensorProto::DataType::TensorProto_DataType_STRING):
        return TensorString::CreateFromIterable(
          tensorShape, GetTensorStringDataFromTensorProto(tensorProto, elementCount)
        );
      case (onnx::TensorProto::DataType::TensorProto_DataType_UINT8):
        return TensorUInt8Bit::CreateFromIterable(
          tensorShape, GetTensorDataFromTensorProto<uint8_t>(tensorProto, elementCount)
        );
      case (onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE):
        return TensorDouble::CreateFromIterable(
          tensorShape, GetTensorDataFromTensorProto<double>(tensorProto, elementCount)
        );
      default:
        throw winrt::hresult_invalid_argument(L"Tensor type for creating tensor from protobuf file not supported.");
        break;
    }
  }
  return nullptr;
}

TensorFloat16Bit ProtobufHelpers::LoadTensorFloat16FromProtobufFile(const std::wstring& filePath) {
  // load from the file path into the onnx format
  onnx::TensorProto tensorProto;
  if (LoadOnnxTensorFromProtobufFile(tensorProto, filePath)) {
    if (tensorProto.has_data_type()) {
      if (onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16 != tensorProto.data_type()) {
        throw winrt::hresult_invalid_argument(L"TensorProto datatype isn't of type Float16.");
      }
    } else {
      std::cerr << "Loading unknown TensorProto datatype as TensorFloat16Bit.\n";
    }

    auto shape =
      winrt::single_threaded_vector<int64_t>(std::vector<int64_t>(tensorProto.dims().begin(), tensorProto.dims().end())
      );
    TensorFloat16Bit singleTensorValue = TensorFloat16Bit::Create(shape.GetView());

    uint16_t* data;
    winrt::com_ptr<ITensorNative> spTensorValueNative;
    singleTensorValue.as(spTensorValueNative);
    uint32_t sizeInBytes;
    spTensorValueNative->GetBuffer(reinterpret_cast<BYTE**>(&data), &sizeInBytes);

    if (!tensorProto.has_raw_data()) {
      throw winrt::hresult_invalid_argument(L"Float16 tensor proto buffers are expected to contain raw data.");
    }

    auto& raw_data = tensorProto.raw_data();
    auto buff = raw_data.c_str();

    memcpy((void*)data, (void*)buff, raw_data.size() * sizeof(char));

    return singleTensorValue;
  }
  return nullptr;
}

winml::LearningModel ProtobufHelpers::CreateModel(
  winml::TensorKind kind, const std::vector<int64_t>& shape, uint32_t num_elements
) {
  onnx::ModelProto model;
  model.set_ir_version(onnx::Version::IR_VERSION);

  // Set opset import
  auto opsetimportproto = model.add_opset_import();
  opsetimportproto->set_version(7);

  onnx::GraphProto& graph = *model.mutable_graph();

  uint32_t begin = 0;
  uint32_t end = num_elements - 1;
  for (uint32_t i = begin; i <= end; i++) {
    onnx::NodeProto& node = *graph.add_node();
    node.set_op_type("Identity");
    if (i == begin && i == end) {
      node.add_input("input");
      node.add_output("output");
    } else if (i == begin) {
      node.add_input("input");
      node.add_output("output" + std::to_string(i));

    } else if (i == end) {
      node.add_input("output" + std::to_string(i - 1));
      node.add_output("output");
    } else {
      node.add_input("output" + std::to_string(i - 1));
      node.add_output("output" + std::to_string(i));
    }
  }

  onnx::TensorProto_DataType dataType;
  switch (kind) {
    case TensorKind::Float:
      dataType = onnx::TensorProto_DataType_FLOAT;
      break;
    case TensorKind::UInt8:
      dataType = onnx::TensorProto_DataType_UINT8;
      break;
    case TensorKind::Int8:
      dataType = onnx::TensorProto_DataType_INT8;
      break;
    case TensorKind::UInt16:
      dataType = onnx::TensorProto_DataType_UINT16;
      break;
    case TensorKind::Int16:
      dataType = onnx::TensorProto_DataType_INT16;
      break;
    case TensorKind::Int32:
      dataType = onnx::TensorProto_DataType_INT32;
      break;
    case TensorKind::Int64:
      dataType = onnx::TensorProto_DataType_INT64;
      break;
    case TensorKind::String:
      dataType = onnx::TensorProto_DataType_STRING;
      break;
    case TensorKind::Boolean:
      dataType = onnx::TensorProto_DataType_BOOL;
      break;
    case TensorKind::Float16:
      dataType = onnx::TensorProto_DataType_FLOAT16;
      break;
    case TensorKind::Double:
      dataType = onnx::TensorProto_DataType_DOUBLE;
      break;
    case TensorKind::UInt32:
      dataType = onnx::TensorProto_DataType_UINT32;
      break;
    case TensorKind::UInt64:
      dataType = onnx::TensorProto_DataType_UINT64;
      break;
    default:
      return nullptr;
  }

  char dim_param = 'a';
  // input
  {
    onnx::ValueInfoProto& variable = *graph.add_input();
    variable.set_name("input");
    variable.mutable_type()->mutable_tensor_type()->set_elem_type(dataType);
    for (auto dim : shape) {
      if (dim == -1) {
        variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(&dim_param, 1);
        dim_param++;
      } else {
        variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
      }
    }

    if (shape.size() > 0) {
      variable.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_denotation("DATA_BATCH");
    }
  }

  // output
  {
    onnx::ValueInfoProto& variable = *graph.add_output();
    variable.set_name("output");
    variable.mutable_type()->mutable_tensor_type()->set_elem_type(dataType);
    for (auto dim : shape) {
      if (dim == -1) {
        variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(&dim_param, 1);
        dim_param++;
      } else {
        variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
      }
    }
  }

  struct BufferStreamAdapter : public std::streambuf {
    RandomAccessStreamReference BufferAsRandomAccessStreamReference() {
      auto buffer = m_dataWriter.DetachBuffer();
      m_dataWriter = DataWriter();

      InMemoryRandomAccessStream stream;
      stream.WriteAsync(buffer).get();
      return RandomAccessStreamReference::CreateFromStream(stream);
    }

   protected:
    virtual int_type overflow(int_type c) {
      if (c != EOF) {
        // convert lowercase to uppercase
        auto temp = static_cast<char>(c);

        m_dataWriter.WriteByte(temp);
      }
      return c;
    }

   private:
    DataWriter m_dataWriter;
  };

  auto size = model.ByteSizeLong();
  auto raw_array = std::unique_ptr<char[]>(new char[size]);
  model.SerializeToArray(raw_array.get(), static_cast<int>(size));

  BufferStreamAdapter buffer;
  std::ostream os(&buffer);

  os.write(raw_array.get(), size);

  return LearningModel::LoadFromStream(buffer.BufferAsRandomAccessStreamReference());
}
