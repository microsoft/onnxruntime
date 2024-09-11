// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"

#include "core/graph/onnx_protobuf.h"

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"

#include "lora/adapter_format_version.h"
#include "lora/adapter_format_utils.h"
#include "lora/lora_adapters.h"

#include <string.h>

#include <fstream>

namespace onnxruntime {
namespace python {

namespace py = pybind11;
namespace {

// Check if the numpy dtype descr property has any of the known types
// that is not supported natively by numpy arrays.
// For example:
// >>> bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
// >>> print(bfloat16.descr)
//    [('bfloat16', '<u2')]
// descr property has an array interface. We query the zero element and
// get a tuple, then query the 0th element of the tuple and check if it is among
// any of the ONNX types that are unsupported natively by numpy.
// If so we adjust the base type such as uint16_t to blfoat16
// See https://github.com/onnx/onnx/blob/main/onnx/_custom_element_types.py
std::optional<std::string> GetDescrPropertyString(const py::dtype& arr_dtype) {
  std::optional<std::string> custom_type;
  try {
    if (py::hasattr(arr_dtype, "descr")) {
      auto descr = py::getattr(arr_dtype, "descr").cast<py::array>();
      if (descr.size() > 0) {
        auto item = descr[0].cast<py::tuple>();
        if (item.size() > 0) {
          custom_type = item[0].cast<std::string>();
        }
      }
    }
  } catch (const py::cast_error&) {
    // Ignore the exception
    PyErr_Clear();
  }
  return custom_type;
}

// bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
py::dtype ConstructCustomDtype(int32_t npy_type, const std::string& custom_type_tag) {
  py::dtype first_arg(npy_type);

  py::dict second_arg;
  second_arg[py::str(custom_type_tag)] = py::make_tuple(first_arg, 0);
  auto tuple = py::make_tuple(std::move(first_arg), std::move(second_arg));

  py::dtype result{py::dtype::from_args(tuple)};
  return result;
}

// Get mapped OnnxDataType from numpy dtype descriptior
// float4e2m1 unsupported at the moment
std::optional<int32_t> GetOnnxDataTypeFromCustomPythonDescr(const std::string& descr) {
  static const std::unordered_map<std::string, int32_t> dtype_descr = {
      {"bfloat16", ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16},
      {"e4m3fn", ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN},
      {"e4m3fnuz", ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ},
      {"e5m2", ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2},
      {"e5m2fnuz", ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ},
      {"int4", ONNX_NAMESPACE::TensorProto_DataType_INT4},
      {"uint4", ONNX_NAMESPACE::TensorProto_DataType_UINT4},
  };

  auto hit = dtype_descr.find(descr);
  if (hit == dtype_descr.end()) {
    return std::nullopt;
  }

  return hit->second;
}

// If a custom type is discovered in numpy array we set the correct ONNX type.
int32_t AdjustOnnxTypeIfNeeded(const py::dtype& arr_dtype, int32_t base_type_from_array) {
  auto descr = GetDescrPropertyString(arr_dtype);
  if (descr.has_value()) {
    auto adjusted_type = GetOnnxDataTypeFromCustomPythonDescr(*descr);
    if (adjusted_type.has_value()) {
      return *adjusted_type;
    }
  }
  return base_type_from_array;
}

std::optional<int> FromOnnxTypeToNumpySupportedType(int32_t onnx_type) {
  // Numpy supported types mapping
  static std::unordered_map<int32_t, int> onnxtype_to_numpy{
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, NPY_BOOL},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, NPY_FLOAT},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, NPY_FLOAT16},
      {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, NPY_DOUBLE},
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, NPY_INT8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, NPY_UINT8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, NPY_INT16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, NPY_UINT16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, NPY_INT},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, NPY_UINT},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, NPY_LONGLONG},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, NPY_ULONGLONG},
      {ONNX_NAMESPACE::TensorProto_DataType_STRING, NPY_STRING},
  };

  auto hit = onnxtype_to_numpy.find(onnx_type);
  if (hit == onnxtype_to_numpy.end())
    return std::nullopt;

  return hit->second;
}

std::optional<std::pair<int, std::string>> GetCustomNumpyTypeFromOnnxType(int32_t onnx_data_type) {
  static const std::unordered_map<int32_t, std::pair<int, std::string>> onnxtype_to_custom_numpy_type = {
      {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, {NPY_UINT16, "bfloat16"}},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, {NPY_UINT8, "e4m3fn"}},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ, {NPY_UINT8, "e4m3fnuz"}},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, {NPY_UINT8, "e5m2"}},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ, {NPY_UINT8, "e5m2fnuz"}},
      {ONNX_NAMESPACE::TensorProto_DataType_INT4, {NPY_INT8, "int4"}},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT4, {NPY_UINT8, "uint4"}}};

  auto hit = onnxtype_to_custom_numpy_type.find(onnx_data_type);
  if (hit == onnxtype_to_custom_numpy_type.end()) {
    return std::nullopt;
  }

  return hit->second;
}

py::dtype ConstructDType(int32_t onnx_type) {
  // check if the type maps to onnx custom type
  auto custom_type = GetCustomNumpyTypeFromOnnxType(onnx_type);
  if (custom_type.has_value()) {
    return ConstructCustomDtype(custom_type->first, custom_type->second);
  }

  auto npy_type = FromOnnxTypeToNumpySupportedType(onnx_type);
  if (npy_type.has_value()) {
    return py::dtype(*npy_type);
  }
  ORT_THROW("Unsupported type detected:", onnx_type);
}

/// <summary>
/// Class that supports writing and reading adapters
/// in innxruntime format
/// </summary>
struct PyAdapterFormatReaderWriter {
  PyAdapterFormatReaderWriter() = default;
  PyAdapterFormatReaderWriter(int format_version, int adapter_version,
                              int model_version,
                              lora::LoraAdapter&& loaded_adapter,
                              py::dict&& params)
      : format_version_(format_version),
        adapter_version_(adapter_version),
        model_version_(model_version),
        loaded_adater_(std::move(loaded_adapter)),
        parameters_(std::move(params)) {}

  int format_version_{adapters::kAdapterFormatVersion};
  int adapter_version_{0};
  int model_version_{0};
  // This container is used when reading the the file so
  // OrtValue objects can be backed by it. Not exposed to Python
  std::optional<lora::LoraAdapter> loaded_adater_;
  // This is a dictionary of string -> OrtValue
  // this is populated directly on write and
  // built on top of the loaded_adapter on read
  py::dict parameters_;
};

}  // namespace

/* */
void addAdapterFormatMethods(pybind11::module& m) {
  py::class_<PyAdapterFormatReaderWriter> adapter_binding(m, "Adapter");
  adapter_binding.def(py::init())
      .def_property_readonly(
          "format_version",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> int { return reader_writer->format_version_; },
          R"pbdoc("Enables user to read format version stored in the file")pbdoc")
      .def_property(
          "adapter_version",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> int { return reader_writer->adapter_version_; },
          [](PyAdapterFormatReaderWriter* reader_writer, int adapter_version) -> void { reader_writer->adapter_version_ = adapter_version; },
          R"pbdoc("Enables user to read format version stored in the file")pbdoc")
      .def_property(
          "adapter_version",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> int { return reader_writer->adapter_version_; },
          [](PyAdapterFormatReaderWriter* reader_writer, int adapter_version) -> void { reader_writer->adapter_version_ = adapter_version; },
          R"pbdoc("Enables user to read/write adapter version stored in the file")pbdoc")
      .def_property(
          "model_version",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> int { return reader_writer->model_version_; },
          [](PyAdapterFormatReaderWriter* reader_writer, int model_version) -> void { reader_writer->model_version_ = model_version; },
          R"pbdoc("Enables user to read/write model version this adapter was created for")pbdoc")
      .def_property(
          "parameters",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> py::dict { return reader_writer->parameters_; },
          [](PyAdapterFormatReaderWriter* reader_writer, py::dict& parameters) -> void {
            reader_writer->parameters_ = parameters;
          },
          R"pbdoc("Enables user to read/write adapter version stored in the file")pbdoc")
      .def(
          "export_adapter",
          [](const PyAdapterFormatReaderWriter* reader_writer, const std::string& file_name) {
            std::ofstream file(file_name, std::ios::binary);
            if (file.fail()) {
              ORT_THROW("Failed to open file:", file_name, " for writing.");
            }

            adapters::utils::AdapterFormatBuilder format_builder;
            for (auto& [n, value] : reader_writer->parameters_) {
              const std::string param_name = py::str(n);
              const OrtValue* ort_value = value.cast<OrtValue*>();
              const Tensor& tensor = ort_value->Get<Tensor>();
              const auto data_span =
                  gsl::make_span<const uint8_t>(reinterpret_cast<const uint8_t*>(tensor.DataRaw()),
                                                tensor.SizeInBytes());
              format_builder.AddParameter(
                  param_name, static_cast<adapters::TensorDataType>(tensor.GetElementType()),
                  tensor.Shape().GetDims(), data_span);
            }

            auto format_span = format_builder.FinishWithSpan(reader_writer->adapter_version_,
                                                             reader_writer->model_version_);
            if (file.write(reinterpret_cast<const char*>(format_span.data()), format_span.size()).fail()) {
              ORT_THROW("Failed to write :", std::to_string(format_span.size()), " bytes to ", file_name);
            }

            if (file.flush().fail()) {
              ORT_THROW("Failed to flush :", file_name, " on close");
            }
          },
          "Save adapter parameters into a onnxruntime adapter file format.")
      .def_static(
          "read_adapter", [](const std::string& file_name) -> std::unique_ptr<PyAdapterFormatReaderWriter> {
            lora::LoraAdapter lora_adapter;
            lora_adapter.Load(file_name);

            auto [begin, end] = lora_adapter.GetParamIterators();
            py::dict params;
            for (; begin != end; ++begin) {
              auto& [name, param] = *begin;
              OrtValue& ort_value = param.GetMapped();
              params[py::str(name)] = py::cast(&ort_value);
            }

            auto py_adapter = std::make_unique<PyAdapterFormatReaderWriter>(
                lora_adapter.FormatVersion(), lora_adapter.AdapterVersion(),
                lora_adapter.ModelVersion(), std::move(lora_adapter), std::move(params));

            return py_adapter;
          },
          "The function returns an instance of the class that contains a dictionary of name -> numpy arrays");
}

}  // namespace python
}  // namespace onnxruntime