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

#include "lora/lora_format_utils.h"
#include "lora/lora_adapters.h"

#include <string.h>

#include <fstream>
#include <unordered_map>

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

}  // namespace

void AddAdapterMethods(pybind11::module& m) {
  m.def(
      "export_adapter", [](const std::string& file_name, int adapter_version, int model_version, const pybind11::dict& adapter_parameters) {
        std::ofstream file(file_name, std::ios::binary);
        if (file.fail()) {
          ORT_THROW("Failed to open file:", file_name, " for writing.");
        }

        lora::utils::AdapterFormatBuilder format_builder;
        for (const auto& [n, arr] : adapter_parameters) {
          const std::string param_name = py::str(n);
          py::array np_array = arr.cast<py::array>();

          py::dtype arr_dtype = np_array.dtype();

          // This is the element type as supported by numpy,
          // however, we can have bfloat16 and float8 custom types defined.
          auto ml_element_type = NumpyTypeToOnnxRuntimeTensorType(arr_dtype.num());
          auto onnx_element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
              ml_element_type->AsPrimitiveDataType()->GetDataType());

          if (!ONNX_NAMESPACE::TensorProto_DataType_IsValid(onnx_element_type)) {
            ORT_THROW("Unsupported tensor ONNX element type: ", onnx_element_type);
          }

          switch (onnx_element_type) {
            case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
            case ONNX_NAMESPACE::TensorProto_DataType_INT8:
            case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
              onnx_element_type =
                  static_cast<ONNX_NAMESPACE::TensorProto_DataType>(AdjustOnnxTypeIfNeeded(arr_dtype,
                                                                                           onnx_element_type));
              break;
            }
            default:
              break;
          };

          gsl::span<const int64_t> shape_span{reinterpret_cast<const int64_t*>(np_array.shape()),
                                              static_cast<size_t>(np_array.ndim())};
          gsl::span<const uint8_t> data_span{reinterpret_cast<const uint8_t*>(np_array.data()),
                                             static_cast<size_t>(np_array.nbytes())};

          format_builder.AddParameter(param_name, static_cast<lora::TensorDataType>(onnx_element_type),
                                      shape_span, data_span);
        }
        auto format_span = format_builder.FinishWithSpan(adapter_version, model_version);
        if (file.write(reinterpret_cast<const char*>(format_span.data()), format_span.size()).fail()) {
          ORT_THROW("Failed to write :", std::to_string(format_span.size()), " bytes to ", file_name);
        }

        if (file.flush().fail()) {
          ORT_THROW("Failed to flush :", file_name, " on close");
        }
      },
      "Save adapter parameters into a lora file format. ");

  class PyAdapter {
   public:
    PyAdapter(int format_version, int adapter_version,
              int model_version, py::dict params) : format_version_(format_version), adapter_version_(adapter_version), model_version_(model_version), parameters_(std::move(params)) {}

    int FormatVersion() const noexcept {
      return format_version_;
    }

    int AdapterVersion() const noexcept {
      return adapter_version_;
    }

    int ModelVersion() const noexcept {
      return model_version_;
    }

    py::dict GetParameters() const noexcept {
      return parameters_;
    }

   private:
    int format_version_;
    int adapter_version_;
    int model_version_;
    py::dict parameters_;
  };

  py::class_<PyAdapter> adapter_binding(m, "LoraAdapter");
  adapter_binding.def(py::init<int, int, int, py::dict>());
  adapter_binding.def("get_format_version", [](PyAdapter* py_adapter) -> int {
    return py_adapter->FormatVersion();
  });
  adapter_binding.def("get_adapter_version", [](PyAdapter* py_adapter) -> int {
    return py_adapter->AdapterVersion();
  });
  adapter_binding.def("get_model_version", [](PyAdapter* py_adapter) -> int {
    return py_adapter->ModelVersion();
  });
  adapter_binding.def("get_arameters", [](PyAdapter* py_adapter) -> py::dict {
    return py_adapter->GetParameters();
  });

  m.def("read_adapter", [](const std::string& file_name) -> std::unique_ptr<PyAdapter> {
    lora::LoadedAdapter adapter;
    adapter.MemoryMap(file_name);

    auto [begin, end] = adapter.GetParamIterators();
    py::dict params;
    for (; begin != end; ++begin) {
      const auto& [name, param] = *begin;
      const auto& tensor = param.GetMapped().Get<Tensor>();

      const auto onnx_type = tensor.GetElementType();
      const auto size_bytes = tensor.SizeInBytes();

      py::dtype dtype = ConstructDType(onnx_type);
      py::array npy_array(dtype, tensor.Shape().GetDims());
      ORT_ENFORCE(npy_array.size(), tensor.Shape().Size());
      memcpy_s(npy_array.mutable_data(), size_bytes, tensor.DataRaw(), size_bytes);
      params[py::str(name)] = std::move(npy_array);
    }

    auto py_adapter = std::make_unique<PyAdapter>(adapter.FormatVersion(), adapter.AdapterVersion(),
                                                 adapter.ModelVersion(), std::move(params));
    return py_adapter;
  });
}

}  // namespace python
}  // namespace onnxruntime