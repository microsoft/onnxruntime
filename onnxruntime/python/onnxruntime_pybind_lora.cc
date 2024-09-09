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

namespace onnxruntime {
namespace python {

namespace py = pybind11;

namespace {

// Check if the numpy dtype descr property has any of the known types
// that is not supported natively by numpy arrays
std::optional<std::string> GetDescrPropertyString(const py::dtype& arr_dtype) {
  std::string custom_type;
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
    return {};
  }
  return custom_type;
}
}  // namespace

void AddLoraMethods(pybind11::module& m) {
  m.def(
      "export_lora_parameters", [](const std::string& file_name, int adapter_version, int model_version, const pybind11::dict& lora_parameters) {
        std::ofstream file(file_name, std::ios::binary);
        if (file.fail()) {
          ORT_THROW("Failed to open file:", file_name, " for writing.");
        }

        lora::utils::AdapterFormatBuilder format_builder;
        for (const auto& [n, arr] : lora_parameters) {
          const std::string param_name = py::str(n);
          py::array np_array = arr.cast<py::array>();

          py::dtype arr_dtype = np_array.dtype();

          // This is the element type as supported by numpy,
          // however, we can have bfloat16 and float8 types custome types defined.
          auto ml_element_type = NumpyTypeToOnnxRuntimeTensorType(arr_dtype.num());
          auto onnx_element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
              ml_element_type->AsPrimitiveDataType()->GetDataType());

          if (!ONNX_NAMESPACE::TensorProto_DataType_IsValid(onnx_element_type)) {
            ORT_THROW("Unsupported tensor ONNX element type: ", onnx_element_type);
          }

          // Adjust for custom ONNX types
          // see https://github.com/onnx/onnx/blob/main/onnx/_custom_element_types.py
          switch (onnx_element_type) {
            // Check if this really means BFloat16 as numpy custom types are conveyed
            // by means of special annotations.
            case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
              auto custom_type = GetDescrPropertyString(arr_dtype);
              if (custom_type.has_value()) {
                // onnx_element_type = map string to type
              }
              break;
            }

           // Check if this really means one of the float8 types
            case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
              auto custom_type = GetDescrPropertyString(arr_dtype);
              if (custom_type.has_value()) {
                // onnx_element_type = map string to type
              }
              break;
            }
            default:
              break;
          };
        }
      },
      "Save lora adapter parameters into a lora file format. ");
}

}  // namespace python
}  // namespace onnxruntime