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
#include "core/session/lora_adapters.h"

#include <string.h>

#include <fstream>

namespace onnxruntime {
namespace python {

namespace py = pybind11;
namespace {
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
        loaded_adapter_(std::move(loaded_adapter)),
        parameters_(std::move(params)) {}

  int format_version_{adapters::kAdapterFormatVersion};
  int adapter_version_{0};
  int model_version_{0};
  // This container is used when reading the the file so
  // OrtValue objects can be backed by it. Not exposed to Python
  std::optional<lora::LoraAdapter> loaded_adapter_;
  // This is a dictionary of string -> OrtValue
  // this is populated directly on write and
  // built on top of the loaded_adapter on read
  py::dict parameters_;
};

}  // namespace

/* */
void addAdapterFormatMethods(pybind11::module& m) {
  py::class_<PyAdapterFormatReaderWriter> adapter_binding(m, "AdapterFormat");
  adapter_binding.def(py::init())
      .def_property_readonly(
          "format_version",
          [](const PyAdapterFormatReaderWriter* reader_writer) -> int { return reader_writer->format_version_; },
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
          // The dict's values are pybind11 wrappers around raw OrtValue* pointers
          // that alias storage owned by PyAdapterFormatReaderWriter::loaded_adapter_
          // (see read_adapter below: entries are produced via py::cast(&ort_value),
          // i.e. non-owning views into the LoraAdapter's params_values_ map).
          //
          // Without an explicit keep-alive, Python users following the natural
          // pattern
          //
          //     params = ort.AdapterFormat.read_adapter(path).parameters
          //
          // would silently get a dict of dangling pointers: the temporary
          // AdapterFormat (and with it the LoraAdapter that backs every OrtValue)
          // is destroyed at the end of the expression, and any subsequent access
          // such as params["x"].numpy() would dereference freed memory.
          //
          // Python callers reasonably expect refcounting to keep parents alive
          // while derived objects are still referenced; this is the binding
          // author's responsibility for non-owning views. py::keep_alive<0, 1>
          // ties the returned dict (return value, index 0) to the owning
          // PyAdapterFormatReaderWriter (self, index 1), so the parent — and
          // therefore the underlying adapter buffer — survives at least as long
          // as the dict the caller holds. def_property() does not accept
          // keep_alive directly, so the policy is attached to the getter via
          // py::cpp_function.
          py::cpp_function(
              [](const PyAdapterFormatReaderWriter* reader_writer) -> py::dict { return reader_writer->parameters_; },
              py::keep_alive<0, 1>()),
          py::cpp_function(
              [](PyAdapterFormatReaderWriter* reader_writer, py::dict& parameters) -> void {
                reader_writer->parameters_ = parameters;
              }),
          R"pbdoc("Enables user to read/write the dictionary of adapter parameters (name -> OrtValue)")pbdoc")
      .def(
          "export_adapter",
          [](const PyAdapterFormatReaderWriter* reader_writer, const std::wstring& path) {
            std::filesystem::path file_path(path);

            adapters::utils::AdapterFormatBuilder format_builder;
            for (auto& [n, value] : reader_writer->parameters_) {
              const std::string param_name = py::str(n);
              const OrtValue* ort_value = value.cast<OrtValue*>();
              const Tensor& tensor = ort_value->Get<Tensor>();
              const auto element_type = tensor.GetElementType();
              // Reject string tensors: Tensor::DataRaw() for a string tensor points to an
              // array of std::string objects, and SizeInBytes() counts the sizeof(std::string)
              // object representation (which contains heap pointers and uninitialized
              // padding). Serializing those bytes would (a) leak runtime addresses
              // (defeating ASLR) and uninitialized heap memory into the adapter file,
              // and (b) produce an unloadable adapter, since reading the bytes back as
              // std::string objects is undefined behavior. The adapter format has no
              // representation for string tensors.
              if (element_type == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
                ORT_THROW("Lora adapter parameter '", param_name,
                          "' has element type STRING, which is not supported by the adapter format.");
              }
              const auto data_span =
                  gsl::make_span<const uint8_t>(reinterpret_cast<const uint8_t*>(tensor.DataRaw()),
                                                tensor.SizeInBytes());
              format_builder.AddParameter(
                  param_name, static_cast<adapters::TensorDataType>(element_type),
                  tensor.Shape().GetDims(), data_span);
            }

            // Build the entire adapter image in memory before touching the
            // filesystem, so any failure (string-tensor rejection above, builder
            // errors below, allocation failure inside FinishWithSpan, etc.)
            // leaves no stray file behind.
            auto format_span = format_builder.FinishWithSpan(reader_writer->adapter_version_,
                                                             reader_writer->model_version_);

            std::ofstream file(file_path, std::ios::binary);
            if (file.fail()) {
              ORT_THROW("Failed to open file:", file_path, " for writing.");
            }

            if (file.write(reinterpret_cast<const char*>(format_span.data()), format_span.size()).fail()) {
              ORT_THROW("Failed to write :", std::to_string(format_span.size()), " bytes to ", file_path);
            }

            if (file.flush().fail()) {
              ORT_THROW("Failed to flush :", file_path, " on close");
            }
          },
          R"pbdoc("Save adapter parameters into a onnxruntime adapter file format.)pbdoc")

      .def_static(
          "read_adapter", [](const std::wstring& file_path) -> std::unique_ptr<PyAdapterFormatReaderWriter> {
            lora::LoraAdapter lora_adapter;
            lora_adapter.Load(file_path);

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
          R"pbdoc(The function returns an instance of the class that contains a dictionary of name -> numpy arrays)pbdoc");

  py::class_<lora::LoraAdapter> lora_adapter_binding(m, "LoraAdapter");
  lora_adapter_binding.def(py::init())
      .def("Load", [](lora::LoraAdapter* adapter, const std::wstring& file_path) { adapter->MemoryMap(file_path); }, R"pbdoc(Memory map the specified file as LoraAdapter)pbdoc");
}

}  // namespace python
}  // namespace onnxruntime