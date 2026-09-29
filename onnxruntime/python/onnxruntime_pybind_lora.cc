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
/// in onnxruntime format.
///
/// Design: Single-source architecture using py::capsule for memory ownership.
///
/// `parameters_` is always the single authoritative dict of name -> OrtValue.
/// On the read path, OrtValues are zero-copy views into a LoraAdapter's buffer.
/// The LoraAdapter is owned by a py::capsule that is pinned as a patient on each
/// OrtValue via pybind11's add_patient. The object graph is:
///
///   PyAdapterFormatReaderWriter -> parameters_ dict -> OrtValues -> capsule -> LoraAdapter
///
/// No edge points back to PyAdapterFormatReaderWriter, so there is no reference
/// cycle (pybind11 instances are not GC-traversable, so a cycle would leak).
/// The capsule keeps the backing memory alive as long as any OrtValue referencing
/// it is alive.
///
/// The setter replaces parameters_ freely. Old OrtValues (and their capsule
/// patient) remain alive as long as Python references to them exist.
/// </summary>
struct PyAdapterFormatReaderWriter {
  PyAdapterFormatReaderWriter() = default;
  PyAdapterFormatReaderWriter(int format_version, int adapter_version,
                              int model_version,
                              py::dict parameters)
      : format_version_(format_version),
        adapter_version_(adapter_version),
        model_version_(model_version),
        parameters_(std::move(parameters)) {}

  int format_version_{adapters::kAdapterFormatVersion};
  int adapter_version_{0};
  int model_version_{0};
  // Single source of truth: name -> OrtValue (views or owning).
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
          [](const PyAdapterFormatReaderWriter* reader_writer) -> py::dict {
            return reader_writer->parameters_;
          },
          [](PyAdapterFormatReaderWriter* reader_writer, py::dict parameters) -> void {
            reader_writer->parameters_ = std::move(parameters);
          },
          R"pbdoc("Enables user to read/write the dictionary of adapter parameters (name -> OrtValue)")pbdoc")
      .def(
          "export_adapter",
          [](const PyAdapterFormatReaderWriter* reader_writer, const std::wstring& path) {
            std::filesystem::path file_path(path);

            adapters::utils::AdapterFormatBuilder format_builder;

            auto add_param = [&format_builder](const std::string& param_name, const OrtValue& ort_value) {
              ORT_ENFORCE(ort_value.IsTensor(),
                          "Lora adapter parameter '", param_name, "' must be a Tensor OrtValue.");
              const Tensor& tensor = ort_value.Get<Tensor>();
              ORT_ENFORCE(tensor.Location().device.Type() == OrtDevice::CPU,
                          "Lora adapter parameter '", param_name, "' must reside on CPU to be exported.");
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
            };

            // Single source: always iterate parameters_ dict.
            for (auto& [n, value] : reader_writer->parameters_) {
              const std::string param_name = py::str(n);
              const OrtValue* ort_value = value.cast<OrtValue*>();
              add_param(param_name, *ort_value);
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
            // Load into a heap-allocated LoraAdapter, then wrap it in a capsule.
            // The capsule is pinned as a patient on each OrtValue view, so the
            // backing memory stays alive as long as any Python reference to an
            // OrtValue (or the dict containing them) exists.
            auto adapter_ptr = std::make_unique<lora::LoraAdapter>();
            adapter_ptr->Load(file_path);
            const int format_version = adapter_ptr->FormatVersion();
            const int adapter_version = adapter_ptr->AdapterVersion();
            const int model_version = adapter_ptr->ModelVersion();

            // Transfer ownership of the LoraAdapter to a capsule.
            // Construct capsule while unique_ptr still owns the pointer, so
            // if capsule allocation throws, unique_ptr cleans up.
            py::capsule adapter_capsule(adapter_ptr.get(), [](void* p) {
              delete static_cast<lora::LoraAdapter*>(p);
            });
            lora::LoraAdapter* raw_adapter = adapter_ptr.release();

            // Build the parameters dict with OrtValue views pinned to the capsule.
            py::dict params;
            auto [begin, end] = raw_adapter->GetParamIterators();
            for (; begin != end; ++begin) {
              auto& [name, param] = *begin;
              OrtValue& ort_value = param.GetMapped();
              // Cast with the capsule as parent: pybind11 attaches the capsule
              // as a patient on the OrtValue handle (strong refcount pin).
              py::object ort_value_obj = py::cast(
                  &ort_value, py::return_value_policy::reference_internal, adapter_capsule);
              params[py::str(name)] = std::move(ort_value_obj);
            }

            return std::make_unique<PyAdapterFormatReaderWriter>(
                format_version, adapter_version, model_version, std::move(params));
          },
          R"pbdoc(The function returns an instance of the class that contains a dictionary of name -> OrtValue)pbdoc");

  py::class_<lora::LoraAdapter> lora_adapter_binding(m, "LoraAdapter");
  lora_adapter_binding.def(py::init())
      .def("Load", [](lora::LoraAdapter* adapter, const std::wstring& file_path) { adapter->MemoryMap(file_path); }, R"pbdoc(Memory map the specified file as LoraAdapter)pbdoc");
}

}  // namespace python
}  // namespace onnxruntime
