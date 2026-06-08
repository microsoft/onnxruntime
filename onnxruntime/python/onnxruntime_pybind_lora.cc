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
/// Design: Two-source architecture (loaded_adapter_ vs parameters_)
///
/// On the read path, OrtValues are zero-copy views into the LoraAdapter's
/// memory-mapped (or loaded) buffer. This avoids duplicating potentially
/// large adapter weights on memory-constrained devices.
///
/// Unifying into a single cached py::dict is not possible because it would
/// create an un-collectable reference cycle:
///   self -> parameters_ dict -> OrtValue (pybind11 patient list) -> self
/// pybind11 instances do not implement tp_traverse, so Python's cyclic GC
/// cannot break this cycle and the object would leak.
///
/// Instead:
///   - loaded_adapter_ holds backing memory; the getter builds a fresh dict
///     each call, pinning `self` on each OrtValue (no cycle since the dict
///     is not stored on self).
///   - parameters_ is used only on the write path (user-supplied values).
///   - The setter clears loaded_adapter_ so that after an explicit
///     set_parameters, both getter and export_adapter use the new dict.
/// </summary>
struct PyAdapterFormatReaderWriter {
  PyAdapterFormatReaderWriter() = default;
  PyAdapterFormatReaderWriter(int format_version, int adapter_version,
                              int model_version,
                              lora::LoraAdapter&& loaded_adapter)
      : format_version_(format_version),
        adapter_version_(adapter_version),
        model_version_(model_version),
        loaded_adapter_(std::move(loaded_adapter)) {}

  int format_version_{adapters::kAdapterFormatVersion};
  int adapter_version_{0};
  int model_version_{0};
  // Read path: owns the backing memory for zero-copy OrtValue views.
  std::optional<lora::LoraAdapter> loaded_adapter_;
  // Write path: user-supplied dict of name -> OrtValue.
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
          // Getter: builds a fresh dict each call. On the read path, each
          // OrtValue is a non-owning view into loaded_adapter_'s buffer.
          // We pin `self` as a patient on each OrtValue (not on the dict,
          // since dicts are not weak-referenceable and keep_alive would fail).
          // Building fresh each call avoids storing the dict on self, which
          // would create an un-collectable cycle (see class comment above).
          [](py::object self) -> py::dict {
            auto* rw = self.cast<PyAdapterFormatReaderWriter*>();
            if (!rw->loaded_adapter_.has_value()) {
              // Write path: return whatever the user previously set.
              return rw->parameters_;
            }
            py::dict params;
            auto [begin, end] = rw->loaded_adapter_->GetParamIterators();
            for (; begin != end; ++begin) {
              auto& [name, param] = *begin;
              OrtValue& ort_value = param.GetMapped();
              // Pin `self` as patient on each OrtValue view.
              py::object ort_value_obj = py::cast(
                  &ort_value, py::return_value_policy::reference_internal, self);
              params[py::str(name)] = std::move(ort_value_obj);
            }
            return params;
          },
          [](PyAdapterFormatReaderWriter* reader_writer, py::dict& parameters) -> void {
            // Clear loaded_adapter_ so subsequent getter and export_adapter
            // operations use the user-supplied dict. Without this, a read →
            // set_parameters → export sequence would silently ignore the new
            // parameters and re-export the original read data.
            reader_writer->loaded_adapter_.reset();
            reader_writer->parameters_ = parameters;
          },
          R"pbdoc("Enables user to read/write the dictionary of adapter parameters (name -> OrtValue)")pbdoc")
      .def(
          "export_adapter",
          [](const PyAdapterFormatReaderWriter* reader_writer, const std::wstring& path) {
            std::filesystem::path file_path(path);

            adapters::utils::AdapterFormatBuilder format_builder;

            // Visit every (name, OrtValue) pair from the appropriate source.
            // We avoid building any intermediate Python dict here so the
            // export path is exception-safe by construction: the builder
            // accumulates in memory, and we only touch the filesystem
            // after FinishWithSpan succeeds.
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

            if (reader_writer->loaded_adapter_.has_value()) {
              auto [begin, end] = reader_writer->loaded_adapter_->GetParamIterators();
              for (; begin != end; ++begin) {
                auto& [name, param] = *begin;
                add_param(name, param.GetMapped());
              }
            } else {
              for (auto& [n, value] : reader_writer->parameters_) {
                const std::string param_name = py::str(n);
                const OrtValue* ort_value = value.cast<OrtValue*>();
                add_param(param_name, *ort_value);
              }
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
            const int format_version = lora_adapter.FormatVersion();
            const int adapter_version = lora_adapter.AdapterVersion();
            const int model_version = lora_adapter.ModelVersion();
            return std::make_unique<PyAdapterFormatReaderWriter>(
                format_version, adapter_version, model_version, std::move(lora_adapter));
          },
          R"pbdoc(The function returns an instance of the class that contains a dictionary of name -> numpy arrays)pbdoc");

  py::class_<lora::LoraAdapter> lora_adapter_binding(m, "LoraAdapter");
  lora_adapter_binding.def(py::init())
      .def("Load", [](lora::LoraAdapter* adapter, const std::wstring& file_path) { adapter->MemoryMap(file_path); }, R"pbdoc(Memory map the specified file as LoraAdapter)pbdoc");
}

}  // namespace python
}  // namespace onnxruntime
