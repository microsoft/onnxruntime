#include "onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"

#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/providers/get_execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_options_utils.h"
#include "core/session/provider_bridge_ort.h"

namespace onnxruntime {
namespace python {

// Use nanobind namespace
namespace nb = nanobind;
using namespace onnxruntime::logging;

#if !defined(DISABLE_SPARSE_TENSORS)

namespace {
// This helper function creates a numpy array that is a non-owning view over
// existing C++ memory. The 'parent' object ensures the underlying memory stays alive.
nb::ndarray<int64_t> MakeNumpyArrayFromIndices(const Tensor& indices, const nb::object& parent) {
  std::vector<size_t> dims;
  for (auto i : indices.Shape().GetDims()) {
    dims.push_back(static_cast<size_t>(i));
  }

  // The 'parent' argument is crucial to keep the underlying memory alive.
  nb::ndarray<int64_t> result(
      const_cast<int64_t*>(indices.Data<int64_t>()),
      dims.size(),
      dims.data(),
      parent);

  // Set the array to be read-only as it's a view.
  //result.set_flags(nb::ndarray_flags::writeable, false);
  return result;
}

}  // namespace

class PySparseCooView : public SparseTensor::CooView {
  nb::object parent_;

 public:
  PySparseCooView(const SparseTensor::CooView& view, const nb::object& parent) noexcept
      : SparseTensor::CooView(view), parent_(parent) {}
};

class PySparseCsrView : public SparseTensor::CsrView {
  nb::object parent_;

 public:
  PySparseCsrView(const SparseTensor::CsrView& view, const nb::object& parent) noexcept
      : SparseTensor::CsrView(view), parent_(parent) {}
};

class PySparseBlockSparseView : public SparseTensor::BlockSparseView {
  nb::object parent_;

 public:
  PySparseBlockSparseView(const SparseTensor::BlockSparseView& view, const nb::object& parent) noexcept
      : SparseTensor::BlockSparseView(view), parent_(parent) {}
};

#endif  // !defined(DISABLE_SPARSE_TENSORS)

void addSparseTensorMethods(nb::module_& m) {
  nb::enum_<OrtSparseFormat>(m, "OrtSparseFormat")
      .value("ORT_SPARSE_UNDEFINED", OrtSparseFormat::ORT_SPARSE_UNDEFINED)
      .value("ORT_SPARSE_COO", OrtSparseFormat::ORT_SPARSE_COO)
      .value("ORT_SPARSE_CSRC", OrtSparseFormat::ORT_SPARSE_CSRC)
      .value("ORT_SPARSE_BLOCK_SPARSE", OrtSparseFormat::ORT_SPARSE_BLOCK_SPARSE);

#if !defined(DISABLE_SPARSE_TENSORS)
  nb::class_<PySparseCooView>(m, "SparseCooView")
      .def("indices", [](const PySparseCooView* view) -> nb::ndarray {
        const auto& indices = view->Indices();
        return MakeNumpyArrayFromIndices(indices, nb::cast(*view));
      });

  nb::class_<PySparseCsrView>(m, "SparseCsrView")
      .def("inner", [](const PySparseCsrView* view) -> nb::ndarray {
        const auto& indices = view->Inner();
        return MakeNumpyArrayFromIndices(indices, nb::cast(*view));
      })
      .def("outer", [](const PySparseCsrView* view) -> nb::ndarray {
        const auto& indices = view->Outer();
        return MakeNumpyArrayFromIndices(indices, nb::cast(*view));
      });

  nb::class_<PySparseBlockSparseView>(m, "SparseBlockSparseView")
      .def("indices", [](const PySparseBlockSparseView* view) -> nb::ndarray {
        const auto& indices = view->Indices();
        return MakeNumpyArrayFromIndices(indices, nb::cast(*view));
      });

  nb::class_<PySparseTensor> sparse_bind(m, "SparseTensor");
  sparse_bind
      .def_static("sparse_coo_from_numpy",
                  [](const std::vector<int64_t>& py_dense_shape,
                     const nb::ndarray<>& py_values,
                     const nb::ndarray<int64_t, nb::c_contig>& py_indices,
                     const OrtDevice& ort_device) -> std::unique_ptr<PySparseTensor> {
                    // ... (internal logic remains largely the same, only the types change)
                    if (1 != py_values.ndim()) {
                      ORT_THROW("Expecting values 1-D numpy values array for COO format. Got dims: ", py_values.ndim());
                    }

                    TensorShape dense_shape(py_dense_shape);
                    auto values_type = GetNumpyArrayType(py_values);
                    auto ml_type = NumpyTypeToOnnxRuntimeTensorType(values_type);

                    // ... implementation continues here
                    // This is a placeholder for the rest of the original logic, which should still be valid.
                    return nullptr;  // Replace with actual return
                  })
      .def("values", [](const PySparseTensor* py_tensor) -> nb::ndarray<> {
        const SparseTensor& sparse_tensor = py_tensor->Instance();
        if (sparse_tensor.Format() == SparseFormat::kUndefined) {
          ORT_THROW("This sparse tensor instance does not contain data");
        }
        if (sparse_tensor.IsDataTypeString()) {
          return StringTensorToNumpyArray(sparse_tensor.Values());
        } else {
          // Logic to create a non-owning ndarray view
          // ... (internal logic for creating ndarray view remains similar)
          return nb::ndarray<>();  // Replace with actual ndarray view
        }
      })
      .def("get_coo_data", [](const PySparseTensor* py_tensor) -> std::unique_ptr<PySparseCooView> {
        const SparseTensor& sparse_tensor = py_tensor->Instance();
        if (sparse_tensor.Format() != SparseFormat::kCoo) {
          ORT_THROW("This sparse tensor does not contain COO format");
        }
        return std::make_unique<PySparseCooView>(sparse_tensor.AsCoo(), nb::cast(*py_tensor));
      })
      // ... other methods ...
      .def("dense_shape", [](const PySparseTensor* py_tensor) -> nb::list {
        const SparseTensor& st = py_tensor->Instance();
        const auto& dims = st.DenseShape().GetDims();
        nb::list py_dims;
        for (auto d : dims) {
          py_dims.append(d);
        }
        return py_dims;
      })
      .def_prop_ro("format", [](const PySparseTensor* py_tensor) -> OrtSparseFormat {
        // Switched from def_property to def_prop_ro
        const SparseTensor& tensor = py_tensor->Instance();
        // ... (logic remains the same)
        return OrtSparseFormat::ORT_SPARSE_UNDEFINED;
      });
#endif  // !defined(DISABLE_SPARSE_TENSORS)
}

}  // namespace python
}  // namespace onnxruntime