// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_mlvalue.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/graph/graph_viewer.h"

#if USE_CUDA
#define BACKEND_PROC "GPU"
#else
#define BACKEND_PROC "CPU"
#endif

#if USE_OPENMP
#define BACKEND_OPENMP "-OPENMP"
#else
#define BACKEND_OPENMP ""
#endif

#if USE_MKLDNN
#define BACKEND_MKLDNN "-MKL-DNN"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#else
#define BACKEND_MKLDNN ""
#endif

#if USE_MKLML
#define BACKEND_MKLML "-MKL-ML"
#else
#define BACKEND_MKLML ""
#endif

#if USE_OPENBLAS
#define BACKEND_OPENBLAS "-OPENBLAS"
#else
#define BACKEND_OPENBLAS ""
#endif

#define BACKEND_DEVICE BACKEND_PROC BACKEND_MKLDNN BACKEND_MKLML BACKEND_OPENBLAS
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_MKLDNN
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
#endif

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(int device_id, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_BrainSlice(int ip, bool f, const char*, const char*, const char*);
}  // namespace onnxruntime

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

#include <iterator>

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

using namespace std;
namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;

static AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<CPUAllocator>();
  return alloc;
}

static const SessionOptions& GetDefaultCPUSessionOptions() {
  static SessionOptions so;
  return so;
}

template <typename T>
void AddNonTensor(onnxruntime::MLValue& val, vector<py::object>& pyobjs) {
  pyobjs.push_back(py::cast(val.Get<T>()));
}
void AddNonTensorAsPyObj(onnxruntime::MLValue& val, vector<py::object>& pyobjs) {
  // Should be in sync with core/framework/datatypes.h
  if (val.Type() == DataTypeImpl::GetType<MapStringToString>()) {
    AddNonTensor<MapStringToString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToInt64>()) {
    AddNonTensor<MapStringToInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToFloat>()) {
    AddNonTensor<MapStringToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToDouble>()) {
    AddNonTensor<MapStringToDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToString>()) {
    AddNonTensor<MapInt64ToString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToInt64>()) {
    AddNonTensor<MapInt64ToInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToFloat>()) {
    AddNonTensor<MapInt64ToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToDouble>()) {
    AddNonTensor<MapInt64ToDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorString>()) {
    AddNonTensor<VectorString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorInt64>()) {
    AddNonTensor<VectorInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorFloat>()) {
    AddNonTensor<VectorFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorDouble>()) {
    AddNonTensor<VectorDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
    AddNonTensor<VectorMapStringToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
    AddNonTensor<VectorMapInt64ToFloat>(val, pyobjs);
  } else {
    throw std::runtime_error("Output is a non-tensor type which is not supported.");
  }
}

void AddTensorAsPyObj(onnxruntime::MLValue& val, vector<py::object>& pyobjs) {
  const Tensor& rtensor = val.Get<Tensor>();
  std::vector<npy_intp> npy_dims;
  const TensorShape& shape = rtensor.Shape();

  for (size_t n = 0; n < shape.NumDimensions(); ++n) {
    npy_dims.push_back(shape[n]);
  }

  MLDataType dtype = rtensor.DataType();
  const int numpy_type = OnnxRuntimeTensorToNumpyType(dtype);
  py::object obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
      shape.NumDimensions(), npy_dims.data(), numpy_type));

  void* outPtr = static_cast<void*>(
      PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

  if (numpy_type != NPY_OBJECT) {
    memcpy(outPtr, rtensor.DataRaw(dtype), dtype->Size() * shape.Size());
  } else {
    // Handle string type.
    py::object* outObj = static_cast<py::object*>(outPtr);
    const std::string* src = rtensor.template Data<std::string>();
    for (int i = 0; i < rtensor.Shape().Size(); i++, src++) {
      outObj[i] = py::cast(*src);
    }
  }
  pyobjs.push_back(obj);
}

class SessionObjectInitializer {
 public:
  typedef const SessionOptions& Arg1;
  typedef logging::LoggingManager* Arg2;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  operator Arg2() {
    static std::string default_logger_id{"Default"};
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
    return &default_logging_manager;
  }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

inline void RegisterExecutionProvider(InferenceSession* sess, onnxruntime::IExecutionProviderFactory& f) {
  auto p = f.CreateProvider();
  auto status = sess->RegisterExecutionProvider(std::move(p));
  if (!status.IsOK()) {
    throw std::runtime_error(status.ErrorMessage().c_str());
  }
}

void InitializeSession(InferenceSession* sess) {
  onnxruntime::common::Status status;

#ifdef USE_CUDA
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_CUDA(0));
  }
#endif

#ifdef USE_MKLDNN
  {
    const bool enable_cpu_mem_arena = true;
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Mkldnn(enable_cpu_mem_arena ? 1 : 0));
  }
#endif
#if 0  //USE_NUPHAR
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Nuphar(0, ""));
  }
#endif

  status = sess->Initialize();
  if (!status.IsOK()) {
    throw std::runtime_error(status.ToString().c_str());
  }
}  // namespace python

void addGlobalMethods(py::module& m) {
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
  m.def(
      "get_device", []() -> std::string { return BACKEND_DEVICE; },
      "Return the device used to compute the prediction (CPU, MKL, ...)");
}

void addObjectMethods(py::module& m) {
  // allow unit tests to redirect std::cout and std::cerr to sys.stdout and sys.stderr
  py::add_ostream_redirect(m, "onnxruntime_ostream_redirect");
  py::class_<SessionOptions>(m, "SessionOptions", R"pbdoc(Configuration information for a session.)pbdoc")
      .def(py::init())
      .def_readwrite("enable_cpu_mem_arena", &SessionOptions::enable_cpu_mem_arena,
                     R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_readwrite("enable_profiling", &SessionOptions::enable_profiling,
                     R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_readwrite("enable_sequential_execution", &SessionOptions::enable_sequential_execution,
                     R"pbdoc(Enables sequential execution, disables parallel execution. Default is true.)pbdoc")
      .def_readwrite("max_num_graph_transformation_steps", &SessionOptions::max_num_graph_transformation_steps,
                     R"pbdoc(Runs optimization steps on the execution graph. Default is 5.)pbdoc")
      .def_readwrite("session_logid", &SessionOptions::session_logid,
                     R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_readwrite("session_log_verbosity_level", &SessionOptions::session_log_verbosity_level,
                     R"pbdoc(Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_readwrite("session_thread_pool_size", &SessionOptions::session_thread_pool_size,
                     R"pbdoc(How many threads in the session thread pool. Default is 0 to let onnxruntime choose.
This parameter is unused unless *enable_sequential_execution* is false.)pbdoc");

  py::class_<RunOptions>(m, "RunOptions", R"pbdoc(Configuration information for a single Run.)pbdoc")
      .def(py::init())
      .def_readwrite("run_log_verbosity_level", &RunOptions::run_log_verbosity_level,
                     "Applies to a particular Run() invocation.")
      .def_readwrite("run_tag", &RunOptions::run_tag,
                     "To identify logs generated by a particular Run() invocation.")
      .def_readwrite("terminate", &RunOptions::terminate,
                     R"pbdoc(Set to True to terminate any currently executing calls that are using this
RunOptions instance. The individual calls will exit gracefully and return an error status.)pbdoc");

  py::class_<ModelMetadata>(m, "ModelMetadata", R"pbdoc(Pre-defined and custom metadata about the model.
It is usually used to identify the model used to run the prediction and
facilitate the comparison.)pbdoc")
      .def_readwrite("producer_name", &ModelMetadata::producer_name, "producer name")
      .def_readwrite("graph_name", &ModelMetadata::graph_name, "graph name")
      .def_readwrite("domain", &ModelMetadata::domain, "ONNX domain")
      .def_readwrite("description", &ModelMetadata::description, "description of the model")
      .def_readwrite("version", &ModelMetadata::version, "version of the model")
      .def_readwrite("custom_metadata_map", &ModelMetadata::custom_metadata_map, "additional metadata");

  py::class_<onnxruntime::NodeArg>(m, "NodeArg", R"pbdoc(Node argument definition, for both input and output,
including arg name, arg type (contains both type and shape).)pbdoc")
      .def_property_readonly("name", &onnxruntime::NodeArg::Name, "node name")
      .def_property_readonly(
          "type", [](const onnxruntime::NodeArg& na) -> std::string {
            return *(na.Type());
          },
          "node type")
      .def(
          "__str__", [](const onnxruntime::NodeArg& na) -> std::string {
            std::ostringstream res;
            res << "NodeArg(name='" << na.Name() << "', type='" << *(na.Type()) << "', shape=";
            auto shape = na.Shape();
            std::vector<py::object> arr;
            if (shape == nullptr || shape->dim_size() == 0) {
              res << "[]";
            } else {
              res << "[";
              for (int i = 0; i < shape->dim_size(); ++i) {
                if (shape->dim(i).has_dim_value()) {
                  res << shape->dim(i).dim_value();
                } else if (shape->dim(i).has_dim_param()) {
                  res << "None";
                }
                if (i < shape->dim_size() - 1) {
                  res << ", ";
                }
              }
              res << "]";
            }
            res << ")";

            return std::string(res.str());
          },
          "converts the node into a readable string")
      .def_property_readonly(
          "shape", [](const onnxruntime::NodeArg& na) -> std::vector<py::object> {
            auto shape = na.Shape();
            std::vector<py::object> arr;
            if (shape == nullptr || shape->dim_size() == 0) {
              return arr;
            }

            arr.resize(shape->dim_size());
            for (int i = 0; i < shape->dim_size(); ++i) {
              if (shape->dim(i).has_dim_value()) {
                arr[i] = py::cast(shape->dim(i).dim_value());
              } else if (shape->dim(i).has_dim_param()) {
                arr[i] = py::none();
              }
            }
            return arr;
          },
          "node shape (assuming the node holds a tensor)");

  py::class_<SessionObjectInitializer>(m, "SessionObjectInitializer");
  py::class_<InferenceSession>(m, "InferenceSession", R"pbdoc(This is the main class used to run a model.)pbdoc")
      .def(py::init<SessionObjectInitializer, SessionObjectInitializer>())
      .def(py::init<SessionOptions, SessionObjectInitializer>())
      .def(
          "load_model", [](InferenceSession* sess, const std::string& path) {
            auto status = sess->Load(path);
            if (!status.IsOK()) {
              throw std::runtime_error(status.ToString().c_str());
            }
            InitializeSession(sess);
          },
          R"pbdoc(Load a model saved in ONNX format.)pbdoc")
      .def(
          "read_bytes", [](InferenceSession* sess, const py::bytes& serializedModel) {
            std::istringstream buffer(serializedModel);
            auto status = sess->Load(buffer);
            if (!status.IsOK()) {
              throw std::runtime_error(status.ToString().c_str());
            }
            InitializeSession(sess);
          },
          R"pbdoc(Load a model serialized in ONNX format.)pbdoc")
      .def("run", [](InferenceSession* sess, std::vector<std::string> output_names, std::map<std::string, py::object> pyfeeds, RunOptions* run_options = nullptr) -> std::vector<py::object> {
        NameMLValMap feeds;
        for (auto _ : pyfeeds) {
          MLValue ml_value;
          CreateGenericMLValue(GetAllocator(), _.first, _.second, &ml_value);
          if (PyErr_Occurred()) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);

            PyObject* pStr = PyObject_Str(ptype);
            std::string sType = py::reinterpret_borrow<py::str>(pStr);
            Py_XDECREF(pStr);
            pStr = PyObject_Str(pvalue);
            sType += ": ";
            sType += py::reinterpret_borrow<py::str>(pStr);
            Py_XDECREF(pStr);
            throw std::runtime_error(sType);
          }
          feeds.insert(std::make_pair(_.first, ml_value));
        }

        std::vector<MLValue> fetches;
        common::Status status;

        if (run_options != nullptr) {
          status = sess->Run(*run_options, feeds, output_names, &fetches);
        } else {
          status = sess->Run(feeds, output_names, &fetches);
        }

        if (!status.IsOK()) {
          auto mes = status.ToString();
          throw std::runtime_error(std::string("Method run failed due to: ") + std::string(mes.c_str()));
        }

        std::vector<py::object> rfetch;
        rfetch.reserve(fetches.size());
        for (auto _ : fetches) {
          if (_.IsTensor()) {
            AddTensorAsPyObj(_, rfetch);
          } else {
            AddNonTensorAsPyObj(_, rfetch);
          }
        }
        return rfetch;
      })
      .def("end_profiling", [](InferenceSession* sess) -> std::string {
        return sess->EndProfiling();
      })
      .def_property_readonly("inputs_meta", [](const InferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
        auto res = sess->GetModelInputs();
        if (!res.first.IsOK()) {
          throw std::runtime_error(res.first.ToString().c_str());
        } else {
          return *(res.second);
        }
      })
      .def_property_readonly("outputs_meta", [](const InferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
        auto res = sess->GetModelOutputs();
        if (!res.first.IsOK()) {
          throw std::runtime_error(res.first.ToString().c_str());
        } else {
          return *(res.second);
        }
      })
      .def_property_readonly("model_meta", [](const InferenceSession* sess) -> const onnxruntime::ModelMetadata& {
        auto res = sess->GetModelMetadata();
        if (!res.first.IsOK()) {
          throw std::runtime_error(res.first.ToString().c_str());
        } else {
          return *(res.second);
        }
      });
}

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to ONNX runtime";

  auto initialize = [&]() {
    // Initialization of the module
    ([]() -> void {
      // import_array1() forces a void return value.
      import_array1();
    })();

    static std::unique_ptr<Environment> env;
    auto status = Environment::Create(env);
    if (!status.IsOK()) {
      throw std::runtime_error(status.ToString().c_str());
    }

    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();

  addGlobalMethods(m);
  addObjectMethods(m);
}

}  // namespace python
}  // namespace onnxruntime
