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

#if USE_NGRAPH
#define BACKEND_NGRAPH "-NGRAPH"
#include "core/providers/ngraph/ngraph_execution_provider.h"
#else
#define BACKEND_NGRAPH ""
#endif

#if USE_OPENVINO
#define BACKEND_OPENVINO "-OPENVINO"
#else
#define BACKEND_OPENVINO ""
#endif

#if USE_OPENBLAS
#define BACKEND_OPENBLAS "-OPENBLAS"
#else
#define BACKEND_OPENBLAS ""
#endif

#define BACKEND_DEVICE BACKEND_PROC BACKEND_MKLDNN BACKEND_MKLML BACKEND_NGRAPH BACKEND_OPENVINO BACKEND_OPENBLAS
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_MKLDNN
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#endif
#ifdef USE_NGRAPH
#include "core/providers/ngraph/ngraph_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
#endif
#ifdef USE_BRAINSLICE
#include "core/providers/brainslice/brainslice_provider_factory.h"
#endif

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_NGraph(const char* ng_backend_type);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const char* device);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(int device_id, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_BrainSlice(uint32_t ip, int, int, bool, const char*, const char*, const char*);
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
void AddNonTensor(OrtValue& val, vector<py::object>& pyobjs) {
  pyobjs.push_back(py::cast(val.Get<T>()));
}
void AddNonTensorAsPyObj(OrtValue& val, vector<py::object>& pyobjs) {
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

void AddTensorAsPyObj(OrtValue& val, vector<py::object>& pyobjs) {
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

#ifdef USE_TENSORRT
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Tensorrt());
  }
#endif

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

#if USE_NGRAPH
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_NGraph("CPU"));
  }
#endif

#ifdef USE_OPENVINO
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_OpenVINO("CPU"));
  }
#endif

#if 0  //USE_NUPHAR
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Nuphar(0, ""));
  }
#endif

#ifdef USE_BRAINSLICE
  {
    RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_BrainSlice(0, -1, -1, false, "", "", ""));
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
  m.def("set_default_logger_severity", [](int severity) {
    ORT_ENFORCE(severity >= 0 && severity <= 4,
                "Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
    logging::LoggingManager* default_logging_manager = SessionObjectInitializer::Get();
    default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity));
  });

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  m.def(
      "get_all_operator_schema", []() -> const std::vector<ONNX_NAMESPACE::OpSchema> {
        return ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
      },
      "Return a vector of OpSchema all registed operators");
#endif
}

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA

void addOpSchemaSubmodule(py::module& m) {
  auto schemadef = m.def_submodule("schemadef");
  schemadef.doc() = "Schema submodule";

  py::class_<ONNX_NAMESPACE::OpSchema> op_schema(schemadef, "OpSchema");
  op_schema.def_property_readonly("file", &ONNX_NAMESPACE::OpSchema::file)
      .def_property_readonly("line", &ONNX_NAMESPACE::OpSchema::line)
      .def_property_readonly("support_level", &ONNX_NAMESPACE::OpSchema::support_level)
      .def_property_readonly(
          "doc", &ONNX_NAMESPACE::OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("since_version", &ONNX_NAMESPACE::OpSchema::since_version)
      .def_property_readonly("deprecated", &ONNX_NAMESPACE::OpSchema::deprecated)
      .def_property_readonly("domain", &ONNX_NAMESPACE::OpSchema::domain)
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::Name)
      .def_property_readonly("min_input", &ONNX_NAMESPACE::OpSchema::min_input)
      .def_property_readonly("max_input", &ONNX_NAMESPACE::OpSchema::max_input)
      .def_property_readonly("min_output", &ONNX_NAMESPACE::OpSchema::min_output)
      .def_property_readonly("max_output", &ONNX_NAMESPACE::OpSchema::max_output)
      .def_property_readonly("attributes", &ONNX_NAMESPACE::OpSchema::attributes)
      .def_property_readonly("inputs", &ONNX_NAMESPACE::OpSchema::inputs)
      .def_property_readonly("outputs", &ONNX_NAMESPACE::OpSchema::outputs)
      .def_property_readonly(
          "has_type_and_shape_inference_function",
          &ONNX_NAMESPACE::OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly(
          "type_constraints", &ONNX_NAMESPACE::OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) {
        return v == std::numeric_limits<int>::max();
      });

  py::class_<ONNX_NAMESPACE::OpSchema::Attribute>(op_schema, "Attribute")
      .def_readonly("name", &ONNX_NAMESPACE::OpSchema::Attribute::name)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::Attribute::description)
      .def_readonly("type", &ONNX_NAMESPACE::OpSchema::Attribute::type)
      .def_property_readonly(
          "_default_value",
          [](ONNX_NAMESPACE::OpSchema::Attribute* attr) -> py::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return out;
          })
      .def_readonly("required", &ONNX_NAMESPACE::OpSchema::Attribute::required);

  py::class_<ONNX_NAMESPACE::OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def_readonly(
          "type_param_str", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::description)
      .def_readonly(
          "allowed_type_strs",
          &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::allowed_type_strs);

  py::enum_<ONNX_NAMESPACE::OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", ONNX_NAMESPACE::OpSchema::Single)
      .value("Optional", ONNX_NAMESPACE::OpSchema::Optional)
      .value("Variadic", ONNX_NAMESPACE::OpSchema::Variadic);

  py::class_<ONNX_NAMESPACE::OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly(
          "description", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetOption)
      .def_property_readonly(
          "isHomogeneous", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetIsHomogeneous);

  py::enum_<ONNX_NAMESPACE::AttributeProto::AttributeType>(op_schema, "AttrType")
      .value("FLOAT", ONNX_NAMESPACE::AttributeProto::FLOAT)
      .value("INT", ONNX_NAMESPACE::AttributeProto::INT)
      .value("STRING", ONNX_NAMESPACE::AttributeProto::STRING)
      .value("TENSOR", ONNX_NAMESPACE::AttributeProto::TENSOR)
      .value("GRAPH", ONNX_NAMESPACE::AttributeProto::GRAPH)
      .value("FLOATS", ONNX_NAMESPACE::AttributeProto::FLOATS)
      .value("INTS", ONNX_NAMESPACE::AttributeProto::INTS)
      .value("STRINGS", ONNX_NAMESPACE::AttributeProto::STRINGS)
      .value("TENSORS", ONNX_NAMESPACE::AttributeProto::TENSORS)
      .value("GRAPHS", ONNX_NAMESPACE::AttributeProto::GRAPHS);

  py::enum_<ONNX_NAMESPACE::OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", ONNX_NAMESPACE::OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", ONNX_NAMESPACE::OpSchema::SupportType::EXPERIMENTAL);
}

#endif  //onnxruntime_PYBIND_EXPORT_OPSCHEMA

void addObjectMethods(py::module& m) {
  py::class_<SessionOptions>(m, "SessionOptions", R"pbdoc(Configuration information for a session.)pbdoc")
      .def(py::init())
      .def_readwrite("enable_cpu_mem_arena", &SessionOptions::enable_cpu_mem_arena,
                     R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_readwrite("enable_profiling", &SessionOptions::enable_profiling,
                     R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_readwrite("enable_mem_pattern", &SessionOptions::enable_mem_pattern,
                     R"pbdoc(Enable the memory pattern optimization. Default is true.)pbdoc")
      .def_readwrite("enable_sequential_execution", &SessionOptions::enable_sequential_execution,
                     R"pbdoc(Enables sequential execution, disables parallel execution. Default is true.)pbdoc")
      .def_readwrite("max_num_graph_transformation_steps", &SessionOptions::max_num_graph_transformation_steps,
                     R"pbdoc(Runs optimization steps on the execution graph. Default is 5.)pbdoc")
      .def_readwrite("session_logid", &SessionOptions::session_logid,
                     R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_readwrite("session_log_severity_level", &SessionOptions::session_log_severity_level,
                     R"pbdoc(Log severity level. Applies to session load, initialization, etc. 
0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("session_log_verbosity_level", &SessionOptions::session_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and session_log_verbosity_level is 0. 
Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_readwrite("session_thread_pool_size", &SessionOptions::session_thread_pool_size,
                     R"pbdoc(How many threads in the session thread pool. Default is 0 to let onnxruntime choose.
This parameter is unused unless *enable_sequential_execution* is false.)pbdoc")
      .def_property_readonly(
          "graph_optimization_level",
          [](const SessionOptions* options) -> uint32_t {
            return static_cast<uint32_t>(options->graph_optimization_level);
          },
          R"pbdoc(Graph optimization level for this session.)pbdoc")
      .def(
          "set_graph_optimization_level",
          [](SessionOptions* options, uint32_t level) -> void {
            options->graph_optimization_level = static_cast<TransformerLevel>(level);
          },
          R"pbdoc(Graph optimization level for this session. 0 disables all optimizations.
Whereas 1 enables basic optimizations and 2 enables all optimizations.)pbdoc");

  py::class_<RunOptions>(m, "RunOptions", R"pbdoc(Configuration information for a single Run.)pbdoc")
      .def(py::init())
      .def_readwrite("run_log_severity_level", &RunOptions::run_log_severity_level,
                     R"pbdoc(Log severity level for a particular Run() invocation. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("run_log_verbosity_level", &RunOptions::run_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and run_log_severity_level is 0. 
Applies to a particular Run() invocation. Default is 0.)pbdoc")
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
                  res << "'" << shape->dim(i).dim_param() << "'";
                } else {
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
                arr[i] = py::cast(shape->dim(i).dim_param());
              } else {
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
          OrtValue ml_value;
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

        std::vector<OrtValue> fetches;
        common::Status status;

        {
          // release GIL to allow multiple python threads to invoke Run() in parallel.
          py::gil_scoped_release release;
          if (run_options != nullptr) {
            status = sess->Run(*run_options, feeds, output_names, &fetches);
          } else {
            status = sess->Run(feeds, output_names, &fetches);
          }
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

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  addOpSchemaSubmodule(m);
#endif
}

}  // namespace python
}  // namespace onnxruntime
