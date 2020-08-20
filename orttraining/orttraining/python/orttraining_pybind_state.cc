// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_state_common.h"

// pybind11/stl.h is needed to support std::unordered_set, etc.
#include <pybind11/stl.h>

#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/mpi_setup.h"
#include "python/onnxruntime_pybind_mlvalue.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;

struct TrainingParameters {
  std::string loss_output_name;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_set<std::string> weights_not_to_train;
  onnxruntime::training::TrainingSession::ImmutableWeights immutable_weights;

  // optimizer
  std::string training_optimizer_name;
  std::string lr_params_feed_name = "Learning_Rate";
  std::unordered_map<std::string, std::unordered_map<std::string, float>> optimizer_attributes_map;
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> optimizer_int_attributes_map;
  bool use_fp16_moments = false;

  bool use_mixed_precision = false;
  bool allreduce_post_accumulation = false;
  float loss_scale = 0.0f;
  int world_rank = 0;
  int world_size = 1;
  int local_rank = 0;
  int local_size = 1;
  int gradient_accumulation_steps = 1;
  int data_parallel_size = 1;
  int horizontal_parallel_size = 1;
  int deepspeed_zero_stage = 0;
  bool enable_grad_norm_clip = true;
  bool set_gradients_as_graph_outputs = false;
  bool use_invertible_layernorm_grad = false;
};

struct TrainingConfigurationResult {
  optional<std::string> loss_scale_input_name;
};

// TODO: this method does not handle parallel optimization.
TrainingConfigurationResult ConfigureSessionForTraining(
    training::TrainingSession* sess, TrainingParameters& parameters) {
  //TODO tix, refactor the mpi related code to populate all fields correctly by default.
  ORT_ENFORCE(parameters.horizontal_parallel_size <= parameters.world_size);
  ORT_ENFORCE(parameters.data_parallel_size <= parameters.world_size);
  if (parameters.world_size % parameters.horizontal_parallel_size != 0) {
    throw std::runtime_error("Cannot split horizontal parallel group because world_size is not divisible");
  }

  auto data_group_size = parameters.world_size / parameters.horizontal_parallel_size;
  if (data_group_size != parameters.data_parallel_size) {
    std::cout << "WARNING: data_parallel_size is not correct, tuned automatically to "
              << data_group_size << std::endl;
    parameters.data_parallel_size = data_group_size;
  }
#if defined(USE_NCCL) || defined(USE_HOROVOD)
  // this condition block is temporary.
  // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
  // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
  bool use_nccl = parameters.allreduce_post_accumulation;
  if (!use_nccl && parameters.world_size > 1) {
    auto mpi_context = training::setup_mpi();
    std::cout << "mpi_context.world_rank: " << mpi_context.world_rank << std::endl;
    std::cout << "mpi_context.local_rank: " << mpi_context.local_rank << std::endl;
    std::cout << "mpi_context.world_size: " << mpi_context.world_size << std::endl;
    std::cout << "mpi_context.local_size: " << mpi_context.local_size << std::endl;
    parameters.local_size = mpi_context.local_size;
    parameters.local_rank = mpi_context.local_rank;
  }
#endif

  training::TrainingSession::TrainingConfiguration config{};
  config.weight_names_to_train = parameters.weights_to_train;
  config.weight_names_to_not_train = parameters.weights_not_to_train;
  config.immutable_weights = parameters.immutable_weights;

  config.gradient_accumulation_steps = parameters.gradient_accumulation_steps;

  config.distributed_config.world_rank = parameters.world_rank;
  config.distributed_config.world_size = parameters.world_size;
  config.distributed_config.local_rank = parameters.local_rank;
  config.distributed_config.local_size = parameters.local_size;
  config.distributed_config.data_parallel_size = parameters.data_parallel_size;
  config.distributed_config.horizontal_parallel_size = parameters.horizontal_parallel_size;

  if (parameters.use_mixed_precision) {
    training::TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_fp16_initializers = true;

    config.mixed_precision_config = mp;
  }

  config.loss_name = parameters.loss_output_name;

  if (!parameters.training_optimizer_name.empty()) {
    training::TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
    opt.name = parameters.training_optimizer_name;
    opt.learning_rate_input_name = parameters.lr_params_feed_name;
    opt.weight_attributes_generator = [&parameters](const std::string& weight_name) {
      const auto it = parameters.optimizer_attributes_map.find(weight_name);
      ORT_ENFORCE(
          it != parameters.optimizer_attributes_map.end(),
          "Failed to find attribute map for weight ", weight_name);
      return it->second;
    };
    opt.weight_int_attributes_generator = [&parameters](const std::string& weight_name) {
      const auto it = parameters.optimizer_int_attributes_map.find(weight_name);
      ORT_ENFORCE(
          it != parameters.optimizer_int_attributes_map.end(),
          "Failed to find int attribute map for weight ", weight_name);
      return it->second;
    };
    opt.use_fp16_moments = parameters.use_fp16_moments;
    opt.do_all_reduce_in_fp16 = true;
    // TODO: this mapping is temporary.
    // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
    // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
    // eventually we will have one all reduce kernel and let opt to have
    // an allreduce_post_accumulation option and remove the use_nccl option.
    opt.use_nccl = parameters.allreduce_post_accumulation;
    opt.deepspeed_zero = onnxruntime::training::ZeROConfig(parameters.deepspeed_zero_stage);
    // TODO: The norm clipping value is 1.0f which is the default used in most frameworks.
    // Need to have another option to support more values in the future.
    opt.enable_grad_norm_clip = parameters.enable_grad_norm_clip;

    config.optimizer_config = opt;
  }

  config.gradient_graph_config.use_invertible_layernorm_grad = parameters.use_invertible_layernorm_grad;
  config.gradient_graph_config.set_gradients_as_graph_outputs = parameters.set_gradients_as_graph_outputs;

  training::TrainingSession::TrainingConfigurationResult config_result{};

  OrtPybindThrowIfError(sess->ConfigureForTraining(config, config_result));

  TrainingConfigurationResult python_config_result{};
  if (config_result.mixed_precision_config_result.has_value()) {
    const auto& mp_config_result = config_result.mixed_precision_config_result.value();
    python_config_result.loss_scale_input_name = mp_config_result.loss_scale_input_name;
  }

  return python_config_result;
}

void addObjectMethodsForTraining(py::module& m) {
  py::class_<TrainingParameters> parameters(m, "TrainingParameters", R"pbdoc(Configuration information for training.)pbdoc");
  parameters.def(py::init())
      .def_readwrite("loss_output_name", &TrainingParameters::loss_output_name)
      .def_readwrite("immutable_weights", &TrainingParameters::immutable_weights)
      .def_readwrite("weights_not_to_train", &TrainingParameters::weights_not_to_train)
      .def_readwrite("weights_to_train", &TrainingParameters::weights_to_train)
      .def_readwrite("training_optimizer_name", &TrainingParameters::training_optimizer_name)
      .def_readwrite("lr_params_feed_name", &TrainingParameters::lr_params_feed_name)
      .def_readwrite("optimizer_attributes_map", &TrainingParameters::optimizer_attributes_map)
      .def_readwrite("optimizer_int_attributes_map", &TrainingParameters::optimizer_int_attributes_map)
      .def_readwrite("use_fp16_moments", &TrainingParameters::use_fp16_moments)
      .def_readwrite("use_mixed_precision", &TrainingParameters::use_mixed_precision)
      .def_readwrite("allreduce_post_accumulation", &TrainingParameters::allreduce_post_accumulation)
      .def_readwrite("loss_scale", &TrainingParameters::loss_scale)
      .def_readwrite("world_rank", &TrainingParameters::world_rank)
      .def_readwrite("world_size", &TrainingParameters::world_size)
      .def_readwrite("gradient_accumulation_steps", &TrainingParameters::gradient_accumulation_steps)
      .def_readwrite("deepspeed_zero_stage", &TrainingParameters::deepspeed_zero_stage)
      .def_readwrite("enable_grad_norm_clip", &TrainingParameters::enable_grad_norm_clip)
      .def_readwrite("set_gradients_as_graph_outputs", &TrainingParameters::set_gradients_as_graph_outputs)
      .def_readwrite("use_invertible_layernorm_grad", &TrainingParameters::use_invertible_layernorm_grad);

  py::class_<TrainingConfigurationResult> config_result(m, "TrainingConfigurationResult", "pbdoc(Configuration result for training.)pbdoc");
  config_result.def(py::init())
      .def_property_readonly("loss_scale_input_name", [](const TrainingConfigurationResult& result) -> py::object {
        if (result.loss_scale_input_name.has_value()) {
          return py::str{result.loss_scale_input_name.value()};
        }
        return py::none();
      });

  py::class_<onnxruntime::training::TrainingSession, InferenceSession> training_session(m, "TrainingSession");
  training_session.def(py::init([](const SessionOptions& so) {
                    Environment& env = get_env();
                    return onnxruntime::make_unique<onnxruntime::training::TrainingSession>(so, env);
                  }))
      .def(py::init([]() {
        Environment& env = get_env();
        return onnxruntime::make_unique<onnxruntime::training::TrainingSession>(GetDefaultCPUSessionOptions(), env);
      }))
      .def("finalize", [](py::object) {
#if defined(USE_NCCL) || defined(USE_HOROVOD)
        training::shutdown_mpi();
#endif
      })
      .def("load_model", [](onnxruntime::training::TrainingSession* sess, const std::string& path, TrainingParameters& parameters) {
        OrtPybindThrowIfError(sess->Load(path));

        const auto config_result = ConfigureSessionForTraining(sess, parameters);

        std::vector<std::string> provider_types = {};
        InitializeSession(sess, provider_types);

        return config_result;
      })
      .def("read_bytes", [](onnxruntime::training::TrainingSession* sess, const py::bytes& serialized_model, TrainingParameters& parameters) {
        std::istringstream buffer(serialized_model);
        OrtPybindThrowIfError(sess->Load(buffer));

        const auto config_result = ConfigureSessionForTraining(sess, parameters);

        std::vector<std::string> provider_types = {};
        InitializeSession(sess, provider_types);

        return config_result;
      })
      .def("get_state", [](onnxruntime::training::TrainingSession* sess) {
        NameMLValMap state_tensors;
        ORT_THROW_IF_ERROR(sess->GetStateTensors(state_tensors));
        auto& data_transfer_manager = sess->GetDataTransferManager();
        //convert to numpy array
        std::map<std::string, py::object> rmap;
        for (auto& kv : state_tensors) {
          if (kv.second.IsTensor()) {
            py::object obj;
            const Tensor& rtensor = kv.second.Get<Tensor>();
            GetPyObjFromTensor(rtensor, obj, &data_transfer_manager);
            rmap.insert({kv.first, obj});
          } else {
            throw std::runtime_error("Non tensor type in session state tensors is not expected.");
          }
        }
        return rmap;
      })
      .def("load_state", [](onnxruntime::training::TrainingSession* sess, std::unordered_map<std::string, py::object>& state, bool strict) {
        NameMLValMap state_tensors;
        for (auto initializer : state) {
          OrtValue ml_value;
          auto px = sess->GetModelInputs();
          if (!px.first.IsOK() || !px.second) {
            throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
          }
          CreateGenericMLValue(px.second, GetAllocator(), initializer.first, initializer.second, &ml_value);
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
          state_tensors.insert(std::make_pair(initializer.first, ml_value));
        }
        ORT_THROW_IF_ERROR(sess->SetStateTensors(state_tensors, strict));
      })
      .def("is_output_fp32_node", [](onnxruntime::training::TrainingSession* sess, const std::string& output_name) {
        return sess->IsGraphOutputFp32Node(output_name);
      });
}
}  // namespace python
}  // namespace onnxruntime
