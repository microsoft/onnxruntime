// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_state_common.h"

// pybind11/stl.h is needed to support std::unordered_set, etc.
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#ifdef ENABLE_TRAINING_APIS
#include <google/protobuf/io/zero_copy_stream_impl.h>
#endif

#include "core/common/parse_string.h"
#include "core/framework/customregistry.h"
#include "core/graph/model.h"
#include "core/session/environment.h"
#include "core/session/custom_ops.h"
#include "core/dlpack/dlpack_converter.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/agent/training_agent.h"
#include "orttraining/core/graph/gradient_config.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "orttraining/core/graph/gradient_definition_registry.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "orttraining/python/orttraining_pybind_common.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

#include "core/framework/stream_execution_context.h"

#ifdef ENABLE_TRAINING_TORCH_INTEROP
#include "orttraining/core/framework/torch/custom_function_register.h"
#endif

#ifdef ENABLE_TRITON
#include "orttraining/core/framework/triton/triton_op_executor.h"
#endif

#ifdef ENABLE_TRAINING_APIS
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/lr_scheduler.h"
#endif

PYBIND11_MAKE_OPAQUE(onnxruntime::OrtValueCache);

namespace onnxruntime {
namespace python {
namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;
using namespace onnxruntime::training;

ORTTrainingPythonEnv& GetTrainingEnv();

void ResolveExtraProviderOptions(const std::vector<std::string>& provider_types,
                                 const ProviderOptionsVector& original_provider_options_vector,
                                 ProviderOptionsVector& merged_options) {
  auto& training_env = GetTrainingEnv();
  std::size_t j = 0;  // index for provider_options_vector
  for (const std::string& type : provider_types) {
    auto it = training_env.ext_execution_provider_info_map_.find(type);
    if (it == training_env.ext_execution_provider_info_map_.end()) {
      if (j < original_provider_options_vector.size() && !original_provider_options_vector[j].empty()) {
        merged_options.push_back(original_provider_options_vector[j]);
      }
    } else {
      ProviderOptions options = it->second.second;
      options.insert({kExecutionProviderSharedLibraryPath, it->second.first});
      if (j < original_provider_options_vector.size() && !original_provider_options_vector[j].empty()) {
        for (auto [k, v] : original_provider_options_vector[j]) {
          options.insert({k, v});
        }
      }
      merged_options.push_back(options);
    }

    j += 1;
  }
}
#ifdef ENABLE_TRAINING_APIS
namespace {
// This function is used to create an execution provider to be passed to Module and Optimizer.
std::vector<std::shared_ptr<IExecutionProvider>>
GetExecutionProvidersForTrainingApis(OrtDevice device) {
  std::vector<std::shared_ptr<IExecutionProvider>> provider;

#ifdef USE_CUDA
  if (device.Type() == OrtDevice::GPU) {
    OrtCUDAProviderOptions provider_options{};
    provider_options.device_id = device.Id();

    if (auto factory = CudaProviderFactoryCreator::Create(&provider_options))
      provider.push_back(factory->CreateProvider());

    return provider;
  }
#endif
  if (device.Type() == OrtDevice::CPU) {
    provider = std::vector<std::shared_ptr<IExecutionProvider>>();
  } else {
    ORT_THROW("Unsupported device type: ", device.Type());
  }
  return provider;
}
}  // namespace
#endif
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
  onnxruntime::training::TrainingSession::OptimizerState optimizer_initial_state;
  std::unordered_map<std::string, std::vector<int>> sliced_schema;
  std::unordered_map<std::string, int> sliced_axes;
  std::vector<std::string> sliced_tensor_names;
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
  int pipeline_parallel_size = 1;
  int num_pipeline_micro_batches = 1;
  int deepspeed_zero_stage = 0;
  bool enable_grad_norm_clip = true;
  bool set_gradients_as_graph_outputs = false;
  bool use_memory_efficient_gradient = false;

  std::string pipeline_cut_info_string = {};

  // recompute
  bool attn_dropout_recompute = false;
  bool gelu_recompute = false;
  bool transformer_layer_recompute = false;
  int number_recompute_layers = 0;
  bool enable_adasum = false;

  // transformation
  int propagate_cast_ops_level = 1;
  std::vector<std::string> propagate_cast_ops_allow;
  GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy propagate_cast_ops_strategy =
      GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill;

  // graph dumping
  std::string model_after_graph_transforms_path;
  std::string model_with_gradient_graph_path;
  std::string model_with_training_graph_path;
};

struct TrainingConfigurationResult {
  optional<std::string> loss_scale_input_name;
};

#ifdef ENABLE_TRAINING_APIS
// Thin wrapper over internal C++ Optimizer
struct PyOptimizer {
  PyOptimizer(const std::string optimizer_model_uri, onnxruntime::training::api::CheckpointState* state,
              std::vector<std::shared_ptr<IExecutionProvider>> providers, PySessionOptions* session_options)
      : optimizer_() {
    auto model_identifiers = onnxruntime::training::api::ModelIdentifiers("", std::nullopt, optimizer_model_uri);
    auto env = GetTrainingEnv().GetORTEnv();
    // XXX: We hope that env will be around when optimizer needs it.
    optimizer_ = std::make_shared<onnxruntime::training::api::Optimizer>(
        model_identifiers, state, session_options->value, *env, providers, session_options->custom_op_domains_);
  }

  std::shared_ptr<onnxruntime::training::api::Optimizer> optimizer_;
};
#endif

struct PyGradientGraphBuilderContext {
  std::unique_ptr<GradientGraphBuilder> builder_;
  std::shared_ptr<Model> model_;
  std::unique_ptr<logging::Logger> logger_;
  std::unique_ptr<GradientGraphConfiguration> gradient_graph_config_;
  std::shared_ptr<CustomRegistry> custom_registry_;
  IOnnxRuntimeOpSchemaRegistryList local_registries_;
  PyGradientGraphBuilderContext(std::unique_ptr<GradientGraphBuilder> builder,
                                std::shared_ptr<Model> model,
                                std::unique_ptr<logging::Logger> logger,
                                std::unique_ptr<GradientGraphConfiguration> gradient_graph_config,
                                std::shared_ptr<CustomRegistry> custom_registry,
                                IOnnxRuntimeOpSchemaRegistryList local_registries)
      : builder_(std::move(builder)),
        model_(model),
        logger_(std::move(logger)),
        gradient_graph_config_(std::move(gradient_graph_config)),
        custom_registry_(custom_registry),
        local_registries_(local_registries) {}
};

// TODO: this method does not handle parallel optimization.
TrainingConfigurationResult ConfigureSessionForTraining(
    training::PipelineTrainingSession* sess, TrainingParameters& parameters) {
  // TODO tix, refactor the mpi related code to populate all fields correctly by default.
  ORT_ENFORCE(parameters.data_parallel_size <= parameters.world_size, "data_parallel_size: ", parameters.data_parallel_size, ", world_size: ", parameters.world_size);
  ORT_ENFORCE(parameters.horizontal_parallel_size <= parameters.world_size, "horizontal_parallel_size: ", parameters.horizontal_parallel_size, ", world_size: ", parameters.world_size);
  ORT_ENFORCE(parameters.pipeline_parallel_size <= parameters.world_size, "pipeline_parallel_size: ", parameters.pipeline_parallel_size, ", world_size: ", parameters.world_size);

  // When DxHxP != the total number of ranks, we try adjusting D so that DxHxP == the total number of ranks.
  if (parameters.world_size != parameters.data_parallel_size * parameters.horizontal_parallel_size * parameters.pipeline_parallel_size) {
    ORT_ENFORCE(parameters.world_size % parameters.horizontal_parallel_size * parameters.pipeline_parallel_size == 0,
                "D, H, P sizes are incorrect. To enable automatic correction, total number of ranks must be a divisible by HxP.");

    const auto new_data_parallel_size = parameters.world_size / (parameters.horizontal_parallel_size * parameters.pipeline_parallel_size);
    parameters.data_parallel_size = new_data_parallel_size;

    const std::string msg = "Cannot distribute " + std::to_string(parameters.world_size) + " ranks for distributed computation with D=" + std::to_string(parameters.data_parallel_size) +
                            ", H=" + std::to_string(parameters.horizontal_parallel_size) + ", P=" + std::to_string(parameters.pipeline_parallel_size) + ", so D is automatically changed to " + std::to_string(new_data_parallel_size);
    LOGS(*(sess->GetLogger()), WARNING) << msg;
  }

  training::PipelineTrainingSession::TrainingConfiguration config{};
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
  config.distributed_config.pipeline_parallel_size = parameters.pipeline_parallel_size;
  config.distributed_config.num_pipeline_micro_batches = parameters.num_pipeline_micro_batches;
  config.distributed_config.sliced_schema = parameters.sliced_schema;
  config.distributed_config.sliced_axes = parameters.sliced_axes;
  config.distributed_config.sliced_tensor_names = parameters.sliced_tensor_names;

  if (parameters.use_mixed_precision) {
    training::PipelineTrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_mixed_precision_initializers = true;

    config.mixed_precision_config = mp;
  }

  if (config.distributed_config.pipeline_parallel_size > 1) {
    training::PipelineTrainingSession::TrainingConfiguration::PipelineConfiguration pipeline_config;

    // Currently don't support auto-partition. User needs to pass in cut information for pipeline
    pipeline_config.do_partition = true;
    assert(!parameters.pipeline_cut_info_string.empty());

    auto process_with_delimiter = [](std::string& input_str, const std::string& delimiter) {
      std::vector<std::string> result;
      size_t pos = 0;
      while ((pos = input_str.find(delimiter)) != std::string::npos) {
        std::string token = input_str.substr(0, pos);
        result.emplace_back(token);
        input_str.erase(0, pos + delimiter.length());
      }
      // push the last split of substring into result.
      result.emplace_back(input_str);
      return result;
    };

    auto process_cut_info = [&](std::string& cut_info_string) {
      std::vector<PipelineTrainingSession::TrainingConfiguration::CutInfo> cut_list;
      const std::string group_delimiter = ",";
      const std::string edge_delimiter = ":";
      const std::string consumer_delimiter = "/";
      const std::string producer_consumer_delimiter = "-";

      auto cut_info_groups = process_with_delimiter(cut_info_string, group_delimiter);
      for (auto& cut_info_group : cut_info_groups) {
        PipelineTrainingSession::TrainingConfiguration::CutInfo cut_info;
        auto cut_edges = process_with_delimiter(cut_info_group, edge_delimiter);
        for (auto& cut_edge : cut_edges) {
          auto process_edge = process_with_delimiter(cut_edge, producer_consumer_delimiter);
          if (process_edge.size() == 1) {
            PipelineTrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0]};
            cut_info.emplace_back(edge);
          } else {
            ORT_ENFORCE(process_edge.size() == 2);
            auto consumer_list = process_with_delimiter(process_edge[1], consumer_delimiter);

            PipelineTrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0], consumer_list};
            cut_info.emplace_back(edge);
          }
        }
        cut_list.emplace_back(cut_info);
      }
      return cut_list;
    };

    pipeline_config.cut_list = process_cut_info(parameters.pipeline_cut_info_string);
    config.pipeline_config = pipeline_config;
  }
  config.loss_name = parameters.loss_output_name;

  if (!parameters.training_optimizer_name.empty()) {
    training::PipelineTrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
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
    opt.use_mixed_precision_moments = parameters.use_fp16_moments;
    opt.do_all_reduce_in_mixed_precision_type = true;
    // TODO: this mapping is temporary.
    // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
    // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
    // eventually we will have one all reduce kernel and let opt to have
    // an allreduce_post_accumulation option and remove the use_nccl option.
    opt.use_nccl = parameters.allreduce_post_accumulation;
    opt.deepspeed_zero = onnxruntime::training::ZeROConfig(parameters.deepspeed_zero_stage);
    opt.enable_grad_norm_clip = parameters.enable_grad_norm_clip;

    // TODO reduction types
    if (parameters.enable_adasum) {
#ifdef USE_CUDA
      opt.adasum_reduction_type = training::AdasumReductionType::GpuHierarchicalReduction;
#else
      opt.adasum_reduction_type = training::AdasumReductionType::CpuReduction;
#endif
    }

    config.optimizer_config = opt;
  }

  if (!parameters.optimizer_initial_state.empty()) {
    config.init_optimizer_states = parameters.optimizer_initial_state;
  }

  config.gradient_graph_config.use_memory_efficient_gradient = parameters.use_memory_efficient_gradient;
  config.gradient_graph_config.set_gradients_as_graph_outputs = parameters.set_gradients_as_graph_outputs;

  config.graph_transformer_config.attn_dropout_recompute = parameters.attn_dropout_recompute;
  config.graph_transformer_config.gelu_recompute = parameters.gelu_recompute;
  config.graph_transformer_config.transformer_layer_recompute = parameters.transformer_layer_recompute;
  config.graph_transformer_config.number_recompute_layers = parameters.number_recompute_layers;
  config.graph_transformer_config.propagate_cast_ops_config.strategy = parameters.propagate_cast_ops_strategy;
  config.graph_transformer_config.propagate_cast_ops_config.level = parameters.propagate_cast_ops_level;
  config.graph_transformer_config.propagate_cast_ops_config.allow = parameters.propagate_cast_ops_allow;

  if (!parameters.model_after_graph_transforms_path.empty()) {
    config.model_after_graph_transforms_path = ToPathString(parameters.model_after_graph_transforms_path);
  }
  if (!parameters.model_with_gradient_graph_path.empty()) {
    config.model_with_gradient_graph_path = ToPathString(parameters.model_with_gradient_graph_path);
  }
  if (!parameters.model_with_training_graph_path.empty()) {
    config.model_with_training_graph_path = ToPathString(parameters.model_with_training_graph_path);
  }

  training::PipelineTrainingSession::TrainingConfigurationResult config_result{};

  OrtPybindThrowIfError(sess->ConfigureForTraining(config, config_result));

  TrainingConfigurationResult python_config_result{};
  if (config_result.mixed_precision_config_result.has_value()) {
    const auto& mp_config_result = config_result.mixed_precision_config_result.value();
    python_config_result.loss_scale_input_name = mp_config_result.loss_scale_input_name;
  }

  return python_config_result;
}

#if defined(USE_MPI)
void CopyMPIContextToTrainingParameters(TrainingParameters& parameters, const logging::Logger* logger) {
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetWorldRank(): " << MPIContext::GetInstance().GetWorldRank();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetLocalRank(): " << MPIContext::GetInstance().GetLocalRank();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetWorldSize(): " << MPIContext::GetInstance().GetWorldSize();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetLocalSize(): " << MPIContext::GetInstance().GetLocalSize();

  parameters.local_rank = MPIContext::GetInstance().GetLocalRank();
  parameters.local_size = MPIContext::GetInstance().GetLocalSize();
  if (parameters.world_rank != MPIContext::GetInstance().GetWorldRank()) {
    if (parameters.world_rank != 0) {
      LOGS(*logger, WARNING) << "TrainingParameters world_rank is not correct, tuned automatically to " << MPIContext::GetInstance().GetWorldRank();
    }
    parameters.world_rank = MPIContext::GetInstance().GetWorldRank();
  }
  if (parameters.world_size != MPIContext::GetInstance().GetWorldSize()) {
    if (parameters.world_size != 1) {
      LOGS(*logger, WARNING) << "TrainingParameters world_size is not correct, tuned automatically to " << MPIContext::GetInstance().GetWorldSize();
    }
    parameters.world_size = MPIContext::GetInstance().GetWorldSize();
  }
}
#endif

std::unordered_map<std::string, std::unordered_map<std::string, py::object>> ConvertORTTensorMapToNumpy(std::unordered_map<std::string, NameMLValMap> c_tensor_state, const DataTransferManager& data_transfer_manager) {
  std::unordered_map<std::string, std::unordered_map<std::string, py::object>> py_tensor_state;
  for (const auto& layer1_item : c_tensor_state) {
    py_tensor_state[layer1_item.first] = {};
    for (const auto& layer2_item : layer1_item.second) {
      assert(layer2_item.second.IsTensor());
      py::object obj;
      const Tensor& rtensor = layer2_item.second.Get<Tensor>();
      GetPyObjFromTensor(rtensor, obj, &data_transfer_manager);
      py_tensor_state[layer1_item.first].insert({layer2_item.first, obj});
    }
  }
  return py_tensor_state;
}

void addObjectMethodsForTraining(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn) {
  py::class_<OrtValueCache, OrtValueCachePtr>(m, "OrtValueCache")
      .def(py::init<>())
      .def("insert", [](const OrtValueCachePtr& cache_ptr, std::string node_arg_name, OrtValue& value) {
        cache_ptr->emplace(node_arg_name, value);
      })
      .def("keys", [](const OrtValueCachePtr& cache_ptr) {
        py::list keys;
        for (auto kv : *cache_ptr.get()) {
          keys.append(kv.first);
        }
        return keys;
      })
      .def("clear", [](const OrtValueCachePtr& cache_ptr) {
        cache_ptr->clear();
      })
      .def("count", [](const OrtValueCachePtr& cache_ptr, std::string node_arg_name) {
        return cache_ptr->count(node_arg_name);
      })
      .def("remove", [](const OrtValueCachePtr& cache_ptr, std::string node_arg_name) {
        const auto& num_entries_erased = cache_ptr->erase(node_arg_name);
        ORT_ENFORCE(num_entries_erased == 1, "NodeArg not found in cache: ", node_arg_name);
      });

  py::class_<TrainingParameters> parameters(m, "TrainingParameters", R"pbdoc(Configuration information for training.)pbdoc");
  parameters.def(py::init())
      .def_readwrite("loss_output_name", &TrainingParameters::loss_output_name)
      .def_readwrite("immutable_weights", &TrainingParameters::immutable_weights)
      .def_readwrite("weights_not_to_train", &TrainingParameters::weights_not_to_train)
      .def_readwrite("weights_to_train", &TrainingParameters::weights_to_train)
      .def_readwrite("sliced_tensor_names", &TrainingParameters::sliced_tensor_names)
      .def_readwrite("training_optimizer_name", &TrainingParameters::training_optimizer_name)
      .def_readwrite("lr_params_feed_name", &TrainingParameters::lr_params_feed_name)
      .def_readwrite("optimizer_attributes_map", &TrainingParameters::optimizer_attributes_map)
      .def_readwrite("optimizer_int_attributes_map", &TrainingParameters::optimizer_int_attributes_map)
      .def_readwrite("sliced_schema", &TrainingParameters::sliced_schema)
      .def_readwrite("sliced_axes", &TrainingParameters::sliced_axes)
      .def_readwrite("use_fp16_moments", &TrainingParameters::use_fp16_moments)
      .def_readwrite("use_mixed_precision", &TrainingParameters::use_mixed_precision)
      .def_readwrite("allreduce_post_accumulation", &TrainingParameters::allreduce_post_accumulation)
      .def_readwrite("loss_scale", &TrainingParameters::loss_scale)
      .def_readwrite("world_rank", &TrainingParameters::world_rank)
      .def_readwrite("world_size", &TrainingParameters::world_size)
      .def_readwrite("data_parallel_size", &TrainingParameters::data_parallel_size)
      .def_readwrite("horizontal_parallel_size", &TrainingParameters::horizontal_parallel_size)
      .def_readwrite("pipeline_parallel_size", &TrainingParameters::pipeline_parallel_size)
      .def_readwrite("pipeline_cut_info_string", &TrainingParameters::pipeline_cut_info_string)
      .def_readwrite("num_pipeline_micro_batches", &TrainingParameters::num_pipeline_micro_batches)
      .def_readwrite("gradient_accumulation_steps", &TrainingParameters::gradient_accumulation_steps)
      .def_readwrite("deepspeed_zero_stage", &TrainingParameters::deepspeed_zero_stage)
      .def_readwrite("enable_grad_norm_clip", &TrainingParameters::enable_grad_norm_clip)
      .def_readwrite("set_gradients_as_graph_outputs", &TrainingParameters::set_gradients_as_graph_outputs)
      .def_readwrite("use_memory_efficient_gradient", &TrainingParameters::use_memory_efficient_gradient)
      .def_readwrite("attn_dropout_recompute", &TrainingParameters::attn_dropout_recompute)
      .def_readwrite("gelu_recompute", &TrainingParameters::gelu_recompute)
      .def_readwrite("transformer_layer_recompute", &TrainingParameters::transformer_layer_recompute)
      .def_readwrite("number_recompute_layers", &TrainingParameters::number_recompute_layers)
      .def_readwrite("data_parallel_size", &TrainingParameters::data_parallel_size)
      .def_readwrite("horizontal_parallel_size", &TrainingParameters::horizontal_parallel_size)
      .def_readwrite("pipeline_parallel_size", &TrainingParameters::pipeline_parallel_size)
      .def("set_optimizer_initial_state",
           [](TrainingParameters& parameters, const std::unordered_map<std::string, std::unordered_map<std::string, py::object>>& py_state) -> void {
             onnxruntime::training::TrainingSession::OptimizerState optim_state;
             for (const auto& weight_it : py_state) {
               auto state = weight_it.second;
               NameMLValMap state_tensors;
               for (auto& initializer : state) {
                 OrtValue ml_value;

                 // InputDeflist is null because parameters havent been tied to session yet
                 // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
                 CreateGenericMLValue(nullptr, GetAllocator(), "", initializer.second, &ml_value, true);
                 ThrowIfPyErrOccured();
                 state_tensors.emplace(initializer.first, ml_value);
               }
               optim_state.emplace(weight_it.first, state_tensors);
             }
             parameters.optimizer_initial_state = optim_state;
           })
      .def_readwrite("model_after_graph_transforms_path", &TrainingParameters::model_after_graph_transforms_path)
      .def_readwrite("model_with_gradient_graph_path", &TrainingParameters::model_with_gradient_graph_path)
      .def_readwrite("model_with_training_graph_path", &TrainingParameters::model_with_training_graph_path)
      .def_readwrite("enable_adasum", &TrainingParameters::enable_adasum)
      .def_readwrite("propagate_cast_ops_level", &TrainingParameters::propagate_cast_ops_level)
      .def_readwrite("propagate_cast_ops_allow", &TrainingParameters::propagate_cast_ops_allow);

#if defined(USE_MPI)
  m.def("get_mpi_context_local_rank", []() -> int { return MPIContext::GetInstance().GetLocalRank(); });
  m.def("get_mpi_context_local_size", []() -> int { return MPIContext::GetInstance().GetLocalSize(); });
  m.def("get_mpi_context_world_rank", []() -> int { return MPIContext::GetInstance().GetWorldRank(); });
  m.def("get_mpi_context_world_size", []() -> int { return MPIContext::GetInstance().GetWorldSize(); });
#endif

  m.def("register_forward_runner", [](py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    size_t function_address = py::cast<size_t>(obj);
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterForwardRunner(function_address);
#else
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("register_backward_runner", [](py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    size_t function_address = py::cast<size_t>(obj);
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterBackwardRunner(function_address);
#else
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("register_torch_autograd_function", [](std::string function_full_qual_name, py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterTorchAutogradFunction(function_full_qual_name, obj.ptr());
#else
        ORT_UNUSED_PARAMETER(function_full_qual_name);
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("register_shape_inference_function", [](std::string function_full_qual_name, py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterShapeInferenceFunction(function_full_qual_name, obj.ptr());
#else
        ORT_UNUSED_PARAMETER(function_full_qual_name);
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("get_shape_inference_function", [](std::string function_full_qual_name) -> py::object {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    auto py_object = pool.TryGettingShapeInferenceFunction(function_full_qual_name);
    if (py_object.has_value()) {
      Py_INCREF(py_object.value());
      return py::reinterpret_steal<py::object>(py_object.value());
    }
#else
        ORT_UNUSED_PARAMETER(function_full_qual_name);
#endif
    return py::none();
  });

  m.def("register_input_alias_function", [](std::string function_full_qual_name, py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterInputAliasFunction(function_full_qual_name, obj.ptr());
#else
        ORT_UNUSED_PARAMETER(function_full_qual_name);
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("register_miscellaneous_const_input", [](py::object obj) -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.RegisterMiscellaneousConstInput(obj.ptr());
#else
        ORT_UNUSED_PARAMETER(obj);
#endif
  });
  m.def("unregister_python_functions", []() -> void {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    // Release all custom python functions registered.
    auto& pool = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance();
    pool.UnRegisterFunctions();
#endif
  });
  m.def("is_torch_interop_default_on", []() -> bool {
#ifdef ENABLE_TRAINING_TORCH_INTEROP
    return true;
#else
        return false;
#endif
  });
  m.def("is_triton_enabled", []() -> bool {
#ifdef ENABLE_TRITON
    return true;
#else
        return false;
#endif
  });
#ifdef ENABLE_TRITON
  m.def("register_triton_op_executor",
        [](py::object config_getter, py::object executor_by_name, py::object executor_by_onnx) -> void {
          training::framework::triton::TritonOpExecutor::Instance().Initialize(
              config_getter.ptr(), executor_by_name.ptr(), executor_by_onnx.ptr());
        });
#endif

  py::class_<TrainingConfigurationResult> config_result(m, "TrainingConfigurationResult", "pbdoc(Configuration result for training.)pbdoc");
  config_result.def(py::init())
      .def_property_readonly("loss_scale_input_name", [](const TrainingConfigurationResult& result) -> py::object {
        if (result.loss_scale_input_name.has_value()) {
          return py::str{result.loss_scale_input_name.value()};
        }
        return py::none();
      });

  // Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
  struct PyTrainingSession : public PyInferenceSession {
    PyTrainingSession(std::shared_ptr<Environment> env, const PySessionOptions& so)
        : PyInferenceSession(env, std::make_unique<PipelineTrainingSession>(so.value, *env)) {
    }
    ~PyTrainingSession() = default;
  };

  py::class_<PyTrainingSession, PyInferenceSession> training_session(m, "TrainingSession");
  training_session
      .def(py::init([](const PySessionOptions& so) {
        auto& training_env = GetTrainingEnv();
        return std::make_unique<PyTrainingSession>(training_env.GetORTEnv(), so);
      }))
      .def(py::init([]() {
        auto& training_env = GetTrainingEnv();
        return std::make_unique<PyTrainingSession>(training_env.GetORTEnv(), GetDefaultCPUSessionOptions());
      }))
      .def("finalize", [](py::object) {
#if defined(USE_MPI)
#ifdef _WIN32
        // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
        // shutdown_mpi() is not called within MPIContext destructor because of DllMain's restriction
        // call shutdown_mpi() here instead.
        MPIContext::shutdown_mpi();
#endif
#endif
      })
      .def("load_model", [ep_registration_fn](PyTrainingSession* sess, const std::string& path, TrainingParameters& parameters, const std::vector<std::string>& provider_types, const ProviderOptionsVector& provider_options) {
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(path));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle()), parameters);

        ProviderOptionsVector merged_options;
        ResolveExtraProviderOptions(provider_types, provider_options, merged_options);

        InitializeSession(sess->GetSessionHandle(), ep_registration_fn, provider_types, merged_options);

        return config_result;
      })
      .def("read_bytes", [ep_registration_fn](PyTrainingSession* sess, const py::bytes& serialized_model, TrainingParameters& parameters, const std::vector<std::string>& provider_types, const ProviderOptionsVector& provider_options) {
        std::istringstream buffer(serialized_model);
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(buffer));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle()), parameters);
        ProviderOptionsVector merged_options;
        ResolveExtraProviderOptions(provider_types, provider_options, merged_options);

        InitializeSession(sess->GetSessionHandle(), ep_registration_fn, provider_types, merged_options);

        return config_result;
      })
      .def("get_state", [](PyTrainingSession* sess) {
        NameMLValMap state_tensors;
        ORT_THROW_IF_ERROR(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->GetStateTensors(state_tensors));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        // convert to numpy array
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
      .def("get_model_state", [](PyTrainingSession* sess, bool include_mixed_precision_weights) {
        std::unordered_map<std::string, NameMLValMap> model_state_tensors;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetModelState(model_state_tensors, include_mixed_precision_weights));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        return ConvertORTTensorMapToNumpy(model_state_tensors, data_transfer_manager);
      })
      .def("get_optimizer_state", [](PyTrainingSession* sess) {
        std::unordered_map<std::string, NameMLValMap> opt_state_tensors;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetOptimizerState(opt_state_tensors));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        return ConvertORTTensorMapToNumpy(opt_state_tensors, data_transfer_manager);
      })
      .def("get_partition_info_map", [](PyTrainingSession* sess) {
        std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> part_info_map;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetPartitionInfoMap(part_info_map));
        return part_info_map;
      })
      .def("load_state", [](PyTrainingSession* sess, std::unordered_map<std::string, py::object>& state, bool strict) {
        NameMLValMap state_tensors;
        for (auto initializer : state) {
          OrtValue ml_value;
          auto px = sess->GetSessionHandle()->GetModelInputs();
          if (!px.first.IsOK() || !px.second) {
            throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
          }
          CreateGenericMLValue(px.second, GetAllocator(), initializer.first, initializer.second, &ml_value);
          ThrowIfPyErrOccured();
          state_tensors.insert(std::make_pair(initializer.first, ml_value));
        }
        ORT_THROW_IF_ERROR(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->SetStateTensors(state_tensors, strict));
      })
      .def("is_output_fp32_node", [](PyTrainingSession* sess, const std::string& output_name) {
        return static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->IsGraphOutputFp32Node(output_name);
      });

  py::class_<PartialGraphExecutionState>(m, "PartialGraphExecutionState")
      .def(py::init([]() {
        return std::make_unique<PartialGraphExecutionState>();
      }));

  py::class_<TrainingAgent>(m, "TrainingAgent", R"pbdoc(This is the main class used to run a ORTModule model.)pbdoc")
      .def(py::init([](PyInferenceSession* session, const std::vector<std::string>& fw_feed_names,
                       const std::vector<OrtDevice>& fw_outputs_device_info,
                       const std::vector<std::string>& bw_fetches_names,
                       const std::vector<OrtDevice>& bw_outputs_device_info,
                       int local_rank) {
        return std::make_unique<TrainingAgent>(*session->GetSessionHandle(), fw_feed_names, fw_outputs_device_info,
                                               bw_fetches_names, bw_outputs_device_info, local_rank);
      }))
      .def("run_forward", [](TrainingAgent* agent, const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState* state, OrtValueCachePtr cache) -> void {
        Status status = agent->RunForward(feeds, fetches, *state, cache);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in forward pass execution: " + status.ErrorMessage());
        }
      })
      .def("run_backward", [](TrainingAgent* agent, const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState* state) -> void {
        Status status = agent->RunBackward(feeds, fetches, *state);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in backward pass execution: " + status.ErrorMessage());
        }
      });

  py::enum_<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(m, "PropagateCastOpsStrategy", py::module_local(), py::arithmetic{})
      .value("NONE", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::None)
      .value("INSERT_AND_REDUCE", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::InsertAndReduce)
      .value("FLOOD_FILL", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill)
      .def("__or__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator|))
      .def("__and__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                        GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator&))
      .def("__eq__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator==))
      .def("__neq__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                        GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator!=));

  py::class_<GraphTransformerConfiguration::PropagateCastOpsConfiguration>
      propagate_cast_ops_config(
          m, "PropagateCastOpsConfiguration",
          R"pbdoc(Propagate cast ops configuration.)pbdoc");
  propagate_cast_ops_config.def(py::init())
      .def_readwrite("strategy", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::strategy)
      .def_readwrite("level", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::level)
      .def_readwrite("allow", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::allow);

  py::class_<GraphTransformerConfiguration> graph_transformer_config(
      m, "GraphTransformerConfiguration",
      R"pbdoc(Graph transformer configuration.)pbdoc");
  graph_transformer_config.def(py::init())
      .def_readwrite("propagate_cast_ops_config", &GraphTransformerConfiguration::propagate_cast_ops_config);

  py::class_<TrainingGraphTransformerConfiguration, GraphTransformerConfiguration> training_graph_transformer_config(
      m, "TrainingGraphTransformerConfiguration",
      R"pbdoc(Training Graph transformer configuration.)pbdoc");
  training_graph_transformer_config.def(py::init())
      .def_readwrite("enable_gelu_approximation", &TrainingGraphTransformerConfiguration::enable_gelu_approximation)
      .def_readwrite("attn_dropout_recompute", &TrainingGraphTransformerConfiguration::attn_dropout_recompute)
      .def_readwrite("gelu_recompute", &TrainingGraphTransformerConfiguration::gelu_recompute)
      .def_readwrite("transformer_layer_recompute", &TrainingGraphTransformerConfiguration::transformer_layer_recompute)
      .def_readwrite("number_recompute_layers", &TrainingGraphTransformerConfiguration::number_recompute_layers)
      .def_readwrite("enable_compute_optimizer", &TrainingGraphTransformerConfiguration::enable_compute_optimizer)
      .def_readwrite("sparse_embedding_input_names", &TrainingGraphTransformerConfiguration::sparse_embedding_input_names)
      .def_readwrite("sparse_label_input_names", &TrainingGraphTransformerConfiguration::sparse_label_input_names)
      .def_readwrite("optimized_pre_grad_filepath", &TrainingGraphTransformerConfiguration::optimized_pre_grad_filepath)
      .def_readwrite("propagate_cast_ops_config", &TrainingGraphTransformerConfiguration::GraphTransformerConfiguration::propagate_cast_ops_config);

  py::class_<OrtModuleGraphBuilderConfiguration> module_graph_builder_config(
      m, "OrtModuleGraphBuilderConfiguration",
      R"pbdoc(Configuration information for module graph builder.)pbdoc");

  py::enum_<Severity>(m, "Severity", py::arithmetic(), py::module_local())
      .value("VERBOSE", logging::Severity::kVERBOSE)
      .value("INFO", logging::Severity::kINFO)
      .value("WARNING", logging::Severity::kWARNING)
      .value("ERROR", logging::Severity::kERROR)
      .value("FATAL", logging::Severity::kFATAL);

  module_graph_builder_config.def(py::init())
      .def_readwrite("initializer_names", &OrtModuleGraphBuilderConfiguration::initializer_names)
      .def_readwrite("initializer_names_to_train", &OrtModuleGraphBuilderConfiguration::initializer_names_to_train)
      .def_readwrite("input_names_require_grad", &OrtModuleGraphBuilderConfiguration::input_names_require_grad)
      .def_readwrite("use_memory_efficient_gradient",
                     &OrtModuleGraphBuilderConfiguration::use_memory_efficient_gradient)
      .def_readwrite("build_gradient_graph", &OrtModuleGraphBuilderConfiguration::build_gradient_graph)
      .def_readwrite("enable_caching", &OrtModuleGraphBuilderConfiguration::enable_caching)
      .def_readwrite("loglevel", &OrtModuleGraphBuilderConfiguration::loglevel);

  py::class_<GraphInfo> graph_info(m, "GraphInfo",
                                   R"pbdoc(The information of split graphs for frontend.)pbdoc");
  graph_info.def(py::init())
      .def_readwrite("user_input_names", &GraphInfo::user_input_names)
      .def_readwrite("user_input_grad_names", &GraphInfo::user_input_grad_names)
      .def_readwrite("initializer_names", &GraphInfo::initializer_names)
      .def_readwrite("initializer_names_to_train", &GraphInfo::initializer_names_to_train)
      .def_readwrite("initializer_grad_names_to_train", &GraphInfo::initializer_grad_names_to_train)
      .def_readwrite("user_output_names", &GraphInfo::user_output_names)
      .def_readwrite("output_grad_indices_non_differentiable", &GraphInfo::output_grad_indices_non_differentiable)
      .def_readwrite("output_grad_indices_require_full_shape", &GraphInfo::output_grad_indices_require_full_shape)
      .def_readwrite("module_output_indices_requires_save_for_backward", &GraphInfo::module_output_indices_requires_save_for_backward)
      .def_readwrite("frontier_node_arg_map", &GraphInfo::frontier_node_arg_map)
      .def_readwrite("cached_node_arg_names", &GraphInfo::cached_node_arg_names)
      .def_readwrite("module_output_gradient_name", &GraphInfo::module_output_gradient_name);

  py::class_<OrtModuleGraphBuilder> ortmodule_graph_builder(m, "OrtModuleGraphBuilder");
  ortmodule_graph_builder.def(py::init([]() { return std::make_unique<OrtModuleGraphBuilder>(); }))
      .def("initialize",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder, const py::bytes& serialized_model,
              const OrtModuleGraphBuilderConfiguration& config) {
             std::istringstream buffer(serialized_model);
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Initialize(buffer, config));
           })
      .def("build",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder,
              const TrainingGraphTransformerConfiguration& config) {
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Build(config));
           })
      .def("build",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder,
              const TrainingGraphTransformerConfiguration& config,
              const std::vector<std::vector<int64_t>>& input_shapes) {
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Build(config, &input_shapes));
           })
      .def("get_gradient_model",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
             return py::bytes(ortmodule_graph_builder->GetGradientModel());
           })
      .def("get_forward_model",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
             return py::bytes(ortmodule_graph_builder->GetForwardModel());
           })
      .def("get_graph_info", [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
        return ortmodule_graph_builder->GetGraphInfo();
      });

  // Provide a convenient and well-documented way to make a gradient graph.
  // It's possible to get the gradient graph through ORTModule by leveraging some "private" fields and not-so-well-documented APIs, so we provide this explicit and tested way to get the gradient graph.
  py::class_<PyGradientGraphBuilderContext> gradient_graph_builder(
      m, "GradientGraphBuilder", R"pbdoc(A utility for making a gradient graph that can be used to help train a model.)pbdoc");
  // Set up methods to match the C++ `GradientGraphBuilder` interface.
  gradient_graph_builder
      .def(py::init([](const py::bytes& serialized_model,
                       const std::unordered_set<std::string>& y_node_arg_names,
                       const std::unordered_set<std::string>& x_node_arg_names,
                       const std::string loss_node_arg_name,
                       PySessionOptions* options) {
             std::shared_ptr<CustomRegistry> custom_registry;
             IOnnxRuntimeOpSchemaRegistryList local_registries;
             if (options && !options->custom_op_domains_.empty()) {
               // Register all custom op domains that will be needed for the session
               ORT_THROW_IF_ERROR(onnxruntime::CreateCustomRegistry(options->custom_op_domains_, custom_registry));
               local_registries.push_back(custom_registry->GetOpschemaRegistry());
             }

             std::shared_ptr<Model> model;
             auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
             logging::Severity severity = logging::Severity::kINFO;
             if (options && options->value.session_log_severity_level >= 0) {
               severity = static_cast<logging::Severity>(options->value.session_log_severity_level);
             }
             logger_ptr->SetSeverity(severity);
             ONNX_NAMESPACE::ModelProto model_proto;
             std::istringstream model_istream(serialized_model);
             ORT_THROW_IF_ERROR(Model::Load(model_istream, &model_proto));
             ORT_THROW_IF_ERROR(Model::Load(model_proto, model,
                                            local_registries.empty() ? nullptr : &local_registries,
                                            *logger_ptr));
             GradientGraphConfiguration gradient_graph_config{};
             gradient_graph_config.set_gradients_as_graph_outputs = true;
             // Save some objects, otherwise they get lost.
             auto gradient_graph_config_ptr = std::make_unique<GradientGraphConfiguration>(gradient_graph_config);

             auto builder = std::make_unique<GradientGraphBuilder>(
                 &model->MainGraph(),
                 y_node_arg_names,
                 x_node_arg_names,
                 loss_node_arg_name,
                 *gradient_graph_config_ptr,
                 *logger_ptr);

             return std::make_unique<PyGradientGraphBuilderContext>(std::move(builder), std::move(model),
                                                                    std::move(logger_ptr), std::move(gradient_graph_config_ptr),
                                                                    custom_registry, std::move(local_registries));
           }),
           py::arg("serialized_model"), py::arg("y_node_arg_names"),
           py::arg("x_node_arg_names"), py::arg("loss_node_arg_name"),
           py::arg("options") = nullptr)
      .def("build", [](PyGradientGraphBuilderContext* gradient_graph_builder) {
        ORT_THROW_IF_ERROR(gradient_graph_builder->builder_->Build());
      })
      .def("save", [](PyGradientGraphBuilderContext* gradient_graph_builder, const std::string& path) {
        ORT_THROW_IF_ERROR(Model::Save(*(gradient_graph_builder->model_), path));
      })
      .def("get_model", [](PyGradientGraphBuilderContext* gradient_graph_builder) {
        std::string model_str;
        gradient_graph_builder->model_->ToProto().SerializeToString(&model_str);
        return py::bytes(model_str);
      });

  py::class_<GradientNodeAttributeDefinition> gradient_node_attribute_definition(
      m, "GradientNodeAttributeDefinition", R"pbdoc(Attribute definition for gradient graph nodes.)pbdoc");

  gradient_node_attribute_definition.def(py::init())
      .def_readwrite("name", &GradientNodeAttributeDefinition::name)
      .def_readwrite("value_json", &GradientNodeAttributeDefinition::value_json)
      .def_readwrite("dtype", &GradientNodeAttributeDefinition::dtype)
      .def_readwrite("is_tensor", &GradientNodeAttributeDefinition::is_tensor);

  py::class_<GradientNodeDefinition> gradient_node_definition(m, "GradientNodeDefinition",
                                                              R"pbdoc(Definition for gradient graph nodes.)pbdoc");

  gradient_node_definition.def(py::init())
      .def_readwrite("op_type", &GradientNodeDefinition::op_type)
      .def_readwrite("domain", &GradientNodeDefinition::domain)
      .def_readwrite("inputs", &GradientNodeDefinition::inputs)
      .def_readwrite("outputs", &GradientNodeDefinition::outputs)
      .def_readwrite("attributes", &GradientNodeDefinition::attributes);

  m.def("register_gradient_definition",
        [](const std::string& key, const std::vector<GradientNodeDefinition>& gradient_def) -> void {
          GradientDefinitionRegistry::Instance().Register(key, gradient_def);
        });

  m.def("register_custom_stop_gradient_edges",
        [](const std::string& key, const std::unordered_set<size_t> edges) -> void {
          GradientDefinitionRegistry::Instance().SetStopGradientEdgesForNode(key, edges);
        });
#ifdef ENABLE_TRAINING_APIS
  py::class_<onnxruntime::training::api::Module> training_module(m, "Module", R"pbdoc(Training Module.)pbdoc");
  training_module
      .def(py::init([](const std::string& model_uri,
                       onnxruntime::training::api::CheckpointState* state,
                       std::optional<std::string> eval_model_uri,
                       OrtDevice device, PySessionOptions* session_options) {
        std::vector<std::shared_ptr<IExecutionProvider>> provider = GetExecutionProvidersForTrainingApis(device);
        auto env = GetTrainingEnv().GetORTEnv();
        auto model_identifiers = onnxruntime::training::api::ModelIdentifiers(model_uri, eval_model_uri, std::nullopt);
        return std::make_unique<onnxruntime::training::api::Module>(model_identifiers,
                                                                    state, session_options->value, *env, provider,
                                                                    session_options->custom_op_domains_);
      }))
      .def("train_step",
           [](onnxruntime::training::api::Module* model,
              const std::vector<py::object>& user_inputs, std::vector<OrtValue>& user_outputs) -> void {
             std::vector<OrtValue> feeds;
             const auto model_inputs_with_error = model->GetTrainingModelInputs();
             ORT_THROW_IF_ERROR(model_inputs_with_error.first);
             ORT_ENFORCE(model_inputs_with_error.second, "Training model graph inputs are not defined.");
             for (size_t idx = 0; idx < user_inputs.size(); ++idx) {
               auto& feed = user_inputs[idx];
               // No need to process 'None's sent in by the user
               // to feed Optional inputs in the graph.
               // We just won't include anything in the feed and ORT
               // will handle such implicit 'None's internally.
               if (!feed.is(py::none())) {
                 const auto feed_name = model->GetTrainingModelInputName(idx);
                 OrtValue ort_value;
                 CreateGenericMLValue(model_inputs_with_error.second, GetAllocator(), feed_name, feed, &ort_value);
                 ThrowIfPyErrOccured();
                 feeds.emplace_back(ort_value);
               }
             }
             ORT_THROW_IF_ERROR(model->TrainStep(feeds, user_outputs));
           })
      .def("train_step_with_ort_values",
           [](onnxruntime::training::api::Module* model,
              const std::vector<OrtValue>& user_inputs, std::vector<OrtValue>& user_outputs) -> void {
             ORT_THROW_IF_ERROR(model->TrainStep(user_inputs, user_outputs));
           })
      .def("eval_step",
           [](onnxruntime::training::api::Module* model,
              const std::vector<py::object>& user_inputs, std::vector<OrtValue>& user_outputs) -> void {
             std::vector<OrtValue> feeds;
             const auto model_inputs_with_error = model->GetEvalModelInputs();
             ORT_THROW_IF_ERROR(model_inputs_with_error.first);
             ORT_ENFORCE(model_inputs_with_error.second, "Eval model graph inputs are not defined.");
             for (size_t idx = 0; idx < user_inputs.size(); ++idx) {
               auto& feed = user_inputs[idx];
               // No need to process 'None's sent in by the user
               // to feed Optional inputs in the graph.
               // We just won't include anything in the feed and ORT
               // will handle such implicit 'None's internally.
               if (!feed.is(py::none())) {
                 const auto feed_name = model->GetEvalModelInputName(idx);
                 OrtValue ort_value;
                 CreateGenericMLValue(model_inputs_with_error.second, GetAllocator(), feed_name, feed, &ort_value);
                 ThrowIfPyErrOccured();
                 feeds.emplace_back(ort_value);
               }
             }
             ORT_THROW_IF_ERROR(model->EvalStep(feeds, user_outputs));
           })
      .def("eval_step_with_ort_values",
           [](onnxruntime::training::api::Module* model,
              const std::vector<OrtValue>& user_inputs, std::vector<OrtValue>& user_outputs) -> void {
             ORT_THROW_IF_ERROR(model->EvalStep(user_inputs, user_outputs));
           })
      .def("lazy_reset_grad",
           [](onnxruntime::training::api::Module* model) -> void {
             ORT_THROW_IF_ERROR(model->LazyResetGrad());
           })
      .def("copy_parameters_to_buffer",
           [](onnxruntime::training::api::Module* model, OrtValue& output, bool trainable_only) -> void {
             ORT_THROW_IF_ERROR(model->CopyParametersToBuffer(output, trainable_only));
           })
      .def("copy_buffer_to_parameters",
           [](onnxruntime::training::api::Module* model, OrtValue& input, bool trainable_only) -> void {
             ORT_THROW_IF_ERROR(model->CopyBufferToParameters(input, trainable_only));
           })
      .def("get_parameters_size",
           [](onnxruntime::training::api::Module* model, bool trainable_only) -> size_t {
             return model->GetParametersSize(trainable_only);
           })
      .def("export_model_for_inferencing",
           [](onnxruntime::training::api::Module* model, const std::string& inference_model_path,
              const std::vector<std::string>& graph_output_names) -> void {
             ORT_ENFORCE(model, "Received a nullptr for expected pointer to class training::api::Module");
             ORT_THROW_IF_ERROR(model->ExportModelForInferencing(inference_model_path,
                                                                 graph_output_names));
           })
      .def("input_names",
           [](onnxruntime::training::api::Module* model, const bool is_training) {
             auto count_method = [&model, is_training]() -> size_t {
               return is_training ? model->GetTrainingModelInputCount() : model->GetEvalModelInputCount();
             };

             auto name_method = [&model, is_training](const size_t index) -> std::string {
               return is_training ? model->GetTrainingModelInputName(index) : model->GetEvalModelInputName(index);
             };

             std::vector<std::string> names;
             for (size_t index = 0; index < count_method(); ++index) {
               names.push_back(name_method(index));
             }

             return names;
           })
      .def("output_names",
           [](onnxruntime::training::api::Module* model, const bool is_training) {
             auto count_method = [&model, is_training]() -> size_t {
               return is_training ? model->GetTrainingModelOutputCount() : model->GetEvalModelOutputCount();
             };

             auto name_method = [&model, is_training](const size_t index) -> std::string {
               return is_training ? model->GetTrainingModelOutputName(index) : model->GetEvalModelOutputName(index);
             };

             std::vector<std::string> names;
             for (size_t index = 0; index < count_method(); ++index) {
               names.push_back(name_method(index));
             }

             return names;
           });

  py::class_<onnxruntime::training::api::CheckpointState>
      checkpoint_state(m, "CheckpointState", R"pbdoc(CheckpointState.)pbdoc");
  checkpoint_state
      .def(py::init())
      .def("add_property",
           [](onnxruntime::training::api::CheckpointState* state,
              const std::string& property_name,
              const std::variant<int64_t, float, std::string>& property_value) {
             state->property_bag.AddProperty(property_name, property_value);
           })
      .def("get_property",
           [](onnxruntime::training::api::CheckpointState* state, const std::string& property_name) {
             return state->property_bag.GetProperty<onnxruntime::training::api::PropertyDataType>(property_name);
           })
      .def("has_property",
           [](onnxruntime::training::api::CheckpointState* state, const std::string& property_name) {
             return state->property_bag.HasProperty(property_name);
           })
      .def("copy_parameter_from",
           [](onnxruntime::training::api::CheckpointState* state,
              const std::string& parameter_name, OrtValue& value) -> void {
             auto it = state->module_checkpoint_state.named_parameters.find(parameter_name);
             if (it == state->module_checkpoint_state.named_parameters.end()) {
               ORT_THROW("Parameter with name ", parameter_name, " does not exist.");
             }
             ORT_THROW_IF_ERROR(it->second->CopyFrom(
                 state->module_checkpoint_state.train_session_data_transfer_mgr, value));
           })
      .def("get_parameter",
           [](onnxruntime::training::api::CheckpointState* state, const std::string& parameter_name) {
             auto it = state->module_checkpoint_state.named_parameters.find(parameter_name);
             if (it == state->module_checkpoint_state.named_parameters.end()) {
               ORT_THROW("Parameter with name ", parameter_name, " does not exist.");
             }
             return it->second;
           })
      .def("has_parameter",
           [](onnxruntime::training::api::CheckpointState* state, const std::string& parameter_name) {
             return state->module_checkpoint_state.named_parameters.count(parameter_name);
           })
      .def("parameter_names",
           [](onnxruntime::training::api::CheckpointState* state) {
             std::vector<std::string> names;
             for ([[maybe_unused]] auto& [name, value] : state->module_checkpoint_state.named_parameters) {
               names.push_back(name);
             }
             std::sort(names.begin(), names.end());
             return names;
           })
      .def("property_names",
           [](onnxruntime::training::api::CheckpointState* state) {
             std::vector<std::string> names;
             for ([[maybe_unused]] auto& [name, value] : state->property_bag) {
               names.push_back(name);
             }
             std::sort(names.begin(), names.end());
             return names;
           });

  py::class_<PyOptimizer>
      training_optimizer(m, "Optimizer", R"pbdoc(Training Optimizer.)pbdoc");
  training_optimizer
      .def(py::init([](const std::string optimizer_model_uri,
                       onnxruntime::training::api::CheckpointState* state,
                       OrtDevice device, PySessionOptions* session_options) {
        std::vector<std::shared_ptr<IExecutionProvider>> providers = GetExecutionProvidersForTrainingApis(device);

        return std::make_unique<PyOptimizer>(optimizer_model_uri, state, providers, session_options);
      }))
      .def("optimizer_step", [](PyOptimizer* optimizer) -> void {
        ORT_THROW_IF_ERROR(optimizer->optimizer_->Step());
      })
      .def("set_learning_rate", [](PyOptimizer* optimizer, float lr) -> void {
        ORT_THROW_IF_ERROR(optimizer->optimizer_->SetLearningRate(lr));
      })
      .def("get_learning_rate", [](PyOptimizer* optimizer) -> float {
        return optimizer->optimizer_->GetLearningRate();
      });
  py::class_<onnxruntime::training::api::LinearLRScheduler>
      lr_scheduler(m, "LinearLRScheduler", R"pbdoc(Learning Rate Scheduler.)pbdoc");
  lr_scheduler.def(py::init([](PyOptimizer* optimizer,
                               int64_t total_step_count,
                               int64_t warmup_step_count,
                               float initial_lr) {
                ORT_THROW_IF_ERROR(optimizer->optimizer_->SetInitialLearningRate(initial_lr));

                return std::make_unique<onnxruntime::training::api::LinearLRScheduler>(
                    optimizer->optimizer_, warmup_step_count, total_step_count);
              }))
      .def("scheduler_step", [](onnxruntime::training::api::LinearLRScheduler* scheduler) -> void {
        ORT_THROW_IF_ERROR(scheduler->Step());
      });

  py::class_<onnxruntime::training::api::Parameter,
             std::unique_ptr<onnxruntime::training::api::Parameter, py::nodelete>>
      parameter(m, "Parameter");
  parameter
      .def_property_readonly("name", &onnxruntime::training::api::Parameter::Name)
      .def_property_readonly("data", &onnxruntime::training::api::Parameter::Data)
      .def_property_readonly("grad", &onnxruntime::training::api::Parameter::Gradient)
      .def_property_readonly("requires_grad", &onnxruntime::training::api::Parameter::RequiresGrad)
      .def("copy_from",
           [](onnxruntime::training::api::Parameter* parameter,
              onnxruntime::training::api::CheckpointState* state,
              OrtValue& value) -> void {
             ORT_THROW_IF_ERROR(parameter->CopyFrom(state->module_checkpoint_state.train_session_data_transfer_mgr, value));
           });

  m.def(
      "save_checkpoint",
      [](const std::vector<py::bytes>& trainable_tensor_protos_pybytes,
         const std::vector<py::bytes>& non_trainable_tensor_protos_pybytes,
         const std::string& checkpoint_path) {
        std::vector<TensorProto> trainable_tensor_protos(trainable_tensor_protos_pybytes.size());
        std::vector<TensorProto> non_trainable_tensor_protos(non_trainable_tensor_protos_pybytes.size());

        auto parse_pybytes_to_tensor_proto =
            [](const std::vector<py::bytes>& tensor_protos_pybytes, std::vector<TensorProto>& tensor_protos) {
              for (size_t i = 0; i < tensor_protos_pybytes.size(); ++i) {
                std::istringstream tensor_proto_istream(tensor_protos_pybytes[i]);
                ORT_ENFORCE(tensor_proto_istream.good(), "Broken tensor proto istream to read.");
                google::protobuf::io::IstreamInputStream zero_copy_input(&tensor_proto_istream);
                const bool result =
                    tensor_protos[i].ParseFromZeroCopyStream(&zero_copy_input) && tensor_proto_istream.eof();
                ORT_ENFORCE(result, "Parse tensor proto failed.");
              }
            };

        parse_pybytes_to_tensor_proto(trainable_tensor_protos_pybytes, trainable_tensor_protos);
        parse_pybytes_to_tensor_proto(non_trainable_tensor_protos_pybytes, non_trainable_tensor_protos);

        ORT_THROW_IF_ERROR(onnxruntime::training::api::SaveCheckpoint(trainable_tensor_protos,
                                                                      non_trainable_tensor_protos,
                                                                      ToPathString(checkpoint_path)));
      });

  m.def("save_checkpoint",
        [](onnxruntime::training::api::CheckpointState* checkpoint_state,
           const std::string& checkpoint_path, const bool include_optimizer_state) -> void {
          ORT_THROW_IF_ERROR(
              onnxruntime::training::api::SaveCheckpoint(*checkpoint_state, ToPathString(checkpoint_path),
                                                         include_optimizer_state));
        });

  m.def("load_checkpoint",
        [](const std::string& checkpoint_path) -> onnxruntime::training::api::CheckpointState {
          onnxruntime::training::api::CheckpointState state;
          ORT_THROW_IF_ERROR(
              onnxruntime::training::api::LoadCheckpoint(ToPathString(checkpoint_path), state));
          return state;
        });

  m.def("get_model_after_loading_checkpoint",
        [](const std::string& checkpoint_path, const py::bytes& serialized_model) {
          ONNX_NAMESPACE::ModelProto model_proto;

          std::istringstream buffer(serialized_model);
          ORT_THROW_IF_ERROR(Model::Load(buffer, &model_proto));
          ORT_THROW_IF_ERROR(
              onnxruntime::training::api::LoadCheckpointToModel(ToPathString(checkpoint_path), model_proto));

          std::string model_proto_str;
          ORT_ENFORCE(model_proto.SerializeToString(&model_proto_str), "Serializing Model failed.");

          return py::bytes(model_proto_str);
        });

  m.def("get_optimized_model",
        [](const py::bytes& serialized_model,
           const std::unordered_set<std::string>& graph_entities_that_require_gradients,
           PySessionOptions* options = nullptr) {
          std::shared_ptr<CustomRegistry> custom_registry;
          IOnnxRuntimeOpSchemaRegistryList local_registries;
          if (options && !options->custom_op_domains_.empty()) {
            // Register all custom op domains that will be needed for the session
            ORT_THROW_IF_ERROR(onnxruntime::CreateCustomRegistry(options->custom_op_domains_, custom_registry));
            local_registries.push_back(custom_registry->GetOpschemaRegistry());
          }

          // Load the serialized model
          std::istringstream buffer(serialized_model);
          ONNX_NAMESPACE::ModelProto model_proto;
          ORT_THROW_IF_ERROR(Model::Load(buffer, &model_proto));

          // Get the ort model from ModelProto model
          auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
          logging::Severity severity = logging::Severity::kINFO;
          if (options && options->value.session_log_severity_level >= 0) {
            severity = static_cast<logging::Severity>(options->value.session_log_severity_level);
          }
          logger_ptr->SetSeverity(severity);
          std::shared_ptr<onnxruntime::Model> ort_model;
          ORT_THROW_IF_ERROR(Model::Load(model_proto, ort_model,
                                         local_registries.empty() ? nullptr : &local_registries,
                                         *logger_ptr));

          Graph& graph = ort_model->MainGraph();
          ORT_THROW_IF_ERROR(graph.Resolve());

          // Register the pretraining graph transformations so that they are run twice
          constexpr size_t NumSteps = 2;
          GraphTransformerManager graph_transformation_mgr{NumSteps};
          std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
              std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

          const auto add_transformers = [&cpu_execution_provider,
                                         &graph_transformation_mgr,
                                         &graph_entities_that_require_gradients](TransformerLevel level) {
            auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
                level, graph_entities_that_require_gradients, TrainingGraphTransformerConfiguration(),
                *cpu_execution_provider);
            for (auto& entry : transformers_to_register) {
              ORT_THROW_IF_ERROR(graph_transformation_mgr.Register(std::move(entry), level));
            }
            return Status::OK();
          };

          for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
            TransformerLevel level = static_cast<TransformerLevel>(i);
            if (TransformerLevel::MaxLevel >= level) {
              ORT_THROW_IF_ERROR(add_transformers(level));
            }
          }

          // Run the graph transformations
          for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
            ORT_THROW_IF_ERROR(
                graph_transformation_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *logger_ptr));
          }

          // Return the optimized model.
          std::string model_str;
          ort_model->ToProto().SerializeToString(&model_str);
          return py::bytes(model_str);
        });

  m.def("is_ortmodule_available",
        []() {
#ifdef __linux__
          return true;
#else
        return false;
#endif
        });
#endif
}

}  // namespace python
}  // namespace onnxruntime
