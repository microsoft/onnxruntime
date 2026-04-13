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
#include "python/onnxruntime_pybind_mlvalue.h"
#include "orttraining/python/orttraining_pybind_common.h"

#include "core/framework/stream_execution_context.h"

#ifdef ENABLE_TRITON
#include "orttraining/core/framework/triton/triton_op_executor.h"
#endif

#ifdef ENABLE_TRAINING_APIS
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/lr_scheduler.h"
#endif

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
#ifdef ENABLE_TRAINING_APIS
// Thin wrapper over internal C++ Optimizer
struct PyOptimizer {
  PyOptimizer(const std::string optimizer_model_uri, onnxruntime::training::api::CheckpointState* state,
              std::vector<std::shared_ptr<IExecutionProvider>> providers, PySessionOptions* session_options)
      : optimizer_() {
    auto model_identifiers = onnxruntime::training::api::ModelIdentifiers("", std::nullopt, optimizer_model_uri);
    // XXX: We hope that env will be around when optimizer needs it.
    optimizer_ = std::make_shared<onnxruntime::training::api::Optimizer>(
        model_identifiers, state, session_options->value, GetTrainingEnv().GetORTEnv().GetEnvironment(), providers, session_options->custom_op_domains_);
  }

  std::shared_ptr<onnxruntime::training::api::Optimizer> optimizer_;
};
#endif

void addObjectMethodsForTraining(py::module& m) {
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

  py::enum_<Severity>(m, "Severity", py::arithmetic(), py::module_local())
      .value("VERBOSE", logging::Severity::kVERBOSE)
      .value("INFO", logging::Severity::kINFO)
      .value("WARNING", logging::Severity::kWARNING)
      .value("ERROR", logging::Severity::kERROR)
      .value("FATAL", logging::Severity::kFATAL);

#ifdef ENABLE_TRAINING_APIS
  py::class_<onnxruntime::training::api::Module> training_module(m, "Module", R"pbdoc(Training Module.)pbdoc");
  training_module
      .def(py::init([](const std::string& model_uri,
                       onnxruntime::training::api::CheckpointState* state,
                       std::optional<std::string> eval_model_uri,
                       OrtDevice device, PySessionOptions* session_options) {
        std::vector<std::shared_ptr<IExecutionProvider>> provider = GetExecutionProvidersForTrainingApis(device);
        auto model_identifiers = onnxruntime::training::api::ModelIdentifiers(model_uri, eval_model_uri, std::nullopt);
        return std::make_unique<onnxruntime::training::api::Module>(model_identifiers,
                                                                    state, session_options->value, GetTrainingEnv().GetORTEnv().GetEnvironment(), provider, session_options->custom_op_domains_);
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
             if (state->module_checkpoint_state.is_nominal_state) {
               ORT_THROW("Cannot copy parameter to a nominal state. Please load all the parameter states first");
             }
             auto it = state->module_checkpoint_state.named_parameters.find(parameter_name);
             if (it == state->module_checkpoint_state.named_parameters.end()) {
               ORT_THROW("Parameter with name ", parameter_name, " does not exist.");
             }
             ORT_THROW_IF_ERROR(it->second->CopyFrom(
                 state->module_checkpoint_state.train_session_data_transfer_mgr, value));
           })
      .def("get_parameter",
           [](onnxruntime::training::api::CheckpointState* state, const std::string& parameter_name) {
             if (state->module_checkpoint_state.is_nominal_state) {
               ORT_THROW("Cannot get parameter from a nominal state. Please load the parameter states first");
             }
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
        // In case the optimizer was constructed using a nominal checkpoint,
        // the optimizer state construction is delayed until the first call to Optimizer::Step().
        // It is expected that the model parameter state is available at this point.
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
         const std::string& checkpoint_path, const bool nominal_checkpoint) {
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
                                                                      ToPathString(checkpoint_path),
                                                                      nominal_checkpoint));
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
          auto logger_ptr = GetOrtEnv()->GetLoggingManager()->CreateLogger("orttraining");
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
#endif
}

}  // namespace python
}  // namespace onnxruntime
