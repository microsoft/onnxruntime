// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/inference_session.h"

#include <memory>
#include <sstream>
#include <unordered_set>
#include <list>

#include "core/common/logging/logging.h"
#include "core/common/task_thread_pool.h"
#include "core/platform/notification.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/customregistry.h"
#include "core/framework/environment.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/path_lib.h"
#include "core/framework/session_state.h"
#include "core/framework/session_state_initializer.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/utils.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/custom_ops_author.h"
#include "core/session/IOBinding.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"

#ifdef USE_EIGEN_THREADPOOL
#include <unsupported/Eigen/CXX11/ThreadPool>
#endif

using namespace ONNX_NAMESPACE;

constexpr OrtCustomOpApi g_custom_op_api = {
    &OrtKernelInfoGetAttribute_float,
    &OrtKernelInfoGetAttribute_int64,

    &OrtGetTensorShapeAndType,

    &OrtGetNumOfDimensions,
    &OrtGetDimensions,
    &OrtSetDims,

    &OrtGetTensorMutableData,

    &OrtReleaseTensorTypeAndShapeInfo,
};

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(const onnxruntime::DataTypeImpl* cpp_type);

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

namespace onnxruntime {
namespace {
template <typename T>
const T* GetDateFormatString();

template <>
inline const char* GetDateFormatString<char>() {
  return "%Y-%m-%d_%H-%M-%S";
}
#ifdef _WIN32
template <>
inline const wchar_t* GetDateFormatString<wchar_t>() {
  return L"%Y-%m-%d_%H-%M-%S";
}
#endif
//TODO: use LoggingManager::GetTimestamp and date::operator<<
// (see ostream_sink.cc for an example)
// to simplify this and match the log file timestamp format.
template <typename T>
inline std::basic_string<T> GetCurrentTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm;  // NOLINT

#ifdef _WIN32
  ORT_ENFORCE(localtime_s(&local_tm, &in_time_t) == 0);
#else
  localtime_r(&in_time_t, &local_tm);
#endif

  T time_str[32];
  OrtStrftime<T>(time_str, sizeof(time_str), GetDateFormatString<T>(), &local_tm);
  return std::basic_string<T>(time_str);
}
}  // namespace
struct CustomOpKernel : OpKernel {
  CustomOpKernel(const OpKernelInfo& info, OrtCustomOp& op) : OpKernel(info), op_(op) {
    if (op_.version != 1)
      throw std::invalid_argument("Unsupported version '" + std::to_string(op_.version) + "' in custom op '" + op.GetName(&op));
    op_.CreateKernel(&op_, &g_custom_op_api, reinterpret_cast<OrtKernelInfo*>(const_cast<OpKernelInfo*>(&info)), &op_kernel_);
  }

  ~CustomOpKernel() {
    op_.KernelDestroy(op_kernel_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    auto* ictx = static_cast<OpKernelContextInternal*>(ctx);
    std::vector<OrtValue*> input_tensors;
    auto input_count = ictx->InputCount();
    for (int i = 0; i < input_count; i++)
      input_tensors.emplace_back(const_cast<OrtValue*>(reinterpret_cast<const OrtValue*>(ictx->GetInputMLValue(i))));

    std::vector<OrtValue*> output_tensors;
    auto output_count = ictx->OutputCount();
    for (int i = 0; i < output_count; i++) {
      OrtTensorTypeAndShapeInfo info;
      op_.KernelGetOutputShape(op_kernel_, input_tensors.data(), input_tensors.size(), i, &info);
      output_tensors.emplace_back(reinterpret_cast<OrtValue*>(ictx->OutputMLValue(0, info.shape)));
    }

    op_.KernelCompute(op_kernel_, input_tensors.data(), input_tensors.size(), output_tensors.data(), output_tensors.size());
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpKernel);

  OrtCustomOp& op_;
  void* op_kernel_;
};

class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options, logging::LoggingManager* logging_manager)
      : session_options_{session_options},
        graph_transformation_mgr_{session_options_.max_num_graph_transformation_steps},
        logging_manager_{logging_manager},
        session_state_{execution_providers_},
        insert_cast_transformer_{"CastFloat16Transformer"} {
    ORT_ENFORCE(Environment::IsInitialized(),
                "Environment must be initialized before creating an InferenceSession.");

    InitLogger(logging_manager);

    // currently the threadpool is used by the parallel executor only and hence
    // there is no point creating it when only sequential execution is enabled.
    if (!session_options.enable_sequential_execution) {
      int pool_size = session_options_.session_thread_pool_size == 0
                          ? std::thread::hardware_concurrency() / 2
                          : session_options_.session_thread_pool_size;

#ifdef USE_EIGEN_THREADPOOL
      thread_pool_ = std::make_unique<Eigen::NonBlockingThreadPool>(pool_size);
#else
      thread_pool_ = std::make_unique<TaskThreadPool>(pool_size);
#endif
    }

    session_state_.SetThreadPool(thread_pool_.get());
    session_profiler_.Initialize(session_logger_);
    session_state_.SetProfiler(session_profiler_);
    if (session_options.enable_profiling) {
      StartProfiling(session_options.profile_file_prefix);
    }
  }

  common::Status RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
    if (p_exec_provider == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for exec provider");
    }

    std::string provider_type = p_exec_provider->Type();
    VLOGS(*session_logger_, 1) << "Adding execution provider of type: " << provider_type;
    execution_providers_.Add(provider_type, std::move(p_exec_provider));

    return Status::OK();
  }

  common::Status RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer,
                                          const std::vector<std::string>& providers,
                                          TransformerLevel level) {
    if (p_graph_transformer == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for graph transformer");
    }
    return graph_transformation_mgr_.Register(std::move(p_graph_transformer), level, providers);
  }

  common::Status AddCustomTransformerList(const std::vector<std::string>& transformers_to_enable) {
    std::copy(transformers_to_enable.begin(), transformers_to_enable.end(),
              std::back_inserter(transformers_to_enable_));

    return Status::OK();
  }

  common::Status AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& op_domains) {
    auto custom_registry = std::make_shared<CustomRegistry>();

    for (auto& domain : op_domains) {
      SchemasContainer schemas_container;

      schemas_container.domain = domain->domain_;
      schemas_container.baseline_opset_version = 1;
      schemas_container.opset_version = 1000;

      for (auto& op : domain->custom_ops_) {
        ONNX_NAMESPACE::OpSchema schema(op->GetName(op), "unknown", 0);

        auto input_count = op->GetInputTypeCount(op);
        for (size_t i = 0; i < input_count; i++) {
          auto type = op->GetInputType(op, i);

          schema.Input(i, "A", "Description",
                       DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)));
        }

        auto output_count = op->GetOutputTypeCount(op);
        for (size_t i = 0; i < output_count; i++) {
          auto type = op->GetOutputType(op, i);

          schema.Output(i, "A", "Description",
                        DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)));
        }

        schema.SinceVersion(1);
        schema.AllowUncheckedAttributes();

        schemas_container.schemas_list.push_back(schema);

        KernelDefBuilder def_builder;
        def_builder.SetName(op->GetName(op))
            .SetDomain(onnxruntime::kOnnxDomain)
            .SinceVersion(1)
            .Provider(onnxruntime::kCpuExecutionProvider);
        KernelCreateFn kernel_create_fn = [&op](const OpKernelInfo& info) -> OpKernel* { return new CustomOpKernel(info, *op); };
        KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);

        custom_registry->RegisterCustomKernel(create_info);
      }

      ORT_RETURN_IF_ERROR(custom_registry->RegisterOpSet(schemas_container.schemas_list,
                                                         schemas_container.domain,
                                                         schemas_container.baseline_opset_version,
                                                         schemas_container.opset_version));
    }
    RegisterCustomRegistry(custom_registry);
    return Status::OK();
  }

  common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry>& custom_registry) {
    if (custom_registry == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for custom registry");
    }

    // Insert session-level customized kernel registry.
    kernel_registry_manager_.RegisterKernelRegistry(custom_registry);
    //    if (custom_schema_registries_.empty())
    //      custom_schema_registries_.push_back();
    custom_schema_registries_.push_back(custom_registry);
    return Status::OK();
  }

  common::Status Load(std::function<common::Status(std::shared_ptr<Model>&)> loader, const std::string& event_name) {
    Status status = Status::OK();
    auto tp = session_profiler_.StartTime();
    try {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (is_model_loaded_) {  // already loaded
        LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
        return common::Status(common::ONNXRUNTIME, common::MODEL_LOADED,
                              "This session already contains a loaded model.");
      }

      std::shared_ptr<onnxruntime::Model> p_tmp_model;
      status = loader(p_tmp_model);
      ORT_RETURN_IF_ERROR(status);

      model_ = p_tmp_model;

      status = DoPostLoadProcessing(*model_);
      ORT_RETURN_IF_ERROR(status);

      // all steps complete, mark the model as loaded.
      is_model_loaded_ = true;
    } catch (const std::exception& ex) {
      status = Status(common::ONNXRUNTIME, common::FAIL, "Exception during loading: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
      status = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
    }

    if (session_profiler_.FEnabled()) {
      session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, event_name, tp);
    }

    return status;
  }

  template <typename T>
  common::Status Load(const T& model_uri) {
    model_location_ = ToWideString(model_uri);
    auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
      return onnxruntime::Model::Load(model_location_, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
    };

    common::Status st = Load(loader, "model_loading_uri");
    if (!st.IsOK()) {
      std::ostringstream oss;
      oss << "Load model from " << ToMBString(model_uri) << " failed:" << st.ErrorMessage();
      return common::Status(st.Category(), st.Code(), oss.str());
    }
    return Status::OK();
  }

  common::Status Load(const ModelProto& model_proto) {
    auto loader = [this, &model_proto](std::shared_ptr<onnxruntime::Model>& model) {
      return onnxruntime::Model::Load(model_proto, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
    };

    return Load(loader, "model_loading_proto");
  }

  common::Status Load(std::unique_ptr<ModelProto> p_model_proto) {
    auto loader = [this, &p_model_proto](std::shared_ptr<onnxruntime::Model>& model) {
      return onnxruntime::Model::Load(std::move(p_model_proto), model,
                                      HasLocalSchema() ? &custom_schema_registries_ : nullptr);
    };

    return Load(loader, "model_loading_proto");
  }

  common::Status Load(std::istream& model_istream) {
    auto loader = [this, &model_istream](std::shared_ptr<onnxruntime::Model>& model) {
      ModelProto model_proto;

      google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
      const bool result = model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
      if (!result) {
        return Status(common::ONNXRUNTIME, common::INVALID_PROTOBUF,
                      "Failed to load model because protobuf parsing failed.");
      }

      return onnxruntime::Model::Load(model_proto, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
    };

    return Load(loader, "model_loading_istream");
  }

  static common::Status TransformGraph(onnxruntime::Graph& graph,
                                       const onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                       const ExecutionProviders& providers,
                                       KernelRegistryManager& kernel_registry_manager,
                                       const InsertCastTransformer& insert_cast_transformer,
                                       SessionState& session_state) {
    // The transformer order:
    // 1. built-in graph rewriter
    // 2. each execution provider's transformer
    // 3. do node placement according to kernel definition
    // 4. insert copy nodes
    // 5. insert cast nodes.

    // first apply global(execution provider independent),  level 1(default/system/basic) graph to graph optimizations
    ORT_RETURN_IF_ERROR(graph_transformer_mgr.ApplyTransformers(graph, TransformerLevel::Level1));

    // Do partitioning based on execution providers' capability.
    GraphPartitioner partitioner(kernel_registry_manager, providers);
    ORT_RETURN_IF_ERROR(partitioner.Partition(graph, session_state.ExportDll(), session_state.GetMutableFuncMgr()));

    // apply transformers except default transformers
    // Default transformers are required for correctness and they are owned and run by inference session
    for (int i = static_cast<int>(TransformerLevel::Level1); i < static_cast<int>(TransformerLevel::MaxTransformerLevel); i++) {
      ORT_RETURN_IF_ERROR(graph_transformer_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i)));
    }

    bool modified = false;
    // Insert cast node/s.
    ORT_RETURN_IF_ERROR(insert_cast_transformer.Apply(graph, modified));

    // Now every node should be already assigned to an execution provider
    for (auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType().empty()) {
        std::ostringstream oss;
        oss << "Could not find an implementation for the node ";
        if (!node.Name().empty()) oss << node.Name() << ":";
        oss << node.OpType();
        if (node.Op()) {
          oss << "(" << node.Op()->since_version() << ")";
        }
        return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, oss.str());
      }
    }

    std::vector<std::string> provider_types;
    for (auto& provider_ptr : providers) {
      provider_types.push_back(provider_ptr->Type());
    }

    // Insert copy node/s.
    MemcpyTransformer copy_transformer{provider_types, kernel_registry_manager};
    ORT_RETURN_IF_ERROR(copy_transformer.Apply(graph, modified));

    return common::Status::OK();
  }

  /// Create SessionState instance for each subgraph as we need that for the GraphPartitioner
  /// This will be initialized by InitializeSubgraphSessions.
  common::Status CreateSubgraphSessionState(Graph& graph, SessionState& session_state) {
    for (auto& node : graph.Nodes()) {
      for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
        auto& name = entry.first;
        Graph* subgraph = entry.second;
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        auto subgraph_session_state = std::make_unique<SessionState>(execution_providers_);
        subgraph_session_state->SetProfiler(session_profiler_);
        subgraph_session_state->SetLogger(*session_logger_);

        // recurse
        ORT_RETURN_IF_ERROR(CreateSubgraphSessionState(*subgraph, *subgraph_session_state));

        // add the subgraph SessionState instance to the parent graph SessionState so it can be retrieved
        // by Compute() via OpKernelContextInternal.
        session_state.AddSubgraphSessionState(node.Index(), name, std::move(subgraph_session_state));
      }
    }

    return Status::OK();
  }

  /// iterate nodes in graph looking for ones with graph attribute/s
  /// @param graph The graph to iterate
  /// @param session_state The SessionState instance for 'graph'.
  /// @remarks We pass in graph and session_state so we can handled nested subgraphs in the future
  common::Status InitializeSubgraphSessions(Graph& graph, SessionState& session_state) {
    for (auto& node : graph.Nodes()) {
      for (const auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
        auto& name = entry.first;
        Graph& subgraph = *entry.second;

        SessionState* subgraph_session_state = session_state.GetMutableSubgraphSessionState(node.Index(), name);
        ORT_ENFORCE(subgraph_session_state, "CreateSubgraphSessionState should have created an entry earlier.");

        // setup everything required to execute the subgraph and save it in subgraph_session_state
        SessionStateInitializer initializer{model_location_, subgraph, *subgraph_session_state, execution_providers_,
                                            kernel_registry_manager_};

        ORT_RETURN_IF_ERROR(initializer.CreatePlan(&node, node.ImplicitInputDefs(),
                                                   session_options_.enable_sequential_execution));

        ORT_RETURN_IF_ERROR(initializer.InitializeAndSave(&node.ImplicitInputDefs()));

        // LOGS(*session_logger_, VERBOSE) << std::make_pair(subgraph_info.session_state->GetExecutionPlan(),
        //                                                   &*subgraph_info.session_state);

        // recurse
        ORT_RETURN_IF_ERROR(InitializeSubgraphSessions(subgraph, *subgraph_session_state));
      }
    }

    return Status::OK();
  }

  common::Status Initialize() {
    Status status = Status::OK();
    auto tp = session_profiler_.StartTime();

    try {
      LOGS(*session_logger_, INFO) << "Initializing session.";
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded.");
      }

      if (is_inited_) {  // already initialized
        LOGS(*session_logger_, INFO) << "Session has already been initialized.";
        return common::Status::OK();
      }

      // Register default CPUExecutionProvider if user didn't provide it through the Register() calls
      if (!execution_providers_.Get(onnxruntime::kCpuExecutionProvider)) {
        LOGS(*session_logger_, INFO) << "Adding default CPU execution provider.";
        CPUExecutionProviderInfo epi{session_options_.enable_cpu_mem_arena};
        ORT_RETURN_IF_ERROR(execution_providers_.Add(onnxruntime::kCpuExecutionProvider,
                                                     std::make_unique<CPUExecutionProvider>(epi)));
      }

      // add predefined transformers
      AddPredefinedTransformers(graph_transformation_mgr_, session_options_.graph_optimization_level, transformers_to_enable_);

      onnxruntime::Graph& graph = model_->MainGraph();

      // Collect the kernel registries from execution provider instances;
      // There are 2 kinds of kernel registries with priority from high to low as below,
      // 1. Custom execution provider type specific kernel registries.
      // 2. common execution provider type specific kernel registries.
      // The 1st and 2nd ones are shared across sessions.
      // The 1st ones should have already been registered via session-level API into KernelRegistryManager.
      //
      // Register 2nd registries into KernelRegistryManager.
      ORT_RETURN_IF_ERROR(kernel_registry_manager_.RegisterKernels(execution_providers_));

      SessionStateInitializer session_initializer{model_location_, graph, session_state_, execution_providers_,
                                                  kernel_registry_manager_};

      // create SessionState for subgraphs as it's needed by the transformers
      ORT_RETURN_IF_ERROR(CreateSubgraphSessionState(graph, session_state_));

      // apply any transformations to the main graph and any subgraphs
      ORT_RETURN_IF_ERROR(TransformGraph(graph, graph_transformation_mgr_,
                                         execution_providers_, kernel_registry_manager_,
                                         insert_cast_transformer_,
                                         session_state_));

      // now that all the transforms are done, call Resolve on the main graph. this will recurse into the subgraphs.
      ORT_RETURN_IF_ERROR(graph.Resolve());

      ORT_RETURN_IF_ERROR(session_initializer.CreatePlan(nullptr, {}, session_options_.enable_sequential_execution));
      ORT_RETURN_IF_ERROR(session_initializer.InitializeAndSave(nullptr));

      // handle any subgraphs
      ORT_RETURN_IF_ERROR(InitializeSubgraphSessions(graph, session_state_));

      session_state_.CalculateNodeIndexInfo();

      is_inited_ = true;

      LOGS(*session_logger_, INFO) << "Session successfully initialized.";
    } catch (const NotImplementedException& ex) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    } catch (const std::exception& ex) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    } catch (...) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Encountered unknown exception in Initialize()");
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    }

    if (session_profiler_.FEnabled()) {
      session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "session_initialization", tp);
    }
    return status;
  }

  int GetCurrentNumRuns() const {
    return current_num_runs_.load();
  }

  static common::Status CheckTypes(MLDataType actual, MLDataType expected) {
    if (actual == expected) {
      return Status::OK();
    }
    auto actual_name = std::string(typeid(*actual).name());
    auto expected_name = std::string(typeid(*expected).name());
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Unexpected input data type. Actual: (" + actual_name + ") , expected: (" + expected_name + ")");
  }

  common::Status ValidateInputs(const std::vector<std::string>& feed_names,
                                const std::vector<MLValue>& feeds) {
    const auto begin_names = feed_names.cbegin();
    const auto end_names = feed_names.cend();
    std::unordered_set<ptrdiff_t> required_feed_ids;
    for (auto& arg : required_input_def_list_) {
      auto& arg_name = arg->Name();
      if (arg_name.empty()) {
        continue;
      }

      auto feed_names_entry = std::find(begin_names, end_names, arg_name);
      if (feed_names_entry == end_names) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Missing required input: ", arg_name);
      }

      auto idx = feed_names_entry - begin_names;
      required_feed_ids.insert(idx);
      auto& input_ml_value = feeds.at(idx);
      auto expected_type = utils::GetMLDataType(*arg);

      if (input_ml_value.IsTensor()) {
        auto expected_element_type = expected_type->AsTensorType()->GetElementType();
        auto input_element_type = input_ml_value.Get<Tensor>().DataType();
        ORT_RETURN_IF_ERROR(CheckTypes(input_element_type, expected_element_type));
      } else {
        auto input_type = input_ml_value.Type();
        ORT_RETURN_IF_ERROR(CheckTypes(input_type, expected_type));
      }
    }

    if (feeds.size() > required_feed_ids.size()) {
      // More feeds are offered.
      // In the case of overriding some initializers (which are also taken as graph inputs).
      for (size_t i = 0; i < feeds.size(); ++i) {
        if (required_feed_ids.count(i) > 0) {
          continue;
        }
        auto iter = input_def_map_.find(feed_names[i]);
        if (input_def_map_.end() == iter) {
          std::ostringstream ostr;
          std::for_each(std::begin(model_input_names_),
                        std::end(model_input_names_),
                        [&ostr](const std::string& elem) {
                          ostr << elem << " ";
                        });
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid Feed Input Names:", feed_names[i],
                                 ". Valid input names are: ", ostr.str());
        }

        auto& input_ml_value = feeds.at(i);
        ORT_ENFORCE(input_ml_value.IsTensor());
        auto input_element_type = input_ml_value.Get<Tensor>().DataType();

        auto expected_type = utils::GetMLDataType(*iter->second);
        auto expected_element_type = expected_type->AsTensorType()->GetElementType();

        ORT_RETURN_IF_ERROR(CheckTypes(input_element_type, expected_element_type));
      }
    }

    return Status::OK();
  }

  common::Status ValidateOutputs(const std::vector<std::string>& output_names,
                                 const std::vector<MLValue>* p_fetches) {
    if (!p_fetches) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "Output vector pointer is NULL");
    }

    if (output_names.empty()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "At least one output should be requested.");
    }

    if (!p_fetches->empty() &&
        (output_names.size() != p_fetches->size())) {
      std::ostringstream ostr;
      ostr << "Output vector incorrectly sized: output_names.size(): " << output_names.size()
           << "p_fetches->size(): " << p_fetches->size();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    bool valid = true;
    std::ostringstream invalid_names;
    for (const auto& name : output_names) {
      if (model_output_names_.find(name) == model_output_names_.end()) {
        valid = false;
        invalid_names << " " << name;
      }
    }

    if (!valid) {
      std::ostringstream ostr;
      std::for_each(std::begin(model_output_names_),
                    std::end(model_output_names_),
                    [&ostr](const std::string& elem) {
                      ostr << elem << " ";
                    });
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "Invalid Output Names:" + invalid_names.str() +
                                " Valid output names are: " + ostr.str());
    }

    // TODO add more validation here like checking shape of the allocated buffers

    return common::Status::OK();
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::string>& feed_names,
             const std::vector<MLValue>& feeds,
             const std::vector<std::string>& output_names,
             std::vector<MLValue>* p_fetches) {
    auto tp = session_profiler_.StartTime();
    Status retval = Status::OK();

    try {
      {
        std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
        if (!is_inited_) {
          LOGS(*session_logger_, ERROR) << "Session was not initialized";
          retval = Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
        }
      }

      ORT_RETURN_IF_ERROR(ValidateInputs(feed_names, feeds));

      // if the output vector is non-empty, ensure that its the same size as the output_names
      ORT_RETURN_IF_ERROR(ValidateOutputs(output_names, p_fetches));

      FeedsFetchesInfo info(feed_names, output_names);
      ORT_RETURN_IF_ERROR(info.SetMLValueIdxs(session_state_.GetMLValueNameIdxMap()));
      FeedsFetchesManager feeds_fetches_manager{std::move(info)};

      if (!run_options.run_tag.empty()) {
        LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
      }

      ++current_num_runs_;

      // TODO should we add this exec to the list of executors? i guess its not needed now?

      // scope of owned_run_logger is just the call to Execute.
      // If Execute ever becomes async we need a different approach
      std::unique_ptr<logging::Logger> owned_run_logger;
      auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

      // info all execution providers InferenceSession:Run started
      // TODO: only call OnRunStart for all providers in-use
      for (auto& xp : execution_providers_) {
        ORT_CHECK_AND_SET_RETVAL(xp->OnRunStart());
      }

      // execute the graph
      ORT_CHECK_AND_SET_RETVAL(
          utils::ExecuteGraph(session_state_, feeds_fetches_manager, feeds, *p_fetches, {},
                              session_options_.enable_sequential_execution, run_options.terminate, run_logger,
                              false));

    } catch (const std::exception& e) {
      retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
    } catch (...) {
      retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
    }

    // info all execution providers InferenceSession:Run ended
    for (auto& xp : execution_providers_) {
      ORT_CHECK_AND_SET_RETVAL(xp->OnRunEnd());
    }

    --current_num_runs_;
    if (session_profiler_.FEnabled()) {
      session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
    }

    return retval;
  }

  std::pair<common::Status, const ModelMetadata*> GetModelMetadata() const {
    {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(common::Status::OK(), &model_metadata_);
  }

  std::pair<common::Status, const InputDefList*> GetModelInputs() const {
    {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(common::Status::OK(), &required_input_def_list_);
  }

  std::pair<common::Status, const OutputDefList*> GetModelOutputs() const {
    {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(common::Status::OK(), &output_def_list_);
  }

  common::Status NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
    {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (!is_inited_) {
        LOGS(*session_logger_, ERROR) << "Session was not initialized";
        return common::Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
      }
    }

    // private constructor, can't use make_unique
    *io_binding = std::unique_ptr<IOBinding>(new IOBinding(session_state_));
    return Status::OK();
  }

  common::Status Run(const RunOptions& run_options, IOBinding& io_binding) {
    // TODO should Run() call io_binding.SynchronizeInputs() or should it let the callers do it?
    // io_binding.SynchronizeInputs();
    return Run(run_options, io_binding.feed_names_, io_binding.feeds_, io_binding.output_names_, &io_binding.outputs_);
  }

  common::Status Run(IOBinding& io_binding) {
    RunOptions run_options;
    return Run(run_options, io_binding);
  }

  template <typename T>
  void StartProfiling(const std::basic_string<T>& file_prefix) {
    std::basic_ostringstream<T> ss;
    ss << file_prefix << "_" << GetCurrentTimeString<T>() << ".json";
    session_profiler_.StartProfiling(ss.str());
  }

  void StartProfiling(const logging::Logger* logger_ptr) {
    session_profiler_.StartProfiling(logger_ptr);
  }

  std::string EndProfiling() {
    if (is_model_loaded_) {
      return session_profiler_.EndProfiling();
    }
    LOGS(*session_logger_, ERROR) << "Could not write a profile because no model was loaded.";
    return std::string();
  }

 private:
  bool HasLocalSchema() const {
    return !custom_schema_registries_.empty();
  }

  // assumes model has already been loaded before
  common::Status DoPostLoadProcessing(onnxruntime::Model& model) {
    // TODO add other post load processing here
    common::Status status = SaveModelMetadata(model);
    return status;
  }

  common::Status SaveModelMetadata(const onnxruntime::Model& model) {
    VLOGS(*session_logger_, 1) << "Saving model metadata";
    const onnxruntime::Graph& graph = model.MainGraph();

    // save model metadata
    model_metadata_.producer_name = model.ProducerName();
    model_metadata_.description = model.DocString();
    model_metadata_.domain = model.Domain();
    model_metadata_.version = model.ModelVersion();
    model_metadata_.custom_metadata_map = model.MetaData();
    model_metadata_.graph_name = graph.Name();

    // save required inputs
    const auto& required_inputs = graph.GetInputs();  // inputs excluding initializers
    required_input_def_list_.reserve(required_inputs.size());
    required_model_input_names_.reserve(required_inputs.size());
    for (const auto& elem : required_inputs) {
      required_input_def_list_.push_back(elem);
      required_model_input_names_.insert(elem->Name());
    }

    // save all valid inputs
    auto& all_inputs = graph.GetInputsIncludingInitializers();
    input_def_map_.reserve(all_inputs.size());
    model_input_names_.reserve(all_inputs.size());
    for (auto elem : all_inputs) {
      input_def_map_.insert({elem->Name(), elem});
      model_input_names_.insert(elem->Name());
    }

    // save outputs
    const auto& outputs = graph.GetOutputs();
    output_def_list_.reserve(outputs.size());
    model_output_names_.reserve(outputs.size());
    for (const auto& elem : outputs) {
      output_def_list_.push_back(elem);
      model_output_names_.insert(elem->Name());
    }

    VLOGS(*session_logger_, 1) << "Done saving model metadata";
    return common::Status::OK();
  }

  // Create a Logger for a single execution if possible. Otherwise use the default logger.
  // If a new logger is created, it will also be stored in new_run_logger,
  // which must remain valid for the duration of the execution.
  // If the default logger is used, new_run_logger will remain empty.
  // The returned value should be used in the execution.
  const logging::Logger& CreateLoggerForRun(const RunOptions& run_options,
                                            std::unique_ptr<logging::Logger>& new_run_logger) {
    const logging::Logger* run_logger;

    // create a per-run logger if we can
    if (logging_manager_ != nullptr) {
      std::string run_log_id{session_options_.session_logid};

      if (!session_options_.session_logid.empty() && !run_options.run_tag.empty()) {
        run_log_id += ":";
      }

      run_log_id += run_options.run_tag;

      if (run_options.run_log_verbosity_level > 0) {
        new_run_logger = logging_manager_->CreateLogger(run_log_id,
                                                        logging::Severity::kVERBOSE,
                                                        false,
                                                        run_options.run_log_verbosity_level);
      } else {
        new_run_logger = logging_manager_->CreateLogger(run_log_id);
      }

      run_logger = new_run_logger.get();
      VLOGS(*run_logger, 1) << "Created logger for run with id of " << run_log_id;
    } else {
      // fallback to using default logger. this does NOT have any session or run specific id/tag in it
      run_logger = session_logger_;
      VLOGS(*run_logger, 1) << "Using default logger for run " << run_options.run_tag;
    }

    return *run_logger;
  }

  void InitLogger(logging::LoggingManager* logging_manager) {
    // create logger for session, using provided logging manager if possible
    if (logging_manager != nullptr) {
      std::string session_logid = !session_options_.session_logid.empty()
                                      ? session_options_.session_logid
                                      : "InferenceSession";  // there's probably a better default...

      if (session_options_.session_log_verbosity_level > 0) {
        owned_session_logger_ = logging_manager->CreateLogger(session_logid,
                                                              logging::Severity::kVERBOSE,
                                                              false,
                                                              session_options_.session_log_verbosity_level);
      } else {
        owned_session_logger_ = logging_manager->CreateLogger(session_logid);
      }
      session_logger_ = owned_session_logger_.get();
    } else {
      session_logger_ = &logging::LoggingManager::DefaultLogger();
    }

    session_state_.SetLogger(*session_logger_);
  }

  // Registers all the predefined transformers with transformer manager
  void AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                 TransformerLevel graph_optimization_level,
                                 const std::vector<std::string>& custom_list) {
    auto add_transformers = [&](TransformerLevel level, std::vector<std::string>&& providers, std::string t_name) {
      // Generate and register rewrite rules for level
      auto rewrite_rules_to_register =
          transformer_utils::GenerateRewriteRules(level, &custom_list);
      if (!rewrite_rules_to_register.empty()) {
        std::unique_ptr<RuleBasedGraphTransformer> graph_rewrite_rules =
            std::make_unique<TopDownRuleBasedTransformer>(t_name + "_RuleBasedTransformer",
                                                          "Apply rewrite rules for " + t_name);
        for (auto& entry : rewrite_rules_to_register) {
          graph_rewrite_rules->Register(std::move(entry));
        }
        transformer_manager.Register(std::move(graph_rewrite_rules), level,
                                     std::move(providers));
      }

      // Generate and register transformers for level
      auto transformers_to_register = transformer_utils::GenerateTransformers(level, &custom_list);
      for (auto& entry : transformers_to_register) {
        transformer_manager.Register(std::move(entry.first), level, std::move(entry.second));
      }
    };

    if ((graph_optimization_level >= TransformerLevel::Level1) || !custom_list.empty()) {
      add_transformers(TransformerLevel::Level1, {}, "Level1");
    }

    if ((graph_optimization_level >= TransformerLevel::Level2) || !custom_list.empty()) {
      add_transformers(TransformerLevel::Level2, {onnxruntime::kCpuExecutionProvider}, "Level2");
    }
  }

  common::Status WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) {
    if (timeout_in_ms > 0) {
      ORT_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
    }
    p_executor_done->WaitForNotification();

    return Status::OK();
  }

  const SessionOptions session_options_;

  onnxruntime::GraphTransformerManager graph_transformation_mgr_;

  // List of transformers to run. When this list is not empty only the transformers in this list
  // will be run regardless of the level set.
  // .i.e This list overrides both SessionOptions.graph_optimization_level and predefined transformers.
  std::vector<std::string> transformers_to_enable_;

  /// Logging manager if provided.
  logging::LoggingManager* logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<logging::Logger> owned_session_logger_;

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const logging::Logger* session_logger_;

  // Profiler for this session.
  profiling::Profiler session_profiler_;

  ExecutionProviders execution_providers_;

  KernelRegistryManager kernel_registry_manager_;
  std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> custom_schema_registries_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<onnxruntime::Model> model_;

  // A set of executors that can run in parallel.
  std::vector<std::unique_ptr<IExecutor>> executors_;  // TODO do we need this vector?

  // Immutable state for each op in the model. Shared by all executors.
  SessionState session_state_;

  ModelMetadata model_metadata_;
  InputDefList required_input_def_list_;
  std::unordered_map<std::string, const NodeArg*> input_def_map_;
  OutputDefList output_def_list_;

  // names of model inputs and outputs used for quick validation.
  std::unordered_set<std::string> required_model_input_names_;
  std::unordered_set<std::string> model_input_names_;
  std::unordered_set<std::string> model_output_names_;

  // Environment for this session
  // not used now; we'll need it when we introduce threadpool
  // statically allocated pointer, no need to manage its lifetime.
  //Env* env_;

  // Threadpool for this session
  //thread::ThreadPool thread_pool_; // not used for now; will add it later when implementing RunAsync
#ifdef USE_EIGEN_THREADPOOL
  std::unique_ptr<Eigen::NonBlockingThreadPool> thread_pool_;
#else
  std::unique_ptr<TaskThreadPool> thread_pool_;
#endif

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  mutable onnxruntime::OrtMutex session_mutex_;  // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;                 // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;                       // GUARDED_BY(session_mutex_)

  InsertCastTransformer insert_cast_transformer_;
  // The file path of where the model was loaded. e.g. /tmp/test_squeezenet/model.onnx
  std::basic_string<PATH_CHAR_TYPE> model_location_;
};  // namespace onnxruntime

//
// InferenceSession
//
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   logging::LoggingManager* logging_manager)
    : impl_(std::make_unique<Impl>(session_options, logging_manager)) {
}

InferenceSession::~InferenceSession() = default;

common::Status InferenceSession::Load(const std::string& model_uri) {
  return impl_->Load(model_uri);
}
#ifdef _WIN32
common::Status InferenceSession::Load(const std::wstring& model_uri) {
  return impl_->Load(model_uri);
}
#endif
common::Status InferenceSession::Load(std::istream& model_istream) {
  return impl_->Load(model_istream);
}

common::Status InferenceSession::Initialize() {
  return impl_->Initialize();
}

common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const std::vector<std::string>& feed_names,
                                     const std::vector<MLValue>& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feed_names, feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return Run({}, feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const NameMLValMap& feeds_map,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;

  auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches);
}

std::pair<common::Status, const ModelMetadata*> InferenceSession::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetModelInputs() const {
  return impl_->GetModelInputs();
}

std::pair<common::Status, const OutputDefList*> InferenceSession::GetModelOutputs() const {
  return impl_->GetModelOutputs();
}

int InferenceSession::GetCurrentNumRuns() {
  return impl_->GetCurrentNumRuns();
}

void InferenceSession::StartProfiling(const std::string& file_prefix) {
  impl_->StartProfiling(file_prefix);
}

#ifdef _WIN32
void InferenceSession::StartProfiling(const std::wstring& file_prefix) { impl_->StartProfiling(file_prefix); }
#endif
void InferenceSession::StartProfiling(const logging::Logger* custom_logger) {
  impl_->StartProfiling(custom_logger);
}

std::string InferenceSession::EndProfiling() {
  return impl_->EndProfiling();
}

common::Status InferenceSession::RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
  return impl_->RegisterExecutionProvider(std::move(p_exec_provider));
}

common::Status InferenceSession::RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer,
                                                          const std::vector<std::string>& providers,
                                                          TransformerLevel level) {

  return impl_->RegisterGraphTransformer(std::move(p_graph_transformer), providers, level);
}

common::Status InferenceSession::AddCustomTransformerList(const std::vector<std::string>& transformers_to_enable) {
  return impl_->AddCustomTransformerList(transformers_to_enable);
}

common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  return impl_->RegisterCustomRegistry(custom_registry);
}

common::Status InferenceSession::Load(const ModelProto& model_proto) {
  return impl_->Load(model_proto);
}

common::Status InferenceSession::Load(std::unique_ptr<ModelProto> p_model_proto) {
  return impl_->Load(std::move(p_model_proto));
}

common::Status InferenceSession::NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
  return impl_->NewIOBinding(io_binding);
}

common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  return impl_->Run(run_options, io_binding);
}

common::Status InferenceSession::Run(IOBinding& io_binding) {
  return impl_->Run(io_binding);
}

common::Status InferenceSession::AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& ops) {
  return impl_->AddCustomOpDomains(ops);
}
}  // namespace onnxruntime
