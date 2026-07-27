// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include <limits>
#include <utility>
#include "core/framework/compile_utils.h"
#include "core/common/inlined_containers.h"
#include "core/framework/execution_providers.h"
#include "core/graph/graph_utils.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/model_saving_options.h"
#include "core/platform/env.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace onnxruntime {
namespace epctx {

Status GetValidatedEpContextPath(const std::filesystem::path& ep_context_path,
                                 const std::filesystem::path& model_path,
                                 std::filesystem::path& context_cache_path,
                                 bool error_if_output_file_exists) {
  if (!ep_context_path.empty()) {
    context_cache_path = ep_context_path;
    if (!(context_cache_path.has_filename() && context_cache_path.extension() != "")) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "context_file_path should not point to a folder.");
    }
  } else if (!model_path.empty()) {
    auto pos = model_path.native().find_last_of(ORT_TSTR("."));
    if (pos != std::string::npos) {
      context_cache_path = model_path.native().substr(0, pos) + ORT_TSTR("_ctx.onnx");
    } else {
      context_cache_path = model_path.native() + ORT_TSTR("_ctx.onnx");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Both ep_context_path and model_path are empty.");
  }

  if (std::filesystem::exists(context_cache_path) && error_if_output_file_exists) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to generate output model since the file '",
                           context_cache_path,
                           "' exists already. Please remove the output model file if you want to re-generate it.");
  }

  return Status::OK();
}

// Serialize an EPContext model into a onnx::ModelProto.
Status EpContextModelToProto(const onnxruntime::Model& ep_context_model,
                             const std::filesystem::path& validated_model_path,
                             const epctx::ModelGenOptions& ep_context_gen_options,
                             /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) {
  // Handle case where initializers are stored inline within the ONNX model.
  if (ep_context_gen_options.AreInitializersEmbeddedInOutputModel()) {
    // if no external ini file specified, set force_embed_external_ini to true to avoid intermediate file creation
    // and force all initializers embed into the ONNX file.
    ModelSavingOptions model_saving_options{/*size_threshold*/ SIZE_MAX};
    model_saving_options.force_embed_external_ini = true;

    model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(std::filesystem::path{},
                                                                        validated_model_path,
                                                                        model_saving_options);
    return Status::OK();
  }

  // Handle case where initializers (with size > threshold) are stored in an external file.
  if (const epctx::ExternalInitializerFileInfo* ext_info = ep_context_gen_options.TryGetExternalInitializerFileInfo();
      ext_info != nullptr) {
    ModelSavingOptions model_saving_options{ext_info->size_threshold};

    model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(ext_info->file_path,
                                                                        validated_model_path,
                                                                        model_saving_options);
    return Status::OK();
  }

  // Handle case where user specified a custom handler function that determines how each initializer is saved.
  if (const epctx::InitializerHandler* custom_handler = ep_context_gen_options.TryGetInitializerHandler();
      custom_handler != nullptr) {
    ORT_RETURN_IF_ERROR(ep_context_model.ToGraphProtoWithCustomInitializerHandling(
        custom_handler->handle_initializer_func,
        custom_handler->state,
        model_proto));
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected location for initializers while generating ",
                         validated_model_path);
}

//
// OutStreamBuf class:
//

OutStreamBuf::OutStreamBuf(BufferWriteFuncHolder write_func_holder)
    : write_func_holder_(write_func_holder), buffer_(65536) {
  setp(buffer_.data(), buffer_.data() + buffer_.size());
}

OutStreamBuf::~OutStreamBuf() {
  sync();
}

// Called when the buffer_ is full. Flushes the buffer_ (via sync()) and then writes the overflow character to buffer_.
std::streambuf::int_type OutStreamBuf::overflow(std::streambuf::int_type ch) {
  if (sync() == -1) {
    return traits_type::eof();
  }

  if (ch != traits_type::eof()) {
    *pptr() = static_cast<char>(ch);
    pbump(1);
  }

  return ch;
}

// Flushes the entire buffer_ to the user's write function.
int OutStreamBuf::sync() {
  if (!last_status_.IsOK()) {
    return -1;
  }

  std::ptrdiff_t num_bytes = pptr() - pbase();
  if (num_bytes == 0) {
    return 0;
  }

  // Can only call pbump() with an int, so can only write at most (2^31 - 1) bytes.
  if (num_bytes > std::numeric_limits<int>::max()) {
    num_bytes = std::numeric_limits<int>::max();
  }

  char* ptr = pbase();

  Status status = Status::OK();

  ORT_TRY {
    status = ToStatusAndRelease(write_func_holder_.write_func(write_func_holder_.stream_state,
                                                              ptr, num_bytes));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Caught exception while calling user's OrtOutStreamWriteFunc callback: ", e.what());
    });
  }

  if (!status.IsOK()) {
    last_status_ = std::move(status);
    return -1;
  }

  pbump(-static_cast<int>(num_bytes));  // Reset internal pointer to point to the beginning of the buffer_
  return 0;
}

Status SaveModelProtoToLocation(ONNX_NAMESPACE::ModelProto& model_proto,
                                const epctx::ModelGenOptions& gen_options,
                                const std::filesystem::path& valid_output_model_path,
                                const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  const epctx::BufferHolder* output_buffer_holder = gen_options.TryGetOutputModelBuffer();
  const epctx::BufferWriteFuncHolder* output_write_func_holder = gen_options.TryGetOutputModelWriteFunc();

  if (output_buffer_holder != nullptr) {
    // Write output model into a buffer ORT allocates for the user.
    size_t buffer_size = model_proto.ByteSizeLong();
    ORT_RETURN_IF(buffer_size > static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Cannot serialize ONNX ModelProto larger than 2GB");

    AllocatorPtr allocator = output_buffer_holder->buffer_allocator;
    IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, buffer_size);
    const bool ok = model_proto.SerializeToArray(buffer.get(), static_cast<int>(buffer_size));
    ORT_RETURN_IF(!ok, "Protobuf serialization failed when saving model to output buffer");

    *output_buffer_holder->buffer_size_ptr = buffer_size;
    *output_buffer_holder->buffer_ptr = buffer.release();
  }
  // Write output model to user's output stream.
  size_t buffer_size = model_proto.ByteSizeLong();
  ORT_RETURN_IF(buffer_size > static_cast<size_t>(std::numeric_limits<int>::max()),
                "Cannot serialize ONNX ModelProto larger than 2GB");

  auto out_stream_buf = std::make_unique<epctx::OutStreamBuf>(*output_write_func_holder);
  std::ostream out_stream(out_stream_buf.get());

  model_proto.SerializeToOstream(&out_stream);
  out_stream.flush();
  ORT_RETURN_IF_ERROR(out_stream_buf->GetStatus());
}
else {
  // Write output model to a file.
  int fd = 0;
  Status status = Env::Default().FileOpenWr(valid_output_model_path, fd);
  ORT_RETURN_IF_ERROR(status);

  ORT_TRY {
    google::protobuf::io::FileOutputStream output(fd);
    bool serialize_result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
    if (!serialize_result) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_PROTOBUF,
                               "Protobuf serialization failed when saving model to ",
                               valid_output_model_path);
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
    });
  }
  if (!status.IsOK()) {
    GSL_SUPPRESS(es .84)
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
    return status;
  }
  ORT_RETURN_IF_ERROR(Env::Default().FileClose(fd));
}

return Status::OK();
}

Status BuildAndSaveOptimizedModel(const onnxruntime::Model& model,
                                  const epctx::ModelGenOptions& gen_options,
                                  const logging::Logger& logger) {
  // Resolve the validated output path before building the proto so that any file-conflict error
  // surfaces early (before spending time on serialization).
  std::filesystem::path valid_output_model_path;
  const epctx::BufferHolder* output_buffer_holder = gen_options.TryGetOutputModelBuffer();
  const epctx::BufferWriteFuncHolder* output_write_func_holder = gen_options.TryGetOutputModelWriteFunc();
  const std::filesystem::path* output_model_path_ptr = gen_options.TryGetOutputModelPath();
  const bool output_is_to_file = (output_buffer_holder == nullptr && output_write_func_holder == nullptr);
  const bool needs_path_for_external_initializers = (gen_options.TryGetExternalInitializerFileInfo() != nullptr);

  if ((output_is_to_file || needs_path_for_external_initializers) &&
      (output_model_path_ptr != nullptr || !model.MainGraph().ModelPath().empty())) {
    std::filesystem::path output_model_path = (output_model_path_ptr != nullptr)
                                                  ? *output_model_path_ptr
                                                  : std::filesystem::path("");
    ORT_RETURN_IF_ERROR(GetValidatedEpContextPath(output_model_path,
                                                  model.MainGraph().ModelPath(),
                                                  valid_output_model_path,
                                                  gen_options.error_if_output_file_exists));
  }

  // Build ModelProto from the current (fully-optimized) model using the same serialization
  // path as the EPContext save so that in-memory initializers (OrtValue-backed) round-trip correctly.
  ONNX_NAMESPACE::ModelProto model_proto;
  if (gen_options.AreInitializersEmbeddedInOutputModel()) {
    ModelSavingOptions model_saving_options{/*size_threshold*/ SIZE_MAX};
    model_saving_options.force_embed_external_ini = true;
    model_proto = model.ToGraphProtoWithExternalInitializers(std::filesystem::path{},
                                                             valid_output_model_path,
                                                             model_saving_options);
  } else if (const epctx::ExternalInitializerFileInfo* ext_info = gen_options.TryGetExternalInitializerFileInfo();
             ext_info != nullptr) {
    ModelSavingOptions model_saving_options{ext_info->size_threshold};
    model_proto = model.ToGraphProtoWithExternalInitializers(ext_info->file_path,
                                                             valid_output_model_path,
                                                             model_saving_options);
  } else if (const epctx::InitializerHandler* custom_handler = gen_options.TryGetInitializerHandler();
             custom_handler != nullptr) {
    ORT_RETURN_IF_ERROR(model.ToGraphProtoWithCustomInitializerHandling(
        custom_handler->handle_initializer_func, custom_handler->state, model_proto));
  } else {
    // No explicit initializer location: embed everything inline.
    ModelSavingOptions model_saving_options{/*size_threshold*/ SIZE_MAX};
    model_saving_options.force_embed_external_ini = true;
    model_proto = model.ToGraphProtoWithExternalInitializers(std::filesystem::path{},
                                                             valid_output_model_path,
                                                             model_saving_options);
  }

  return SaveModelProtoToLocation(model_proto, gen_options, valid_output_model_path, logger);
}

Status CreateEpContextModel(const ExecutionProviders& execution_providers,
                            const Graph& graph,
                            const epctx::ModelGenOptions& ep_context_gen_options,
                            const logging::Logger& logger) {
  InlinedVector<const Node*> all_ep_context_nodes;
  for (const auto& ep : execution_providers) {
    const InlinedVector<const Node*> ep_context_nodes = ep->GetEpContextNodes();
    all_ep_context_nodes.insert(all_ep_context_nodes.begin(), ep_context_nodes.begin(), ep_context_nodes.end());
  }

  if (all_ep_context_nodes.size() < 1) {
    auto action_if_no_compiled_nodes = ep_context_gen_options.action_if_no_compiled_nodes;

    ORT_RETURN_IF(action_if_no_compiled_nodes == epctx::ModelGenOptions::ActionIfNoCompiledNodes::kReturnError,
                  "Unable to compile any nodes. Check that the session EPs support compilation and can execute "
                  "at least one subgraph in the model.");

    if (action_if_no_compiled_nodes == epctx::ModelGenOptions::ActionIfNoCompiledNodes::kDontGenerateModel) {
      LOGS(logger, WARNING) << "Unable to compile any nodes. ONNX Runtime will not generate a compiled model. "
                               "Either the session EPs do not support compilation or the model is already compiled.";
      // Note: this path is only taken if a model is compiled with the original compilation approach that uses
      // session options configs only. The explicit compile API instead only chooses between
      // kReturnError and kGenerateModel.
      return Status::OK();
    }

    // Assert so that this is caught in a test in DEBUG builds (in case a new enum value is added)
    assert(action_if_no_compiled_nodes == epctx::ModelGenOptions::ActionIfNoCompiledNodes::kGenerateModel);
    LOGS(logger, INFO) << "Unable to compile any nodes but will still generate an output model. "
                          "Either the session EPs do not support compilation or the model is already compiled.";
  }

  auto get_ep_context_node = [&all_ep_context_nodes](const std::string& node_name) -> std::pair<bool, const Node*> {
    for (auto& node : all_ep_context_nodes) {
      if (node_name == node->Name()) {
        return std::make_pair(true, node);
      }
    }
    return std::make_pair(false, static_cast<const Node*>(nullptr));
  };

  const epctx::BufferHolder* output_buffer_holder = ep_context_gen_options.TryGetOutputModelBuffer();
  const epctx::BufferWriteFuncHolder* output_write_func_holder = ep_context_gen_options.TryGetOutputModelWriteFunc();
  const std::filesystem::path* output_model_path_ptr = ep_context_gen_options.TryGetOutputModelPath();

  // Determine whether we need to resolve/validate a file system path for the output model.
  // A path is needed when:
  //   - Writing the output model to a file (not to a buffer or write function)
  //   - Writing initializers to an external file (needs the model path to compute the external file location)
  const bool output_is_to_file = (output_buffer_holder == nullptr && output_write_func_holder == nullptr);
  const bool needs_path_for_external_initializers =
      (ep_context_gen_options.TryGetExternalInitializerFileInfo() != nullptr);

  std::filesystem::path valid_output_model_path;
  if ((output_is_to_file || needs_path_for_external_initializers) &&
      (output_model_path_ptr != nullptr || !graph.ModelPath().empty())) {
    std::filesystem::path output_model_path = (output_model_path_ptr != nullptr) ? *output_model_path_ptr
                                                                                 : std::filesystem::path("");
    ORT_RETURN_IF_ERROR(GetValidatedEpContextPath(output_model_path,
                                                  graph.ModelPath(),
                                                  valid_output_model_path,
                                                  ep_context_gen_options.error_if_output_file_exists));
  }

  // Utility function to detect a fused node with an unsupported domain.
  // Ex: when compiling an already compiled model, an EPContext node in the input model would be wrapped
  // into a fused node with a domain like "QNN". Such fused nodes do not pass ONNX correctness checks, so
  // we should detect them here and return a better error message. Otherwise, an ORT_INVALID_GRAPH error is raised
  // with a confusing error message *after* the invalid model has been saved/generated.
  // Note: This only applies to the explicit compile API. The original compilation approach (via session options),
  // early exits above (without error) if the model is already compiled.
  auto is_invalid_fused_node = [&graph](const Node& node) {
    const std::unordered_map<std::string, int>& supported_domains = graph.DomainToVersionMap();
    return (node.NodeType() == Node::Type::Fused) && (supported_domains.find(node.Domain()) == supported_domains.end());
  };

  Model ep_context_model(graph.Name(), false, graph.GetModel().MetaData(),
                         graph.GetModel().ModelPath(),  // use source model path so that external initializers can find the data file path
                         IOnnxRuntimeOpSchemaRegistryList{graph.GetSchemaRegistry()},
                         graph.DomainToVersionMap(), {}, logger);
  auto& ep_graph = ep_context_model.MainGraph();
  ep_graph.SetDescription(graph.Description());

  // Set inputs outputs explicitly to make sure the order is same as the user model.
  auto inputs = graph.GetInputs();
  auto outputs = graph.GetOutputs();

  InlinedVector<const NodeArg*> ep_graph_inputs;
  ep_graph_inputs.reserve(inputs.size());
  for (auto& input : inputs) {
    auto input_arg = graph.GetNodeArg(input->Name());
    auto& ep_graph_input_arg = ep_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
    ep_graph_inputs.push_back(&ep_graph_input_arg);
  }

  InlinedVector<const NodeArg*> ep_graph_outputs;
  ep_graph_outputs.reserve(outputs.size());
  for (auto& output : outputs) {
    auto output_arg = graph.GetNodeArg(output->Name());
    auto& ep_graph_output_arg = ep_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
    ep_graph_outputs.push_back(&ep_graph_output_arg);
  }

  ep_graph.SetInputs(ep_graph_inputs);
  ep_graph.SetOutputs(ep_graph_outputs);

  for (const auto& node : graph.Nodes()) {
    // the fused node and EPContext node has same node name
    auto ep_context_node = get_ep_context_node(node.Name());
    // Use EpContext node created by the EPs if name matched, otherwise use node from original model
    if (ep_context_node.first) {
      ep_graph.AddNode(*ep_context_node.second);
    } else if (is_invalid_fused_node(node)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Encountered an invalid node while compiling a model. ",
                             "Please ensure the input model is not already compiled.");
    } else {
      ep_graph.AddNode(node);
    }
  }

  // handle initializers
  for (const auto& [name, _] : graph.GetAllInitializedTensors()) {
    if (ep_graph.GetNodeArg(name) != nullptr) {
      graph_utils::MakeInitializerCopyIfNotExist(graph, ep_graph, name);
    }
  }

  ORT_RETURN_IF_ERROR(ep_graph.Resolve());

  // Generate EP compatibility strings for OrtEp types and add to model metadata
  // At this point, the graph has been populated with all the EPContext nodes
  {
    const GraphViewer graph_viewer(ep_graph);
    for (const auto& ep : execution_providers) {
      try {
        // Generate the compatibility string for this EP
        std::string compatibility_string = ep->GetCompiledModelCompatibilityInfo(graph_viewer);
        if (!compatibility_string.empty()) {
          // Create a unique key for this EP's compatibility info
          // Use format: "ep_compatibility_info.<EP_TYPE>"
          // All EPs in a session must have a unique Type() value, so this will be unique for the generated model
          std::string metadata_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + ep->Type();
          auto& model_metadata = ep_context_model.MetaData();
          auto [it, was_inserted] =
              model_metadata.insert_or_assign(metadata_key, compatibility_string);
          if (!was_inserted) {
            LOGS(logger, WARNING) << "Overwriting existing EP compatibility info for key: " << metadata_key << " (EP: " << ep->Type() << ")";
          }
          LOGS(logger, VERBOSE) << "Added EP compatibility info for " << ep->Type() << " with key: " << metadata_key;
        }
      } catch (const std::exception& ex) {
        LOGS(logger, WARNING) << "Failed to generate compatibility string for EP " << ep->Type() << ": " << ex.what();
      }
    }
  }

  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_RETURN_IF_ERROR(EpContextModelToProto(ep_context_model, valid_output_model_path, ep_context_gen_options,
                                            /*out*/ model_proto));

  ORT_RETURN_IF_ERROR(SaveModelProtoToLocation(model_proto, ep_context_gen_options,
                                               valid_output_model_path, logger));

  return Status::OK();
}

}  // namespace epctx
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
