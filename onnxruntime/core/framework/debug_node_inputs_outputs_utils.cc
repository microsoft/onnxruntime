// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#include "core/framework/debug_node_inputs_outputs_utils.h"
#include "core/framework/print_tensor_utils.h"
#include <iomanip>
#include <cctype>
#include <string>

#ifdef DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB
#include <sqlite3.h>
#endif

#include "core/common/path_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace utils {

namespace {

struct TensorMetadata {
  std::string name;
  std::string producer;
  std::string consumer;
  std::string device_type;
  size_t step;
};

bool FilterNode(const NodeDumpOptions& dump_options, const Node& node) {
  auto match_pattern =
      [](const std::string& value, const std::string& delimited_patterns) {
        // match all if empty
        if (delimited_patterns.empty()) return true;

        // search for exact match against delimited patterns
        auto pattern_begin = delimited_patterns.begin();
        while (true) {
          const auto pattern_end = std::find(
              pattern_begin, delimited_patterns.end(), kFilterPatternDelimiter);

          if (std::equal(value.begin(), value.end(), pattern_begin, pattern_end)) return true;

          if (pattern_end == delimited_patterns.end()) break;

          pattern_begin = pattern_end + 1;
        }

        return false;
      };

  return match_pattern(node.Name(), dump_options.filter.name_pattern) &&
         match_pattern(node.OpType(), dump_options.filter.op_type_pattern);
}

template <typename T>
void DumpTensorToStdOut(const Tensor& tensor, const NodeDumpOptions& dump_options) {
  onnxruntime::utils::PrintCpuTensor<T>(tensor, dump_options.snippet_threshold, dump_options.snippet_edge_items);
}

PathString MakeTensorFileName(const std::string& tensor_name, const NodeDumpOptions& dump_options) {
  auto make_valid_name = [](std::string name) {
    std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    return name;
  };

  return path_utils::MakePathString(make_valid_name(tensor_name), dump_options.file_suffix, ".tensorproto");
}

void DumpTensorToFile(const Tensor& tensor, const std::string& tensor_name, const Path& file_path) {
  auto tensor_proto = utils::TensorToTensorProto(tensor, tensor_name);
  const PathString file_path_str = file_path.ToPathString();
  int output_fd;
  ORT_THROW_IF_ERROR(Env::Default().FileOpenWr(file_path_str, output_fd));
  try {
    ORT_ENFORCE(
        tensor_proto.SerializeToFileDescriptor(output_fd),
        "Failed to write tensor to file - tensor: ", tensor_name, ", file: ", ToUTF8String(file_path_str));
  } catch (...) {
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(output_fd));
    throw;
  }
  ORT_THROW_IF_ERROR(Env::Default().FileClose(output_fd));
}

#ifdef DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB
sqlite3* SqliteConnection() {
  static thread_local std::unique_ptr<sqlite3, decltype(&sqlite3_close)> sqlite_db(
      []() {
        std::stringstream ss;
        ss << "-pid" << Env::Default().GetSelfPid() << ".db";
        const auto& opt = NodeDumpOptionsFromEnvironmentVariables();
        auto sqlite_db_prefix = opt.sqlite_db_prefix;
        auto sqlite_db_path = sqlite_db_prefix.Concat(ss.str()).ToPathString();

        sqlite3* db;
        int rc = sqlite3_open(sqlite_db_path.c_str(), &db);
        ORT_ENFORCE(rc == SQLITE_OK, "Failed to connect to sqlite3 db ", sqlite_db_path.c_str());

        const char* sql_create_tensor_table =
            "Create table if not exists Tensors ( "
            "  Step int not null, "
            "  Name text not null, "
            "  Value TensorProto, "
            "  DeviceType text, "
            "  TracedProducer NodeArg, "
            "  TracedConsumers NodeArgList, "
            "  primary key (Step, Name) "
            ");";

        const char* error_message = nullptr;
        rc = sqlite3_exec(db, sql_create_tensor_table, nullptr, 0, (char**)&error_message);
        ORT_ENFORCE(rc == SQLITE_OK,
                    "Failed to create Tensors table in sqlite3 db ", sqlite_db_path.c_str(),
                    " on ", error_message);

        const char* sql_create_node_table =
            "Create table if not exists Nodes ( "
            "  ExecutionCounter int, "
            "  Name text primary key not null, "
            "  OpType text not null, "
            "  ExecutionProvider text "
            ");";

        rc = sqlite3_exec(db, sql_create_node_table, nullptr, 0, (char**)&error_message);
        ORT_ENFORCE(rc == SQLITE_OK,
                    "Failed to create Nodes table in sqlite3 db ", sqlite_db_path.c_str(),
                    " on ", error_message);

        return db;
      }(),
      &sqlite3_close);

  return sqlite_db.get();
}

#define SQL_OK(command) \
  ORT_ENFORCE((command) == SQLITE_OK, "Failed sql operation on ", sqlite3_errmsg(SqliteConnection()))

void SqlStepWithRetry(sqlite3_stmt* stmt, int sql_expected) {
  int attempt = 0;
  while (true) {
    int rc = sqlite3_step(stmt);
    if (rc == sql_expected) {
      return;
    }

    if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) {
      if (attempt % 10000 == 0) {
        std::cerr << "Warning: Pid " << Env::Default().GetSelfPid()
                  << " gently spinning on sql db busy or locked\n";
      }
      Env::Default().SleepForMicroseconds(100);
      attempt++;
      continue;
    }

    ORT_THROW("Failed sql step for ", sqlite3_expanded_sql(stmt), " on ", sqlite3_errmsg(SqliteConnection()));
  }
}

bool TensorExistsInSqlDb(const TensorMetadata& tensor_metadata) {
  static thread_local std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_uptr(
      []() {
        sqlite3* db = SqliteConnection();

        const char* sql_tensor_exists =
            "select count(name) from Tensors where Name == ? and Step == ?;";

        sqlite3_stmt* stmt = nullptr;
        SQL_OK(sqlite3_prepare_v2(db, sql_tensor_exists, -1, &stmt, nullptr));

        return stmt;
      }(),
      &sqlite3_finalize);

  sqlite3_stmt* stmt = stmt_uptr.get();

  SQL_OK(sqlite3_reset(stmt));
  SQL_OK(sqlite3_bind_text(stmt, 1, tensor_metadata.name.c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_int(stmt, 2, (int)tensor_metadata.step));
  SqlStepWithRetry(stmt, SQLITE_ROW);
  bool exists = sqlite3_column_int(stmt, 0) > 0;
  SqlStepWithRetry(stmt, SQLITE_DONE);

  return exists;
}

void InsertTensorInSqlDb(const Tensor& tensor, const TensorMetadata& tensor_metadata) {
  static thread_local std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_uptr(
      []() {
        sqlite3* db = SqliteConnection();

        const char* sql_insert_tensor =
            "Insert into Tensors (Step, Name, Value, DeviceType, TracedProducer, TracedConsumers) "
            " values (?, ?, ?, ?, \"\", \"\"); ";

        sqlite3_stmt* stmt = nullptr;
        SQL_OK(sqlite3_prepare_v2(db, sql_insert_tensor, -1, &stmt, nullptr));

        return stmt;
      }(),
      &sqlite3_finalize);

  sqlite3_stmt* stmt = stmt_uptr.get();

  SQL_OK(sqlite3_reset(stmt));
  SQL_OK(sqlite3_bind_int(stmt, 1, tensor_metadata.step));
  SQL_OK(sqlite3_bind_text(stmt, 2, tensor_metadata.name.c_str(), -1, SQLITE_TRANSIENT));

  auto tensor_proto = utils::TensorToTensorProto(tensor, tensor_metadata.name);
  std::string bytes = tensor_proto.SerializeAsString();
  const char* data = bytes.data();
  int size = bytes.size();

  SQL_OK(sqlite3_bind_blob(stmt, 3, data, size, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_text(stmt, 4, tensor_metadata.device_type.c_str(), -1, SQLITE_TRANSIENT));

  SqlStepWithRetry(stmt, SQLITE_DONE);
}

void UpdateTensorUsageInSqlDb(const TensorMetadata& tensor_metadata) {
  static thread_local std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_uptr(
      []() {
        sqlite3* db = SqliteConnection();

        const char* sql_update_tensor =
            "Update Tensors set "
            "  TracedProducer = TracedProducer || ?, "
            "  TracedConsumers = TracedConsumers || ? "
            "where Name = ? and Step = ?;";

        sqlite3_stmt* stmt = nullptr;
        SQL_OK(sqlite3_prepare_v2(db, sql_update_tensor, -1, &stmt, nullptr));

        return stmt;
      }(),
      &sqlite3_finalize);

  sqlite3_stmt* stmt = stmt_uptr.get();

  SQL_OK(sqlite3_reset(stmt));
  SQL_OK(sqlite3_bind_text(stmt, 1, tensor_metadata.producer.c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_text(stmt, 2, tensor_metadata.consumer.c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_text(stmt, 3, tensor_metadata.name.c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_int(stmt, 4, tensor_metadata.step));

  SqlStepWithRetry(stmt, SQLITE_DONE);
}

void DumpTensorToSqliteDb(const Tensor& tensor, const TensorMetadata& tensor_metadata) {
  if (!TensorExistsInSqlDb(tensor_metadata)) {
    InsertTensorInSqlDb(tensor, tensor_metadata);
  }

  UpdateTensorUsageInSqlDb(tensor_metadata);
}

void InsertNodePlacementToSqliteDb(const NodeDumpContext& dump_context, const Node& node) {
  static thread_local std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_uptr(
      []() {
        sqlite3* db = SqliteConnection();

        const char* sql_insert_node =
            "Insert or Ignore into Nodes (ExecutionCounter, Name, OpType, ExecutionProvider) "
            " values (?, ?, ?, ?);";

        sqlite3_stmt* stmt = nullptr;
        SQL_OK(sqlite3_prepare_v2(db, sql_insert_node, -1, &stmt, nullptr));

        return stmt;
      }(),
      &sqlite3_finalize);

  sqlite3_stmt* stmt = stmt_uptr.get();

  SQL_OK(sqlite3_reset(stmt));
  SQL_OK(sqlite3_bind_int(stmt, 1, dump_context.program_counter));
  SQL_OK(sqlite3_bind_text(stmt, 2, node.Name().c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_text(stmt, 3, node.OpType().c_str(), -1, SQLITE_TRANSIENT));
  SQL_OK(sqlite3_bind_text(stmt, 4, node.GetExecutionProviderType().c_str(), -1, SQLITE_TRANSIENT));

  SqlStepWithRetry(stmt, SQLITE_DONE);
}
#endif  // DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB

void DumpCpuTensor(
    const NodeDumpOptions& dump_options,
    const Tensor& tensor, const TensorMetadata& tensor_metadata) {
  switch (dump_options.data_destination) {
    case NodeDumpOptions::DataDestination::StdOut: {
      DispatchOnTensorType(tensor.DataType(), DumpTensorToStdOut, tensor, dump_options);
      break;
    }
    case NodeDumpOptions::DataDestination::TensorProtoFiles: {
      const Path tensor_file = dump_options.output_dir / Path::Parse(MakeTensorFileName(tensor_metadata.name, dump_options));
      DumpTensorToFile(tensor, tensor_metadata.name, tensor_file);
      break;
    }
    case NodeDumpOptions::DataDestination::SqliteDb: {
#ifdef DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB
      DumpTensorToSqliteDb(tensor, tensor_metadata);
#else
      ORT_THROW("Recompile with --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1 onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB=1");
#endif
      break;
    }
    default:
      ORT_THROW("Unsupported data destination type: ", static_cast<int>(dump_options.data_destination));
  }
}

void DumpTensor(
    const NodeDumpOptions& dump_options,
    const Tensor& tensor, TensorMetadata& tensor_metadata,
    const SessionState& session_state) {
  // check tensor is on CPU before dumping it
  auto& tensor_location = tensor.Location();
  if (tensor_location.device.Type() == OrtDevice::CPU ||
      tensor_location.mem_type == OrtMemTypeCPUInput ||
      tensor_location.mem_type == OrtMemTypeCPUOutput) {
    tensor_metadata.device_type = "CPU";
    DumpCpuTensor(dump_options, tensor, tensor_metadata);
  } else {
    std::cout << tensor_location << "\n";

#if defined(USE_CUDA) || defined(USE_ROCM)
    const auto data_type = tensor.DataType();
    // Dumping GPU only when cuda is enabled.
    if (tensor_location.device.Type() == OrtDevice::GPU) {
      const auto& execution_providers = session_state.GetExecutionProviders();
      const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      auto cpu_allocator = session_state.GetAllocator(cpu_execution_provider->GetOrtDeviceByMemType(OrtMemTypeDefault));
      Tensor cpu_tensor{data_type, tensor.Shape(), cpu_allocator};
      const auto& data_transfer_mgr = session_state.GetDataTransferMgr();
      auto status = data_transfer_mgr.CopyTensor(tensor, cpu_tensor);
      if (status == common::Status::OK()) {
        tensor_metadata.device_type = "GPU";
        DumpCpuTensor(dump_options, cpu_tensor, tensor_metadata);
      } else {
        std::cout << " failed to transfer data to cpu.\n";
      }
    }
#else
    ORT_UNUSED_PARAMETER(session_state);
#endif
  }
}

}  // namespace

const NodeDumpOptions& NodeDumpOptionsFromEnvironmentVariables() {
  static const NodeDumpOptions node_dump_options = []() {
    namespace env_vars = debug_node_inputs_outputs_env_vars;

    NodeDumpOptions opts{};

    // Preserve existing behavior of printing the shapes by default. Turn it off only if the user has requested so
    // explicitly by setting the value of the env variable to 0.
    opts.dump_flags = NodeDumpOptions::DumpFlags::None;
    if (ParseEnvironmentVariableWithDefault<bool>(env_vars::kDumpShapeData, true)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::Shape;
    }

    if (ParseEnvironmentVariableWithDefault<bool>(env_vars::kDumpInputData, false)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::InputData;
    }
    if (ParseEnvironmentVariableWithDefault<bool>(env_vars::kDumpOutputData, false)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::OutputData;
    }
    if (ParseEnvironmentVariableWithDefault<bool>(env_vars::kDumpNodePlacement, true)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::NodePlacement;
    }

    opts.filter.name_pattern = Env::Default().GetEnvironmentVar(env_vars::kNameFilter);
    opts.filter.op_type_pattern = Env::Default().GetEnvironmentVar(env_vars::kOpTypeFilter);

    const std::string destination = ParseEnvironmentVariableWithDefault<std::string>(
        env_vars::kDumpDataDestination, "stdout");

    if (destination == "files") {
      opts.data_destination = NodeDumpOptions::DataDestination::TensorProtoFiles;
    } else if (destination == "sqlite") {
      opts.data_destination = NodeDumpOptions::DataDestination::SqliteDb;
    } else if (destination != "stdout") {
      ORT_THROW("Unsupported data destination type: ", destination);
    }

    // Snippet options for StdOut
    opts.snippet_threshold = ParseEnvironmentVariableWithDefault<int>(env_vars::kSnippetThreshold, kDefaultSnippetThreshold);
    opts.snippet_edge_items = ParseEnvironmentVariableWithDefault<int>(env_vars::kSnippetEdgeItems, kDefaultSnippetEdgeItems);

    if (ParseEnvironmentVariableWithDefault<bool>(env_vars::kAppendRankToFileName, false)) {
      std::string rank = Env::Default().GetEnvironmentVar("OMPI_COMM_WORLD_RANK");
      if (rank.empty()) {
        opts.file_suffix = "_default_rank_0";
      } else {
        opts.file_suffix = "_rank_" + rank;
      }
    }

    opts.output_dir = Path::Parse(ToPathString(Env::Default().GetEnvironmentVar(env_vars::kOutputDir)));

    std::string sqlite_db_prefix =
        ParseEnvironmentVariableWithDefault<std::string>(env_vars::kSqliteDbPrefix, "execution-trace");
    opts.sqlite_db_prefix = Path::Parse(ToPathString(sqlite_db_prefix));

    // check for confirmation for dumping data to files for all nodes
    const bool is_input_or_output_requested = ((opts.dump_flags & NodeDumpOptions::DumpFlags::InputData) != 0) ||
                                              ((opts.dump_flags & NodeDumpOptions::DumpFlags::OutputData) != 0);

    if (is_input_or_output_requested &&
        opts.data_destination == NodeDumpOptions::DataDestination::TensorProtoFiles &&
        opts.filter.name_pattern.empty() && opts.filter.op_type_pattern.empty()) {
      ORT_ENFORCE(
          ParseEnvironmentVariableWithDefault<bool>(env_vars::kDumpingDataToFilesForAllNodesIsOk, false),
          "The current environment variable configuration will dump node input or output data to files for every node. "
          "This may cause a lot of files to be generated. Set the environment variable ",
          env_vars::kDumpingDataToFilesForAllNodesIsOk, " to confirm this is what you want.");
    }

    return opts;
  }();

  return node_dump_options;
}

static bool IsAnyOutputDumped(const NodeDumpOptions& dump_options) {
  return dump_options.dump_flags != NodeDumpOptions::DumpFlags::None;
}

static void PrintIf(bool boolean_expression, const std::string& message) {
  if (boolean_expression) {
    std::cout << message;
  }
}

void DumpNodeInputs(
    const NodeDumpOptions& dump_options,
    const NodeDumpContext& dump_context,
    const OpKernelContext& context,
    const Node& node,
    const SessionState& session_state) {
  const bool is_any_output_dumped = IsAnyOutputDumped(dump_options);
  if (!is_any_output_dumped) {
    return;
  }

  if (!FilterNode(dump_options, node)) return;

  if (context.GetComputeStream())
    context.GetComputeStream()->Flush();

  bool should_dump_node_placement = (dump_options.dump_flags & NodeDumpOptions::DumpFlags::NodePlacement) != 0;
  if (dump_context.iteration == 1 && should_dump_node_placement) {
    PrintIf(should_dump_node_placement, MakeString(" Placement: ", node.GetExecutionProviderType(), "\n"));
#ifdef DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB
    InsertNodePlacementToSqliteDb(dump_context, node);
#endif
  }

  std::cout << "-----------\n";
  std::cout << node.OpType() << " node: " << node.Name() << "\n";

  const auto& input_defs = node.InputDefs();
  TensorMetadata tensor_metadata;

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {
      std::cout << "Input " << i << " Name: " << input_defs[i]->Name() << "\n";

      const auto* type = context.InputType(i);

      if (type) {
        if (type->IsTensorType()) {
          if (const auto* tensor = context.Input<Tensor>(i); tensor != nullptr) {
            const auto& shape = tensor->Shape();

            const bool is_shape_set = (dump_options.dump_flags & NodeDumpOptions::DumpFlags::Shape) != 0;
            PrintIf(is_shape_set, MakeString(" Shape: ", shape, "\n"));

            if ((dump_options.dump_flags & NodeDumpOptions::DumpFlags::InputData) != 0) {
              tensor_metadata.name = input_defs[i]->Name();
              tensor_metadata.step = dump_context.iteration;
              tensor_metadata.consumer = node.Name() + ":" + std::to_string(i);
              DumpTensor(dump_options, *tensor, tensor_metadata, session_state);
            }
          } else {
            std::cout << " is empty optional tensor.\n";
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // this could happen with an empty Optional input
        std::cout << " was missing data type\n";
      }
    } else {
      std::cout << "Input " << i << " is optional and was not provided.\n";
    }
  }
}

void DumpNodeInputs(
    const NodeDumpContext& dump_context,
    const OpKernelContext& context,
    const Node& node,
    const SessionState& session_state) {
  DumpNodeInputs(NodeDumpOptionsFromEnvironmentVariables(), dump_context, context, node, session_state);
}

void DumpNodeOutputs(
    const NodeDumpOptions& dump_options,
    const NodeDumpContext& dump_context,
    OpKernelContext& context,
    const Node& node,
    const SessionState& session_state) {
  const bool is_any_output_dumped = IsAnyOutputDumped(dump_options);
  if (!is_any_output_dumped) {
    return;
  }

  if (!FilterNode(dump_options, node)) return;

  if (context.GetComputeStream())
    context.GetComputeStream()->Flush();

  bool should_dump_node_placement = (dump_options.dump_flags & NodeDumpOptions::DumpFlags::NodePlacement) != 0;
  if (dump_context.iteration == 1 && should_dump_node_placement) {
    PrintIf(should_dump_node_placement, MakeString(" Placement: ", node.GetExecutionProviderType(), "\n"));
#ifdef DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB
    InsertNodePlacementToSqliteDb(dump_context, node);
#endif
  }

  std::cout << "-----------\n";
  const auto& output_defs = node.OutputDefs();
  TensorMetadata tensor_metadata;

  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {
      std::cout << "Output " << i << " Name: " << output_defs[i]->Name() << "\n";

      const auto* type = context.OutputType(i);
      if (type) {
        if (type->IsTensorType()) {
          if (const auto* tensor = context.Output<Tensor>(i); tensor != nullptr) {
            const auto& shape = tensor->Shape();

            const bool is_shape_set = (dump_options.dump_flags & NodeDumpOptions::DumpFlags::Shape) != 0;
            PrintIf(is_shape_set, MakeString(" Shape: ", shape, "\n"));

            if ((dump_options.dump_flags & NodeDumpOptions::DumpFlags::OutputData) != 0) {
              tensor_metadata.name = output_defs[i]->Name();
              tensor_metadata.step = dump_context.iteration;
              tensor_metadata.producer = node.Name() + ":" + std::to_string(i);
              DumpTensor(dump_options, *tensor, tensor_metadata, session_state);
            }
          } else {
            std::cout << " is empty optional tensor.\n";
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen in a successful run
        std::cout << "missing data type\n";
      }
    } else {
      std::cout << "Output " << i << " is optional and was not produced.\n";
    }

    std::cout << std::endl;
  }
}

void DumpNodeOutputs(
    const NodeDumpContext& dump_context,
    OpKernelContext& context,
    const Node& node,
    const SessionState& session_state) {
  DumpNodeOutputs(NodeDumpOptionsFromEnvironmentVariables(), dump_context, context, node, session_state);
}

}  // namespace utils
}  // namespace onnxruntime

#endif
