// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#include "core/framework/debug_node_inputs_outputs_utils.h"

#include <iomanip>

#include "core/common/path_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace utils {

namespace {

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

std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  return out << value.ToFloat();
}

std::ostream& operator<<(std::ostream& out, const MLFloat16& value) {
  return out << static_cast<float>(value);
}

template <typename T>
void DumpTensorToStdOut(const Tensor& tensor) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.DataAsSpan<T>();

  auto print_val = [](const T& value) {
    if (std::is_floating_point<T>::value)
      std::cout << std::setprecision(8) << value;
    else
      std::cout << value;
  };

  for (size_t row = 0; row < num_rows; ++row) {
    print_val(data[row * row_size]);
    for (size_t i = 1; i < row_size; ++i) {
      std::cout << ", ";
      print_val(data[row * row_size + i]);
    }
    std::cout << "\n";
  }

  std::cout << std::endl;
}

PathString MakeTensorFileName(const std::string& tensor_name) {
  auto make_valid_name = [](std::string name) {
    std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    return name;
  };

  return path_utils::MakePathString(make_valid_name(tensor_name), ".tensorproto");
}

void DumpTensorToFile(const Tensor& tensor, const std::string& tensor_name, const Path& file_path) {
  auto tensor_proto = utils::TensorToTensorProto(tensor, tensor_name);
  const PathString file_path_str = file_path.ToPathString();
  int output_fd;
  ORT_THROW_IF_ERROR(Env::Default().FileOpenWr(file_path_str, output_fd));
  try {
    ORT_ENFORCE(
        tensor_proto.SerializeToFileDescriptor(output_fd),
        "Failed to write tensor to file - tensor: ", tensor_name, ", file: ", ToMBString(file_path_str));
  } catch (...) {
    ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(output_fd));
    throw;
  }
  ORT_THROW_IF_ERROR(Env::Default().FileClose(output_fd));
}

void DumpCpuTensor(
    const NodeDumpOptions& dump_options,
    const Tensor& tensor, const std::string& tensor_name) {
  switch (dump_options.data_destination) {
    case NodeDumpOptions::DataDestination::StdOut: {
      DispatchOnTensorType(tensor.DataType(), DumpTensorToStdOut, tensor);
      break;
    }
    case NodeDumpOptions::DataDestination::TensorProtoFiles: {
      const Path tensor_file = dump_options.output_dir / Path::Parse(MakeTensorFileName(tensor_name));
      DumpTensorToFile(tensor, tensor_name, tensor_file);
      break;
    }
    default:
      ORT_THROW("Unsupported data destination type: ", static_cast<int>(dump_options.data_destination));
  }
}

void DumpTensor(
    const NodeDumpOptions& dump_options,
    const Tensor& tensor, const std::string& tensor_name,
    const SessionState& session_state) {
  // check tensor is on CPU before dumping it
  auto& tensor_location = tensor.Location();
  const auto data_type = tensor.DataType();
  if (tensor_location.device.Type() == OrtDevice::CPU ||
      tensor_location.mem_type == OrtMemTypeCPUInput ||
      tensor_location.mem_type == OrtMemTypeCPUOutput) {
    DumpCpuTensor(dump_options, tensor, tensor_name);
  } else {
    std::cout << tensor_location << "\n";

#ifdef USE_CUDA
    // Dumping GPU only when cuda is enabled.
    if (tensor_location.device.Type() == OrtDevice::GPU) {
      const auto& execution_providers = session_state.GetExecutionProviders();
      const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      auto cpu_allocator = cpu_execution_provider->GetAllocator(0, OrtMemTypeDefault);
      Tensor cpu_tensor{data_type, tensor.Shape(), cpu_allocator};
      const auto& data_transfer_mgr = session_state.GetDataTransferMgr();
      auto status = data_transfer_mgr.CopyTensor(tensor, cpu_tensor);
      if (status == common::Status::OK()) {
        DumpCpuTensor(dump_options, cpu_tensor, tensor_name);
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

    auto get_bool_env_var = [](const char* env_var) {
      const auto val = Env::Default().GetEnvironmentVar(env_var);
      if (val.empty()) return false;
      std::istringstream s{val};
      int i;
      ORT_ENFORCE(
          s >> i && s.eof(),
          "Failed to parse environment variable ", env_var, ": ", val);
      return i != 0;
    };

    NodeDumpOptions opts{};

    opts.dump_flags = NodeDumpOptions::DumpFlags::ShapeOnly;
    if (get_bool_env_var(env_vars::kDumpInputData)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::InputData;
    }
    if (get_bool_env_var(env_vars::kDumpOutputData)) {
      opts.dump_flags |= NodeDumpOptions::DumpFlags::OutputData;
    }

    opts.filter.name_pattern = Env::Default().GetEnvironmentVar(env_vars::kNameFilter);
    opts.filter.op_type_pattern = Env::Default().GetEnvironmentVar(env_vars::kOpTypeFilter);

    if (get_bool_env_var(env_vars::kDumpDataToFiles)) {
      opts.data_destination = NodeDumpOptions::DataDestination::TensorProtoFiles;
    }

    opts.output_dir = Path::Parse(ToPathString(Env::Default().GetEnvironmentVar(env_vars::kOutputDir)));

    // check for confirmation for dumping data to files for all nodes
    if (opts.dump_flags != NodeDumpOptions::DumpFlags::ShapeOnly &&
        opts.data_destination == NodeDumpOptions::DataDestination::TensorProtoFiles &&
        opts.filter.name_pattern.empty() && opts.filter.op_type_pattern.empty()) {
      ORT_ENFORCE(
          get_bool_env_var(env_vars::kDumpingDataToFilesForAllNodesIsOk),
          "The current environment variable configuration will dump node input or output data to files for every node. "
          "This may cause a lot of files to be generated. Set the environment variable ",
          env_vars::kDumpingDataToFilesForAllNodesIsOk, " to confirm this is what you want.");
    }

    return opts;
  }();

  return node_dump_options;
}

void DumpNodeInputs(
    const NodeDumpOptions& dump_options,
    const OpKernelContext& context, const Node& node, const SessionState& session_state) {
  if (!FilterNode(dump_options, node)) return;

  std::cout << "-----------\n";
  std::cout << node.OpType() << " node: " << node.Name() << "\n";

  const auto& input_defs = node.InputDefs();

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {
      std::cout << "Input " << i << " Name: " << input_defs[i]->Name();

      const auto* type = context.InputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Input<Tensor>(i);
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";

          if ((dump_options.dump_flags & NodeDumpOptions::DumpFlags::InputData) != 0) {
            DumpTensor(dump_options, tensor, input_defs[i]->Name(), session_state);
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << " was missing data type\n";
      }
    } else {
      std::cout << "Input " << i << " is optional and was not provided.\n";
    }
  }
}

void DumpNodeInputs(
    const OpKernelContext& context, const Node& node, const SessionState& session_state) {
  DumpNodeInputs(NodeDumpOptionsFromEnvironmentVariables(), context, node, session_state);
}

void DumpNodeOutputs(const NodeDumpOptions& dump_options, OpKernelContext& context, const Node& node, const SessionState& session_state) {
  if (!FilterNode(dump_options, node)) return;

  std::cout << "-----------\n";
  const auto& output_defs = node.OutputDefs();

  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {
      std::cout << "Output " << i << " Name: " << output_defs[i]->Name();

      const auto* type = context.OutputType(i);
      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Output<Tensor>(i);
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";

          if ((dump_options.dump_flags & NodeDumpOptions::DumpFlags::OutputData) != 0) {
            DumpTensor(dump_options, tensor, output_defs[i]->Name(), session_state);
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << "missing data type\n";
      }
    } else {
      std::cout << "Output " << i << " is optional and was not produced.\n";
    }

    std::cout << std::endl;
  }
}

void DumpNodeOutputs(
    OpKernelContext& context, const Node& node, const SessionState& session_state) {
  DumpNodeOutputs(NodeDumpOptionsFromEnvironmentVariables(), context, node, session_state);
}

}  // namespace utils
}  // namespace onnxruntime

#endif
