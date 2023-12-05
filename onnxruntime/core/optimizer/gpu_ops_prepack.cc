// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Module Abstract:
//   This module defines the logic for prepacking weights
//   (aka Initializers in onnxruntime) of GPU operators.
//   Unlike CPU operators, overriding the PrePack() method
//   of class OpKernel results in GPU memory fragmentation.
//   So we try to rewrite the weight tensors during graph
//   optimization phase to avoid this problem
//
//   Unfortunately, there are still some seriouse problems
//   with this approach:
//   1. Rewriting of the initializer tensors is restricted
//      by operator shape inferencing rules. For example,
//      there are 3 initializers for MatMulNBits<float16>,
//      we can't combine them into a single initializer.
//      And we have to make sure the operator's shape inference
//      logic does NOT verify the initializer's shape.
//   2. These rewriting logic is tightly coupled to each GPU
//      operators. It really should be defined together with
//      these operators, instead of defining them in a complete
//      different module.
//   3. The logic of prepacking depends on underlying GPU
//      hardware.  Currently this part is hard-coded for SM80.

#if defined(USE_CUDA) && !defined(USE_ROCM)

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/gpu_ops_prepack.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

#include "blk_q4/f16_prepack_sm80.h"

#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {

extern ProviderInfo_CUDA* TryGetProviderInfo_CUDA();

/**
 * @brief Read initialized tensor from protobuf, and store it in ort_value.
 * Keep in mind that ort_value is the owner of the tensor memory after calling this function.
 */
inline Status GetOrtValue(const NodeArg* arg, const Graph& graph, OrtValue& ort_value) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_RETURN_IF_NOT(graph.GetInitializedTensor(arg->Name(), tensor_proto),
                    "Missing initializer for ", arg->Name());

  return utils::TensorProtoToOrtValue(
      Env::Default(), graph.ModelPath(), *tensor_proto,
      std::make_shared<CPUAllocator>(), ort_value);
}

template <typename T>
inline gsl::span<T> make_span(std::string& str) {
  return gsl::make_span(reinterpret_cast<T*>(str.data()), str.size() / sizeof(T));
}

//
// Prepacking logic specific to MatMulNBits<float16> on sm80
//

static inline bool IsNodeMatMulNbitsFp16(const Node& node) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMulNBits", {1}, kMSDomain)) {
    return false;
  }
  const auto* acts = node.InputDefs()[0];
  if (acts == nullptr || acts->Type() == nullptr || acts->Type()->find("float16") == std::string::npos) {
    return false;
  }
  return true;
}

template <int block_size, bool column_quant_blk>
void Sm80BlkQ4PrepackT(
    int rows, int columns,
    gsl::span<const uint8_t> weights,
    gsl::span<const MLFloat16> scales,
    gsl::span<const uint8_t> zp,
    std::string& packed_w,
    std::string& packed_scales,
    std::string& packed_zp) {
  using Base = onnxruntime::cuda::BlockwiseQuantization<
      MLFloat16,
      block_size,
      4,
      column_quant_blk>;
  const auto q_weight_shape = Base::get_quant_weights_shape(rows, columns);
  const auto meta_shape = Base::get_quant_meta_shape(rows, columns);

  packed_w.resize(SafeInt<size_t>(q_weight_shape.product() * sizeof(uint8_t)));
  Base::prepack_weights(
      rows, columns, weights,
      make_span<uint8_t>(packed_w));

  packed_scales.resize(SafeInt<size_t>(meta_shape.product() * sizeof(MLFloat16)));
  Base::prepack_quant_scales(
      rows, columns, scales,
      make_span<MLFloat16>(packed_scales));

  if (!zp.empty()) {
    packed_zp.resize(SafeInt<size_t>(meta_shape.product() * sizeof(uint8_t)));
    Base::prepack_quant_offsets(
        rows, columns, zp,
        make_span<uint8_t>(packed_zp));
  }
}

void Sm80BlkQ4Prepack(
    int block_size, bool column_quant_blk,
    int rows, int columns,
    gsl::span<const uint8_t> weights,
    gsl::span<const MLFloat16> scales,
    gsl::span<const uint8_t> zp,
    std::string& packed_w,
    std::string& packed_scales,
    std::string& packed_zp) {
  switch (block_size) {
    case 16:
      if (column_quant_blk) {
        Sm80BlkQ4PrepackT<16, true>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      } else {
        Sm80BlkQ4PrepackT<16, false>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      }
      break;
    case 32:
      if (column_quant_blk) {
        Sm80BlkQ4PrepackT<32, true>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      } else {
        Sm80BlkQ4PrepackT<32, false>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      }
      break;
    case 64:
      if (column_quant_blk) {
        Sm80BlkQ4PrepackT<64, true>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      } else {
        Sm80BlkQ4PrepackT<64, false>(rows, columns, weights, scales, zp, packed_w, packed_scales, packed_zp);
      }
      break;
    default:
      ORT_THROW("Unsupported block size: ", block_size);
  }
}

/**
 *@brief Prepack weights of the operator MatMulNBits<float16>.
 * The caller should make sure the node is of type MatMulNBits<float16>.
 */
Status PackMatMulNBitsFp16(Node& node, Graph& graph, bool& modified) {
  modified = false;
  int64_t att_i;

  //
  // Verify prepacking is needed and supported
  //
  Status status = graph_utils::TryGetNodeAttribute(node, "prepacked", att_i);
  bool prepacked = status.IsOK() ? att_i != 0 : false;
  if (prepacked) {
    return Status::OK();  // already prepacked, nothing to do
  }

  ORT_RETURN_IF_ERROR(graph_utils::TryGetNodeAttribute<int64_t>(node, "bits", att_i));
  int nbits = SafeInt<int>(att_i);
  if (nbits != 4) {
    return Status::OK();  // only support 4 bits for now
  }

  // A single dimension can not exceed 2G yet.
  ORT_RETURN_IF_ERROR(graph_utils::TryGetNodeAttribute<int64_t>(node, "K", att_i));
  int k = SafeInt<int>(att_i);
  ORT_RETURN_IF_ERROR(graph_utils::TryGetNodeAttribute<int64_t>(node, "N", att_i));
  int n = SafeInt<int>(att_i);

  ORT_RETURN_IF_ERROR(graph_utils::TryGetNodeAttribute<int64_t>(node, "block_size", att_i));
  int block_size = SafeInt<int>(att_i);

  status = graph_utils::TryGetNodeAttribute(node, "column_wise_blocking", att_i);
  bool column_wise_quant_blk = status.IsOK() ? att_i != 0 : true;

  auto* provider_info = TryGetProviderInfo_CUDA();
  ORT_ENFORCE(provider_info != nullptr, "Failed to query CUDA provider info while prepacking cuda operators.");
  int major, minor;
  ORT_ENFORCE(provider_info->GetCurrentGpuDeviceVersion(&major, &minor) == nullptr,
              "Failed to query CUDA device version while prepacking cuda operators.");

  if (!onnxruntime::cuda::BlkQuantGemmSm80Supported(block_size, column_wise_quant_blk, k, n, major, minor)) {
    return Status::OK();  // not supported
  }

  //
  // Verification passed, start prepacking
  //
  auto& node_name = node.Name();
  auto& mutable_input_defs = node.MutableInputDefs();
  if (mutable_input_defs.size() < 3 || mutable_input_defs.size() > 4) {
    return Status::OK();  // not supported
  }

  NodeArg* old_weights_arg = mutable_input_defs[1];
  NodeArg* old_scales_arg = mutable_input_defs[2];
  NodeArg* old_zp_arg = nullptr;

  // holders of the packed weight tensor memory
  std::string packed_weights;
  std::string packed_scales;
  std::string packed_zp;

  {
    // owners of the weight tensor memory, keep around until consumed by the prepacking function
    OrtValue weights_val;
    OrtValue scales_val;
    OrtValue zp_val;

    ORT_RETURN_IF_ERROR(GetOrtValue(old_weights_arg, graph, weights_val));
    const gsl::span<uint8_t const> weights = weights_val.GetMutable<Tensor>()->DataAsSpan<uint8_t>();

    ORT_RETURN_IF_ERROR(GetOrtValue(old_scales_arg, graph, scales_val));
    const gsl::span<MLFloat16 const> scales = scales_val.GetMutable<Tensor>()->DataAsSpan<MLFloat16>();

    gsl::span<uint8_t const> zp;
    if (mutable_input_defs.size() > 3) {
      old_zp_arg = mutable_input_defs[3];
      if (old_zp_arg != nullptr && old_zp_arg->Exists()) {
        ORT_RETURN_IF_ERROR(GetOrtValue(old_zp_arg, graph, zp_val));
        Tensor* zp_tensor_ptr = zp_val.GetMutable<Tensor>();
        if (!zp_tensor_ptr->IsDataType<uint8_t>()) {
          return Status::OK();  // not supported
        }
        zp = zp_tensor_ptr->DataAsSpan<uint8_t>();
      }
    }

    Sm80BlkQ4Prepack(block_size, column_wise_quant_blk, k, n, weights, scales, zp, packed_weights, packed_scales, packed_zp);

#if 0
    // debug print if prepacked tests fail
    std::cout << "   ======  packed weight     ======  " << std::endl << std::hex;
    const gsl::span<uint8_t> packed_weights_span(reinterpret_cast<uint8_t*>(packed_weights.data()), packed_weights.size());
    for (int r = 0; r < k; r++) {
      for (int c = 0; c < n/2; c++) {
        std::cout << std::setw(2) << std::setfill('0') << static_cast<int>(packed_weights_span[c * k + r]) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::dec;
#endif
  }

  //
  // write packed weight tensor to node parameters
  //
  ONNX_NAMESPACE::TensorProto packed_weights_proto;
  packed_weights_proto.set_name(graph.GenerateNodeArgName(node_name + "_prepacked_weight"));
  packed_weights_proto.add_dims(packed_weights.size() / sizeof(uint8_t));
  packed_weights_proto.set_data_type(onnxruntime::utils::ToTensorProtoElementType<uint8_t>());
  packed_weights_proto.set_raw_data(std::move(packed_weights));
  NodeArg& packed_weights_arg = graph_utils::AddInitializer(graph, packed_weights_proto);
  graph.RemoveConsumerNode(old_weights_arg->Name(), &node);
  mutable_input_defs[1] = &packed_weights_arg;
  graph.AddConsumerNode(packed_weights_arg.Name(), &node);

  ONNX_NAMESPACE::TensorProto packed_scales_proto;
  packed_scales_proto.set_name(graph.GenerateNodeArgName(node_name + "_prepacked_scales"));
  packed_scales_proto.add_dims(packed_scales.size() / sizeof(MLFloat16));
  packed_scales_proto.set_data_type(onnxruntime::utils::ToTensorProtoElementType<MLFloat16>());
  packed_scales_proto.set_raw_data(std::move(packed_scales));
  NodeArg& packed_scales_arg = graph_utils::AddInitializer(graph, packed_scales_proto);
  graph.RemoveConsumerNode(old_scales_arg->Name(), &node);
  mutable_input_defs[2] = &packed_scales_arg;
  graph.AddConsumerNode(packed_scales_arg.Name(), &node);

  if (!packed_zp.empty()) {
    ONNX_NAMESPACE::TensorProto packed_zp_proto;
    packed_zp_proto.set_name(graph.GenerateNodeArgName(node_name + "_prepacked_zp"));
    packed_zp_proto.add_dims(packed_zp.size() / sizeof(uint8_t));
    packed_zp_proto.set_data_type(onnxruntime::utils::ToTensorProtoElementType<uint8_t>());
    packed_zp_proto.set_raw_data(std::move(packed_zp));
    NodeArg& packed_zp_arg = graph_utils::AddInitializer(graph, packed_zp_proto);
    graph.RemoveConsumerNode(old_zp_arg->Name(), &node);
    mutable_input_defs[3] = &packed_zp_arg;
    graph.AddConsumerNode(packed_zp_arg.Name(), &node);
  }

  node.AddAttribute("prepacked", static_cast<int64_t>(1));
  modified = true;
  return Status::OK();
}

Status GpuOpsPrepack::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // node was removed as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.GetExecutionProviderType() != onnxruntime::kCudaExecutionProvider) {
      continue;  // only interested in CUDA nodes
    }

    // Run prepack if the node is MatMulNBits<float16>.
    // When we have more operators to support, we should use a map to dispatch the prepack function
    // instead of adding a whole bunch of if branches here.
    if (IsNodeMatMulNbitsFp16(node)) {
      bool packed = false;
      ORT_RETURN_IF_ERROR(PackMatMulNBitsFp16(node, graph, packed));
      modified |= packed;
      continue;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // USE_CUDA && !USE_ROCM
