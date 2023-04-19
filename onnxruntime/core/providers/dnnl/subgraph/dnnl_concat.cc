// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_concat.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlConcat::DnnlConcat() {}

void DnnlConcat::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  int64_t input_rank = -1;
  std::vector<dnnl::memory::desc> src_mds;
  for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
    const auto& input_tensor = node.Input(static_cast<int>(IN_DATA_0 + i));
    if (input_rank == -1) {
      // Tensor rank is assumed to be the same for all inputs
      const auto tensor_rank = static_cast<int64_t>(input_tensor.Dim().size());
      if (tensor_rank > 0) {
        input_rank = tensor_rank;
      }
    }
    auto src_mem = sp.GetMemory(input_tensor);
    src_mds.push_back(src_mem.get_desc());
  }

  auto axis = GetAxis(node, input_rank != -1 ? input_rank : 0);

  // Create primitive descriptor
  auto concat_pd = dnnl::concat::primitive_desc(static_cast<int>(axis), src_mds, dnnl_engine);

  // Create primitive memory objects
  std::vector<dnnl::memory> concat_src_mems;
  for (size_t i = 0; i < src_mds.size(); ++i) {
    auto concat_src_mem = sp.GetMemoryAndReshape(
        node.Input(static_cast<int>(IN_DATA_0 + i)),
        concat_pd.src_desc(static_cast<int>(i)),
        dnnl_engine);
    concat_src_mems.push_back(concat_src_mem);
  }
  auto concat_dst_mem = dnnl::memory(concat_pd.dst_desc(), dnnl_engine);

  // Create primitive arguments
  std::unordered_map<int, dnnl::memory> concat_args;
  for (int n = 0; n < static_cast<int>(concat_src_mems.size()); ++n)
    concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, concat_src_mems[n]});
  concat_args.insert({DNNL_ARG_DST, concat_dst_mem});

  // Create and execute primitive
  auto concat_op = dnnl::concat(concat_pd);

  sp.AddPrimitive(concat_op, concat_args);
  sp.SetMemory(node.Output(OUT_CONCAT), concat_dst_mem);
}

int64_t DnnlConcat::GetAxis(DnnlNode& node, int64_t input_rank) {
  auto axis_attr = node.Attributes().find("axis");
  ORT_ENFORCE(axis_attr != node.Attributes().end(),
              "Axis attribute is not provided");
  ORT_ENFORCE(axis_attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT,
              "Axis value is not an integer");

  int64_t signed_axis = axis_attr->second().i();
  if (signed_axis < 0) {
    signed_axis += input_rank;
  }
  return signed_axis;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
