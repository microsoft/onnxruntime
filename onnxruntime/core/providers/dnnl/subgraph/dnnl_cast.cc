// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_cast.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlCast::DnnlCast() {}

void DnnlCast::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // Get the DNNL engine
  auto dnnl_engine = sp.GetEngine();

  // Get the memory from the input node
  auto src_mem = sp.GetMemory(node.Input(IN_INPUT));
  auto src_tag = node.Input(IN_INPUT).Format();
  auto src_md = src_mem.get_desc();
  auto src_dims = src_md.get_dims();

  // dst characteristics
  dnnl::memory::data_type dst_type;
  dnnl::memory::format_tag dst_tag;

  // Get the target data type
  auto dst_type_desc = GetTo(node);

  // Check fot the target datat ype
  switch (dst_type_desc) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      dst_type = dnnl::memory::data_type::f32;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      dst_type = dnnl::memory::data_type::f16;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
      dst_type = dnnl::memory::data_type::bf16;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      dst_type = dnnl::memory::data_type::s32;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      dst_type = dnnl::memory::data_type::s8;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      dst_type = dnnl::memory::data_type::u8;
      break;
    }
    default:
      ORT_THROW("Unsupported data type: ", dst_type_desc);
      break;
  }
  // Be aware that the output memory will be in plain format
  // and depending on the operation you do next, this wont be as
  // efficient as you'd like
  // If the format tag is any
  if (src_tag == dnnl::memory::format_tag::any) {
    // Define a plain data ND format
    dst_tag = sp.GetDnnlFormat(src_dims.size());
  } else {
    // Else use the same as the source
    dst_tag = src_tag;
  }

  // Generate the dst memory descriptor
  auto dst_md = dnnl::memory::desc(src_md.get_dims(), dst_type, dst_tag);

  // Create the reorder primitive descriptor.
  auto reorder_pd = dnnl::reorder::primitive_desc(dnnl_engine, src_md, dnnl_engine, dst_md);
  // Get the dst memory
  auto dst_mem = dnnl::memory(reorder_pd.dst_desc(), dnnl_engine);

  // If using GPU this will move the memory from the CPU to the GPU.
  src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT), reorder_pd.src_desc(), dnnl_engine);

  // OneDNN uses reorder to cast the src_md data to the dst_md data type
  auto reorder = dnnl::reorder(reorder_pd);

  // Add primitive to the graph
  sp.AddPrimitive(reorder, {{DNNL_ARG_SRC, src_mem},
                            {DNNL_ARG_DST, dst_mem}});

  // Support scalar return values
  if (sp.IsScalar(node.Input(OUT_OUTPUT))) {
    sp.SetMemory(node.Output(OUT_OUTPUT), dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_OUTPUT), dst_mem);
  }
}

int64_t DnnlCast::GetTo(DnnlNode& node) {
  // Get the attribute
  auto attr = node.Attributes().find("to");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  } else {
    // to attribute should always exist in order to cast
    ORT_THROW("TO(CAST TARGET DATA TYPE) DOES NOT EXIST");
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime