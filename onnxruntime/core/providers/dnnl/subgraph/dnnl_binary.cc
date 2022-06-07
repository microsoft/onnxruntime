#include "dnnl_binary.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlBinary::DnnlBinary() {}

void DnnlBinary::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  dnnl::algorithm algo = dnnl_util::OrtOperatorToDnnlAlgorithm(node.OpType());

  // GetMemory in OrtFormat. Broadcasting and mix format binary ops can result in computation failure
  auto binary_src0_mem = sp.GetMemoryInOrtFormat(node.Input(IN_A), eng);
  auto binary_src1_mem = sp.GetMemoryInOrtFormat(node.Input(IN_B), eng);
  auto src_0_ori_md = binary_src0_mem.get_desc();
  auto src_1_ori_md = binary_src1_mem.get_desc();

  auto src_0_dims = src_0_ori_md.dims();
  auto src_1_dims = src_1_ori_md.dims();
  if (src_0_dims.size() != src_1_dims.size()) {
    while (src_0_dims.size() < src_1_dims.size()) {
      src_0_dims.insert(src_0_dims.begin(), 1);
    }
    while (src_0_dims.size() > src_1_dims.size()) {
      src_1_dims.insert(src_1_dims.begin(), 1);
    }
  }

  auto src_0_md = src_0_ori_md.reshape(src_0_dims);
  auto src_1_md = src_1_ori_md.reshape(src_1_dims);

  auto output_shape = src_0_dims;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] == 1) {
      output_shape[i] = src_1_dims[i];
    }
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

  auto binary_d = dnnl::binary::desc(algo, src_0_md, src_1_md, dst_md);
  auto binary_pd = dnnl::binary::primitive_desc(binary_d, eng);

  auto binary_dst_mem = dnnl::memory(binary_pd.dst_desc(), eng);
  auto binary_prim = dnnl::binary(binary_pd);

  sp.AddPrimitive(binary_prim, {{DNNL_ARG_SRC_0, binary_src0_mem},
                                {DNNL_ARG_SRC_1, binary_src1_mem},
                                {DNNL_ARG_DST, binary_dst_mem}});

  if (sp.IsScalar(node.Input(IN_A)) && sp.IsScalar(node.Input(IN_B))) {
    sp.SetMemory(node.Output(OUT_Y), binary_dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_Y), binary_dst_mem);
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
