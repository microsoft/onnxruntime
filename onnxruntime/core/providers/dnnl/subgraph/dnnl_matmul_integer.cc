// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_matmul_integer.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"

#include <unordered_set>
#include <vector>
#include <string>

namespace onnxruntime {
namespace ort_dnnl {

DnnlMatMulInteger::DnnlMatMulInteger() {}

void DnnlMatMulInteger::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
  std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                     "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
  auto eng = sp.GetEngine();

  bool has_postop_fusion = false;
  std::vector<std::string> post_ops;

  if (node.OpType() == "MatMulIntegerPostOps") {
    has_postop_fusion = true;
    post_ops = node.GetPostOps();

    int binary_count = 0;
    // Check we have enough inputs for MatMul and the binary post ops
    for (size_t i = 0; i < post_ops.size(); ++i) {
      if (binary_ops.count(post_ops[i]) != 0) {
        assert(node.Input(IN_BINARY_0 + binary_count).Exists());
        binary_count++;
      }
    }
  }

  auto src_dims = sp.GetMemory(node.Input(IN_A)).get_desc().get_dims();
  auto weights_dims = sp.GetMemory(node.Input(IN_B)).get_desc().get_dims();

  if (src_dims.size() != weights_dims.size()) {
    while (src_dims.size() < weights_dims.size()) {
      src_dims.insert(src_dims.begin(), 1);
    }
    while (src_dims.size() > weights_dims.size()) {
      weights_dims.insert(weights_dims.begin(), 1);
    }
  }

  auto src_md = dnnl::memory::desc(src_dims, node.Input(IN_A).Type(), dnnl::memory::format_tag::any);
  auto weights_md = dnnl::memory::desc(weights_dims, node.Input(IN_B).Type(), dnnl::memory::format_tag::any);

  auto output_shape = src_dims;
  output_shape.pop_back();
  output_shape.emplace_back(weights_dims.back());
  for (size_t i = 0; i < output_shape.size() - 2; i++) {
    if (output_shape[i] == 1) {
      output_shape[i] = weights_dims[i];
    }
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

  dnnl::primitive_attr matmul_attr;

  bool has_a_zero_point = node.Input(IN_A_ZERO_POINT).Name() != "";
  bool has_b_zero_point = node.Input(IN_B_ZERO_POINT).Name() != "";

  if (has_a_zero_point) {
    matmul_attr.set_zero_points_mask(DNNL_ARG_SRC, /* mask */ 0);
  }

  if (has_b_zero_point) {
    matmul_attr.set_zero_points_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);
  }

  /*
  create a post op binary with possible unsqueezing in order to make sure onednn properly broadcast
  current limitation
  1. is no unsqueeze for matmul output as it is not exposed due to post op fusion
  2. the third input has to be reordered to plain format
     (eg, no memory format propagation if the third input is internal to subgraph)
  3. adding 1s to front (unsqueeze/expand) in logical dims would possibly fail if
     physical layout is not plain format
  */
  if (has_postop_fusion) {
    int binary_count = 0;
    dnnl::post_ops ops;
    for (size_t i = 0; i < post_ops.size(); ++i) {
      dnnl::algorithm algo = dnnl_util::OrtOperatorToDnnlAlgorithm(post_ops[i]);
      // Handle Binary post ops including the input memory
      if (binary_ops.count(post_ops[i]) != 0) {
        auto ori_binary_md = sp.GetMemory(node.Input(IN_BINARY_0 + binary_count).Name()).get_desc();
        auto ori_binary_dims = ori_binary_md.get_dims();
        auto binary_mem_dims = ori_binary_dims;
        if (ori_binary_dims.size() != output_shape.size()) {
          if (ori_binary_dims.size() > output_shape.size()) {
            ORT_THROW("add fusion with matmul output broadcasting by unsqueezing is not supported");
          }
          // expand the input (from the binary op) if needed to support broadcasting
          while (binary_mem_dims.size() < output_shape.size()) {
            binary_mem_dims.insert(binary_mem_dims.begin(), 1);
          }
        }

        // expand the dims by 1s (should always be possible)
        // will throw exception if not possible
        auto binary_md = ori_binary_md.reshape(binary_mem_dims);
        // Possible improvment: use format any to choose the best layout
        ops.append_binary(algo, binary_md);
        binary_count++;
        // Handle Elementwise post ops. Some of these require obtaining an 'alpha' attribute
      } else if (elementwise_ops.count(post_ops[i]) != 0) {
        float post_op_alpha = 0.0;
        switch (algo) {
          case dnnl::algorithm::eltwise_relu: {
            // Need to check operator since both Relu and LeakyRelu are covered by algorithm::eltwise_relu
            if (post_ops[i] == "LeakyRelu") {
              post_op_alpha = GetFloatAttr(node, "alpha", /*default_alpha*/ 0.01f);
            } else {
              post_op_alpha = 0.0;
            }
            break;
          }
          case dnnl::algorithm::eltwise_elu: {
            post_op_alpha = GetFloatAttr(node, "alpha", /*default_alpha*/ 1.0f);
            break;
          }
          case dnnl::algorithm::eltwise_soft_relu: {
            if (post_ops[i] == "Softplus") {
              post_op_alpha = 1.0f;
            }
            break;
          }
          default:
            post_op_alpha = 0.0;
        }
        ops.append_eltwise(algo, post_op_alpha, 0.0f);
      }
    }
    matmul_attr.set_post_ops(ops);
  }

  auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, weights_md, dst_md, matmul_attr);

  auto matmul_src_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(), eng);
  auto matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_B), matmul_pd.weights_desc(), eng);

  auto matmul_dst_mem = dnnl::memory(matmul_pd.dst_desc(), eng);
  auto matmul_prim = dnnl::matmul(matmul_pd);

  std::unordered_map<int, dnnl::memory> mem_map({{DNNL_ARG_SRC, matmul_src_mem},
                                                 {DNNL_ARG_WEIGHTS, matmul_weights_mem},
                                                 {DNNL_ARG_DST, matmul_dst_mem}});

  if (has_a_zero_point) {
    auto zp_A_mem_desc_s32 = dnnl::memory::desc({1}, dnnl::memory::data_type::s32, {1});
    auto& tensor = node.Input(IN_A_ZERO_POINT);
    auto zp_A_mem_s32 = sp.GetMemoryAndReshape(tensor, zp_A_mem_desc_s32, eng);
    mem_map[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = zp_A_mem_s32;
  }

  if (has_b_zero_point) {
    auto zp_B_mem_desc_s32 = dnnl::memory::desc({1}, dnnl::memory::data_type::s32, {1});
    auto& tensor = node.Input(IN_B_ZERO_POINT);
    auto zp_B_mem_s32 = sp.GetMemoryAndReshape(tensor, zp_B_mem_desc_s32, eng);
    mem_map[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zp_B_mem_s32;
  }

  if (has_postop_fusion) {
    // add to memory map for extra binary inputs
    int binary_count = 0;
    for (size_t i = 0; i < post_ops.size(); ++i) {
      if (binary_ops.count(post_ops[i]) != 0) {
        dnnl::algorithm algo;
        dnnl::memory::desc binary_mem_desc;
        matmul_pd.get_primitive_attr().get_post_ops().get_params_binary(static_cast<int>(i), algo, binary_mem_desc);
        auto binary_post_op_mem = sp.GetMemoryAndReshape(node.Input(IN_BINARY_0 + binary_count), binary_mem_desc, eng);
        mem_map[DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(i)) | DNNL_ARG_SRC_1] = binary_post_op_mem;
        binary_count++;
      }
    }
  }

  sp.AddPrimitive(matmul_prim, mem_map /*, {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}*/);

  sp.SetMemory(node.Output(OUT_Y), matmul_dst_mem);
}

float DnnlMatMulInteger::GetFloatAttr(DnnlNode& node, std::string attr_name, float default_value) {
  auto attr = node.Attributes().find(attr_name);
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return default_value;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
