// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_gemm.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlGemm::DnnlGemm() {}

/*
Gemm implementation:
Gemm:
  Inputs:
    0) A - Input Tensor
    1) B - Input Tensor
    2) C - Input Tensor (optional if Opset is 11 or later)
  Outputs:
    0) Y - Output Tensor

               +-----------+
    (A)        |           |
    ---------->+           |     AB               +------+
    (B)        | MatMul    +--------------------->+      | alphaAB
    ---------->+           |     (alpha)          | Mul  +---+
               |           |     *--------------->+      |   |     +------+
               +-----------+                      +------+   +---->+      |   (Y) alphaAB + betaC
                                                                   | Add  +---------------------->
    (C)                                           +------+   +---->+      |
    --------------------------------------------->+      |   |     +------+
                                 (beta)           | Mul  +---+
                                 *--------------->+      | betaC
                                                  +------+

Attributes (alpha, beta, transA, transB)

To compose Gemm: (algorithm)
(1) perform `MatMul` on input tensors A and B result (AB)
(2) if `Mul` the result of (1) by alpha attribute (alphaAB)
(3) if C is optional return result from (2) and end
(4) if C is avalible `Mul` input C tensor by beta attribute (betaC)
(5) `Add` result from (2) to result from (4) (alphaAB + betaC)
(6) Return output from (5) and end

OneDNN algorithm:
(1) perform `MatMul` of tensor A and tensor B with `Output scales` set to alpha (0)
(2) if C is optional return output from (1) and end
(3) if C is avalible perform binary `Add` of output from (0) and input C with input C's `scale` attribute set to beta
(4) return output from (4) and end

*/

void DnnlGemm::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  auto a_dims = sp.GetMemory(node.Input(IN_A)).get_desc().get_dims();
  auto b_dims = sp.GetMemory(node.Input(IN_B)).get_desc().get_dims();

  bool input_c_exists = node.Input(IN_C).Exists();

  if (a_dims.size() != b_dims.size()) {
    while (a_dims.size() < b_dims.size()) {
      a_dims.insert(a_dims.begin(), 1);
    }
    while (a_dims.size() > b_dims.size()) {
      b_dims.insert(b_dims.begin(), 1);
    }
  }

  dnnl::memory::desc a_md;
  dnnl::memory::desc b_md;

  bool transA = GetTransA(node);
  bool transB = GetTransB(node);

  dnnl::memory::dim M = (transA) ? a_dims[1] : a_dims[0];
  dnnl::memory::dim K = (transA) ? a_dims[0] : a_dims[1];
  dnnl::memory::dim N = (transB) ? b_dims[0] : b_dims[1];

  dnnl::memory::dims a_strides = (transA) ? dnnl::memory::dims{dnnl::memory::dim(1), M} : dnnl::memory::dims{K, dnnl::memory::dim(1)};
  dnnl::memory::dims b_strides = (transB) ? dnnl::memory::dims{dnnl::memory::dim(1), K} : dnnl::memory::dims{N, dnnl::memory::dim(1)};

  a_md = dnnl::memory::desc({M, K}, node.Input(IN_A).Type(), a_strides);
  b_md = dnnl::memory::desc({K, N}, node.Input(IN_B).Type(), b_strides);

  dnnl::memory::dims output_shape{M, N};

  dnnl::primitive_attr matmul_attr;
  // scale the output from MatMul to alpha
  float alpha = GetAlpha(node);
  // Set the value to scale DNNL_ARG_SRC with mask 0
  matmul_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  // Create the memory object related to the scale
  auto alpha_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng);
  // Write the alpha value into the memory object
  sp.WriteToDnnlMemory<float>(alpha_mem, {alpha});

  auto matmul_dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), {N, 1});

  dnnl::matmul::primitive_desc matmul_pd;
  matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, matmul_dst_md, matmul_attr);

  auto matmul_a_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(), eng, transA);
  auto matmul_b_mem = sp.GetMemoryAndReshape(node.Input(IN_B), matmul_pd.weights_desc(), eng, transB);
  auto gemm_dst_mem = dnnl::memory(matmul_pd.dst_desc(), eng);

  auto matmul_op = dnnl::matmul(matmul_pd);

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, matmul_a_mem});
  args.insert({DNNL_ARG_WEIGHTS, matmul_b_mem});
  args.insert({DNNL_ARG_DST, gemm_dst_mem});
  // Set alpha_mem to scale the output
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_mem});

  sp.AddPrimitive(matmul_op, args);

  if (input_c_exists) {
    auto c_original_md = sp.GetMemory(node.Input(IN_C)).get_desc();
    auto c_dims = c_original_md.get_dims();
    if (c_dims.size() != a_dims.size()) {
      while (c_dims.size() < a_dims.size()) {
        c_dims.insert(c_dims.begin(), 1);
      }
    }

    auto c_md = c_original_md.reshape(c_dims);

    auto y_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

    // Scale input C by beta before adding it to the MatMul output.
    dnnl::primitive_attr binary_attr;
    float beta = GetBeta(node);
    // Set the value to scale DNNL_ARG_SRC_1 with mask 0
    binary_attr.set_scales_mask(DNNL_ARG_SRC_1, 0);
    // Create the memory object related to the scale
    auto beta_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng);
    // Write the alpha value into the memory object
    sp.WriteToDnnlMemory<float>(beta_mem, {beta});

    auto binary_pd = dnnl::binary::primitive_desc(eng, dnnl::algorithm::binary_add,
                                                  matmul_pd.dst_desc(), c_md, y_md, binary_attr);

    auto binary_c_mem = sp.GetMemoryAndReshape(node.Input(IN_C), binary_pd.src1_desc(), eng);

    auto binary_op = dnnl::binary(binary_pd);

    sp.AddPrimitive(binary_op, {{DNNL_ARG_SRC_0, gemm_dst_mem},
                                {DNNL_ARG_SRC_1, binary_c_mem},
                                {DNNL_ARG_DST, gemm_dst_mem},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, beta_mem}});
  }
  sp.SetMemory(node.Output(OUT_Y), gemm_dst_mem);
}

float DnnlGemm::GetAlpha(DnnlNode& node) {
  auto attr = node.Attributes().find("alpha");
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return 1.0;
}
float DnnlGemm::GetBeta(DnnlNode& node) {
  auto attr = node.Attributes().find("beta");
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return 1.0;
}

bool DnnlGemm::GetTransA(DnnlNode& node) {
  auto attr = node.Attributes().find("transA");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}

bool DnnlGemm::GetTransB(DnnlNode& node) {
  auto attr = node.Attributes().find("transB");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
