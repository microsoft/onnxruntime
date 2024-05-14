// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_matmul.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"
#include <vector>
#include <unordered_set>
#include <string>

namespace onnxruntime {
namespace ort_dnnl {

inline static dnnl::memory::format_tag get_default_format(const dnnl::memory::dims& tensor_dims) {
  switch (tensor_dims.size()) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    default:
      return dnnl::memory::format_tag::undef;
  }
}

DnnlMatMul::DnnlMatMul() {}

// This handles ONNX defined "MatMul" as well as two other variations of MatMul
// "MatMulPostOps" is a OneDNN only fusion of MatMul and upto 32 elementwise or binary ops.
//    See dnnl_subgraph_transformer.cc MatMulBinaryEltwise(...).
// "FusedMatMul" is a ContribOperator defined here:
//    https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
//    Depending on its attributes "FusedMatMul" can transpose eather input to the MatMul and scale the resulting output
void DnnlMatMul::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
  std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                     "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};

  auto eng = sp.GetEngine();

  bool has_postop_fusion = false;
  std::vector<std::string> post_ops;

  if (node.OpType() == "MatMulPostOps") {
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

  bool is_fusedmatmul = false;
  bool transA = false;
  bool transBatchA = false;
  bool transB = false;
  bool transBatchB = false;
  float alpha = 1.0;
  if (node.OpType() == "FusedMatMul") {
    // Fused matmul is matmul modified to behave like numpy:
    // https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
    is_fusedmatmul = true;
    transA = GetTransA(node);
    transBatchA = GetTransBatchA(node);
    transB = GetTransB(node);
    transBatchB = GetTransBatchB(node);
    alpha = GetAlpha(node);
  }

  auto src_dims = sp.GetMemory(node.Input(IN_A)).get_desc().get_dims();
  auto weights_dims = sp.GetMemory(node.Input(IN_B)).get_desc().get_dims();

  // If this is required for transposed inputs, then this will be done later on in the code.
  if (src_dims.size() != weights_dims.size()) {
    while (src_dims.size() < weights_dims.size() && (!transA && !transBatchA)) {
      src_dims.insert(src_dims.begin(), 1);
    }
    while (src_dims.size() > weights_dims.size() && (!transB && !transBatchB)) {
      weights_dims.insert(weights_dims.begin(), 1);
    }
  }

  auto dataA_dims = src_dims;
  auto ndataA_dims = src_dims.size();
  dnnl::memory::dims transposedA_dims(ndataA_dims, 0);

  auto dataB_dims = weights_dims;
  auto ndataB_dims = weights_dims.size();
  dnnl::memory::dims transposedB_dims(ndataB_dims, 0);

  auto dataA_mem = sp.GetMemory(node.Input(IN_A));
  auto dataB_mem = sp.GetMemory(node.Input(IN_B));

  // Holds transposed matrices A and B. ToDo: Eliminate its usage if in place transpose is possbile for FusedMatmul
  dnnl::memory::desc transposedA_md;
  dnnl::memory transposedA_mem;

  dnnl::memory::desc transposedB_md;
  dnnl::memory transposedB_mem;

  if (is_fusedmatmul) {
    if (transA || transBatchA) {
      dnnl::memory::dims strides = GetStrides(dataA_dims, transA, transBatchA, transposedA_dims);

      dnnl::memory::desc intermediateA_md = dnnl::memory::desc(dataA_dims, node.Input(IN_A).Type(), strides);
      dnnl::memory intermediateA_mem = dnnl::memory(intermediateA_md, eng);

      auto traspose_primitive = dnnl::reorder(dataA_mem, intermediateA_mem);
      sp.AddPrimitive(traspose_primitive, {{DNNL_ARG_FROM, dataA_mem}, {DNNL_ARG_TO, intermediateA_mem}});

      while (transposedA_dims.size() < weights_dims.size()) {
        transposedA_dims.insert(transposedA_dims.begin(), 1);
      }

      // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
      // that will have the correct dimentions and correct memory::format
      transposedA_md = dnnl::memory::desc(transposedA_dims, node.Input(IN_A).Type(), sp.GetDnnlFormat(transposedA_dims.size()));
      transposedA_mem = dnnl::memory(transposedA_md, eng, nullptr);
      void* handle = intermediateA_mem.get_data_handle();
      transposedA_mem.set_data_handle(handle);
    }
    if (transB || transBatchB) {  // Exact same logic for matrix B as used for matrix A
      dnnl::memory::dims strides = GetStrides(dataB_dims, transB, transBatchB, transposedB_dims);

      dnnl::memory::desc intermediateB_md = dnnl::memory::desc(dataB_dims, node.Input(IN_B).Type(), strides);
      dnnl::memory intermediateB_mem = dnnl::memory(intermediateB_md, eng);

      auto traspose_primitive = dnnl::reorder(dataB_mem, intermediateB_mem);
      sp.AddPrimitive(traspose_primitive, {{DNNL_ARG_FROM, dataB_mem}, {DNNL_ARG_TO, intermediateB_mem}});

      while (src_dims.size() > transposedB_dims.size()) {
        transposedB_dims.insert(transposedB_dims.begin(), 1);
      }

      // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
      // that will have the correct dimentions and correct memory::format
      transposedB_md = dnnl::memory::desc(transposedB_dims, node.Input(IN_B).Type(), sp.GetDnnlFormat(transposedB_dims.size()));
      transposedB_mem = dnnl::memory(transposedB_md, eng, nullptr);
      void* handle = intermediateB_mem.get_data_handle();
      transposedB_mem.set_data_handle(handle);
    }
  }

  dnnl::memory::desc src_md;
  if (transA || transBatchA) {
    src_md = transposedA_md;
  } else {
    src_md = dnnl::memory::desc(src_dims, node.Input(IN_A).Type(), get_default_format(src_dims));
  }

  dnnl::memory::desc weights_md;
  if (transB || transBatchB) {
    weights_md = transposedB_md;
  } else {
    weights_md = dnnl::memory::desc(weights_dims, node.Input(IN_B).Type(), get_default_format(weights_dims));
  }

  auto output_shape = src_dims;
  if (transA || transBatchA) {
    output_shape = transposedA_dims;
  }
  output_shape.pop_back();
  if (transB || transBatchB) {
    output_shape.emplace_back(transposedB_dims.back());
  } else {
    output_shape.emplace_back(weights_dims.back());
  }

  for (size_t i = 0; i < output_shape.size() - 2; i++) {
    if (output_shape[i] == 1) {
      if (transB || transBatchB) {
        output_shape[i] = transposedB_dims[i];
      } else {
        output_shape[i] = weights_dims[i];
      }
    }
  }

  /*
  create a post op binary with possible unsqueezing in order to make sure onednn properly broadcast
  current limitation
  1. is no unsqueeze for matmul output as it is not exposed due to post op fusion
  2. the third input has to be reordered to plain format (eg, no memory format propogation if the third input is internal to subgraph)
  3. adding 1s to front (unsqueeze/expand) in logical dims would possibly fail if physcial layout is not plain format
  */
  dnnl::primitive_attr attr;
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
    attr.set_post_ops(ops);
  }

  if (is_fusedmatmul) {
    // Set the value to scale DNNL_ARG_SRC with mask 0
    attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), get_default_format(output_shape));

  auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, weights_md, dst_md, attr);

  dnnl::memory matmul_src_mem, matmul_weights_mem;
  auto matmul_dst_mem = dnnl::memory(matmul_pd.dst_desc(), eng);
  auto matmul_prim = dnnl::matmul(matmul_pd);

  if (transA || transBatchA) {
    matmul_src_mem = transposedA_mem;
  } else {
    matmul_src_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(), eng);
  }
  if (transB || transBatchB) {
    matmul_weights_mem = transposedB_mem;
  } else {
    matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_B), matmul_pd.weights_desc(), eng);
  }

  // a default memory map for matmul
  std::unordered_map<int, dnnl::memory> mem_map({{DNNL_ARG_SRC, matmul_src_mem},
                                                 {DNNL_ARG_WEIGHTS, matmul_weights_mem},
                                                 {DNNL_ARG_DST, matmul_dst_mem}});

  if (is_fusedmatmul) {
    // Create the memory object related to the scale
    auto alpha_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng);
    // Write the alpha value into the memory object
    sp.WriteToDnnlMemory<float>(alpha_mem, {alpha});
    // Set alpha_mem to scale the output
    mem_map.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_mem});
  }

  // add to memory map with extra third input if fused with add
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

  sp.AddPrimitive(matmul_prim, mem_map);
  sp.SetMemory(node.Output(OUT_Y), matmul_dst_mem);
}

dnnl::memory::dims DnnlMatMul::GetStrides(dnnl::memory::dims& data_dims,
                                          bool trans,
                                          bool transBatch,
                                          dnnl::memory::dims& transposed_dims) {
  std::vector<uint32_t> permA;
  std::vector<uint32_t> N_A;
  auto ndata_dims = data_dims.size();
  uint32_t M_A, Batch;
  // Temp vector to hold indices of the dims, will be used to track transposes required
  for (uint32_t i = 0; i < ndata_dims; i++)
    permA.push_back(i);
  Batch = permA[0];             // Batch Dimension
  M_A = permA[ndata_dims - 1];  // M Dimension
  if (ndata_dims == 4)          // This will only be used if transBatch is true
    N_A.push_back(permA[ndata_dims - 3]);
  N_A.push_back(permA[ndata_dims - 2]);
  if (trans && !transBatch) {  // Swap last two dimensions for Trans only
    auto n = permA[ndata_dims - 1];
    permA[ndata_dims - 1] = permA[ndata_dims - 2];
    permA[ndata_dims - 2] = n;
  } else if (!trans && transBatch) {  // If transBatch only, {Batch, N, M} ---> {N, Batch, M}
    uint32_t i;
    for (i = 0; i < N_A.size(); i++) {
      permA[i] = N_A[i];
    }
    permA[i] = Batch;
  } else {  // If both trans and transBatch is true, then end result should be {Batch, N, M} ----> {N, M, Batch}
    uint32_t i;
    for (i = 0; i < N_A.size(); i++) {
      permA[i] = N_A[i];
    }
    permA[i] = M_A;
    permA[i + 1] = Batch;
  }
  dnnl::memory::dims strides(ndata_dims, 0);
  dnnl::memory::dim total_stride = 1;
  for (int i = (int)ndata_dims - 1; i >= 0; i--) {
    transposed_dims[i] = data_dims[permA[i]];
    strides[permA[i]] = total_stride;
    total_stride *= data_dims[permA[i]];
  }

  dnnl::memory::dims strides_inverse;
  strides_inverse.reserve(ndata_dims);
  for (size_t i = 0; i < ndata_dims; ++i) {
    strides_inverse.push_back(strides[ndata_dims - i - 1]);
  }

  return strides;
}

bool DnnlMatMul::GetTransA(DnnlNode& node) {
  auto attr = node.Attributes().find("transA");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}

bool DnnlMatMul::GetTransBatchA(DnnlNode& node) {
  auto attr = node.Attributes().find("transBatchA");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}

bool DnnlMatMul::GetTransB(DnnlNode& node) {
  auto attr = node.Attributes().find("transB");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}

bool DnnlMatMul::GetTransBatchB(DnnlNode& node) {
  auto attr = node.Attributes().find("transBatchB");
  if (attr != node.Attributes().end()) {
    return (attr->second().i() != 0);
  }
  return false;
}

float DnnlMatMul::GetAlpha(DnnlNode& node) {
  auto attr = node.Attributes().find("alpha");
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return 1.0;
}

float DnnlMatMul::GetFloatAttr(DnnlNode& node, std::string attr_name, float default_value) {
  auto attr = node.Attributes().find(attr_name);
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return default_value;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
