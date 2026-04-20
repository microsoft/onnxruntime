// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_layernorm.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlLayerNorm::DnnlLayerNorm() {}

/*
Layer Normalization and Skip Layer Normalization implementation:
Layer Norm:
  Inputs:
    0) X - Input Tensor
    1) G - Gamma Tensor (Scaling factor in the LN formula)
    2) B - Bias Tensor (Shift value in the LN formula. Optional)
  Outputs:
    0) Y - Output Tensor
    1) M - Mean Tensor (Optional)
    2) I - Inverse std Tensor (Optional)

               +-----------+
(X) ---------->+           +----------> (Y)
               |           |
(G) ---------->+ LayerNorm +----------> (M)
               |           |
(B) ---------->+           +----------> (I)
               +-----------+



Skip Layer Norm:
  Inputs:
    0) X - Input Tensor
    1) S - Skip Tensor
    2) G - Gamma Tensor (Scaling factor in the LN formula)
    4) E - Beta Tensor (Shift value in the LN formula. Optional)
    4) B - Bias Tensor (Bias when adding X + S. Optional)
  Outputs:
    0) Y - Output Tensor
    1) M - Mean Tensor (Optional)
    2) I - Inverse std Tensor (Optional)

(X) ---------->+-----------+
               |    Add    |
(S) ---------->+-----------+
                     |                           +-----------+
                     |              (X + S + B)  |           |
                     +-----------+-------------->+           +----------> (Y)
                     |    Add    |               |           |
(B) ---------------->+-----------+   (G) ------->+ LayerNorm +----------> (M)
                                                 |           |
                                     (E) ------->+           +----------> (I)
                                                 |           |
                                                 +-----------+

Attributes (epsilon)
*/
void DnnlLayerNorm::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // Get engine
  auto dnnl_engine = sp.GetEngine();

  // Make sure every input's dimension follows the spec
  ValidateDims(sp, node);

  // Optional input flag
  bool shift_exists = false;

  // Input positions
  int shift_pos, scale_pos;

  // Get src desc
  auto src_md = sp.GetMemory(node.Input(IN_INPUT)).get_desc();

  auto out_md = dnnl::memory::desc(src_md.get_dims(), node.Input(IN_INPUT).Type(), dnnl::memory::format_tag::any);

  // Init src mem
  dnnl::memory src_mem;

  // Init out mem
  dnnl::memory out_mem;

  // This contains the layer norm op and its parameters
  ln_components op_comps;
  if (node.OpType() == "SkipLayerNormalization") {
    // Check if shift is available
    shift_exists = node.Input(IN_BETA).Exists();

    // Fix positions for arguments
    shift_pos = IN_BETA;
    scale_pos = IN_SLN_GAMMA;

    // Move the src to GPU if needed
    src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT), src_md, dnnl_engine);

    // Make dst desc, must be same as src
    auto dst_md = dnnl::memory::desc(src_md.get_dims(), node.Output(OUT_OUTPUT).Type(), dnnl::memory::format_tag::any);

    // Add src + skip
    {
      // get skip desc
      auto skip_md = sp.GetMemory(node.Input(IN_SKIP)).get_desc();
      // Move the skip to GPU if needed
      auto skip_mem = sp.GetMemoryAndReshape(node.Input(IN_SKIP), skip_md, dnnl_engine);

      // Create and add primitive
      auto add_skip_pd = dnnl::binary::primitive_desc(dnnl_engine, dnnl::algorithm::binary_add, src_md, skip_md, dst_md);
      auto add_skip = dnnl::binary(add_skip_pd);
      std::unordered_map<int, dnnl::memory> add_skip_mem_map({{DNNL_ARG_SRC_0, src_mem}, {DNNL_ARG_SRC_1, skip_mem}, {DNNL_ARG_DST, src_mem}});
      sp.AddPrimitive(add_skip, add_skip_mem_map);
    }

    // Add src + skip + bias
    if (node.Input(IN_SLN_BIAS).Exists()) {
      // get bias desc
      auto bias_md = sp.GetMemory(node.Input(IN_SLN_BIAS)).get_desc();
      // Move the bias to GPU if needed
      auto bias_mem = sp.GetMemoryAndReshape(node.Input(IN_SLN_BIAS), bias_md, dnnl_engine);
      // Get bias dims
      auto bias_dims = bias_md.get_dims();
      // Get src dims
      auto src_dims = src_md.get_dims();

      // To follow the spec means our bias will always have less dimensions that our input
      // so we add the extra dimensions, reshape it and let OneDNN broadcast the value
      while (bias_dims.size() < src_dims.size()) {
        bias_dims.insert(bias_dims.begin(), 1);
      }
      bias_md = bias_md.reshape(bias_dims);

      // Create and add primitive
      auto add_bias_pd = dnnl::binary::primitive_desc(dnnl_engine, dnnl::algorithm::binary_add, src_md, bias_md, dst_md);
      auto add_bias = dnnl::binary(add_bias_pd);
      std::unordered_map<int, dnnl::memory> add_bias_mem_map({{DNNL_ARG_SRC_0, src_mem}, {DNNL_ARG_SRC_1, bias_mem}, {DNNL_ARG_DST, src_mem}});
      sp.AddPrimitive(add_bias, add_bias_mem_map);
    }

  } else if (node.OpType() == "LayerNormalization") {
    // Check if shift is available
    shift_exists = node.Input(IN_LN_BIAS).Exists();

    // Fix positions for arguments
    shift_pos = IN_LN_BIAS;
    scale_pos = IN_LN_GAMMA;

    // Move the src to GPU if needed
    src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT), src_md, dnnl_engine);

  } else {
    ORT_THROW("Unknown LayerNormalization flavor");
  }

  // X = LayerNornm(X)
  // Check if we are training and need the extra outputs for backprop
  dnnl::prop_kind prop_kind;
#if 0  // defined(ENABLE_TRAINING)
  prop_kind = dnnl::prop_kind::forward_training;
#else
  prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

  // If beta is available use shift, else only scale
  dnnl::normalization_flags op_flags = dnnl::normalization_flags::use_scale;
  if (shift_exists) {
    op_flags |= dnnl::normalization_flags::use_shift;
  }

  // Get epsilon to avoid zero division
  float epsilon = GetEpsilon(node);
  // Primitive desciptor
  auto lnorm_pd = dnnl::layer_normalization_forward::primitive_desc(dnnl_engine, prop_kind, src_md, out_md, epsilon, op_flags);
  // Primitive
  auto lnorm_prim = dnnl::layer_normalization_forward(lnorm_pd);
  out_mem = dnnl::memory(lnorm_pd.dst_desc(), dnnl_engine);

  // Define primitive arguments
  std::unordered_map<int, dnnl::memory> lnorm_args = {{DNNL_ARG_SRC, src_mem},
                                                      {DNNL_ARG_DST, out_mem}};
  // Get gamma
  auto gamma_mem = sp.GetMemory(node.Input(scale_pos));
  gamma_mem = sp.GetMemoryAndReshape(node.Input(scale_pos), gamma_mem.get_desc(), dnnl_engine);
  if (node.Input(scale_pos).Type() != dnnl::memory::data_type::f32) {
    //  casting to fp32 if input with other data type
    auto gamma_md = gamma_mem.get_desc();
    auto dims = gamma_md.get_dims();
    auto strides = gamma_md.get_strides();
    dnnl::memory::dims gamma_strides_vec;
    for (size_t i = 0; i < dims.size(); i++) {
      gamma_strides_vec.push_back(strides[i]);
    }
    auto gamma_mem_f32 = CastAndTransformMemory(sp, gamma_mem, dnnl::memory::data_type::f32, gamma_strides_vec);
    lnorm_args.insert({DNNL_ARG_SCALE, gamma_mem_f32});
  } else {
    //  no casting if input with fp32
    lnorm_args.insert({DNNL_ARG_SCALE, gamma_mem});
  }

  // Get Beta and add shift if available
  if (shift_exists) {
    auto beta_mem = sp.GetMemory(node.Input(shift_pos));
    beta_mem = sp.GetMemoryAndReshape(node.Input(shift_pos), beta_mem.get_desc(), dnnl_engine);
    if (node.Input(shift_pos).Type() != dnnl::memory::data_type::f32) {
      //  casting to fp32 if input with other data type
      auto beta_md = beta_mem.get_desc();
      auto dims = beta_md.get_dims();
      auto strides = beta_md.get_strides();
      dnnl::memory::dims beta_strides_vec;
      for (size_t i = 0; i < dims.size(); i++) {
        beta_strides_vec.push_back(strides[i]);
      }
      auto beta_mem_f32 = CastAndTransformMemory(sp, beta_mem, dnnl::memory::data_type::f32, beta_strides_vec);
      lnorm_args.insert({DNNL_ARG_SHIFT, beta_mem_f32});
    } else {
      //  no casting if input with fp32
      lnorm_args.insert({DNNL_ARG_SHIFT, beta_mem});
    }
  }

// Check outputs used for training
#if 0   // defined(ENABLE_TRAINING)
  // If Mean exists
  if (node.OutputCount() > 1) {
    if (node.Output(OUT_MEAN).Exists()) {
      auto mean_mem = dnnl::memory(lnorm_pd.mean_desc(), dnnl_engine);
      lnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
      sp.SetMemory(node.Output(OUT_MEAN), mean_mem);
    }
    // If Variance exists
    if (node.Output(OUT_INV_STD_VAR).Exists()) {
      auto variance_mem = dnnl::memory(lnorm_pd.variance_desc(), dnnl_engine);
      lnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
      sp.SetMemory(node.Output(OUT_INV_STD_VAR), variance_mem);
    }
  }
#endif  // ENABLE_TRAINING

  sp.AddPrimitive(lnorm_prim, lnorm_args);

  sp.SetMemory(node.Output(OUT_OUTPUT), out_mem, true);
}

void DnnlLayerNorm::ValidateDims(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // Get input and evaluate
  auto input_dims = sp.GetMemory(node.Input(IN_INPUT)).get_desc().get_dims();
  auto input_dims_size = input_dims.size();

  // Check the inputs are supported by OneDNN, this is mandatory since sometimes
  // we can't check the input size in the node capability
  if ((input_dims_size > 5) || (input_dims_size < 2)) {
    ORT_THROW("Input tensor dimensionality is not supported by OneDNN, got ", input_dims_size);
  }

  // To make this function compliant with all possible layernorm flavors,
  // define gamma and shift input position, depending on the operation
  int gamma_pos, shift_pos;
  if (node.OpType() == "SkipLayerNormalization") {
    // Get skip and evaluate
    auto skip_dims = sp.GetMemory(node.Input(IN_SKIP)).get_desc().get_dims();
    if (input_dims != skip_dims) {
      ORT_THROW("Input and skip dimmentions do not match");
    }

    // Check if bias was provided and evaluate
    if (node.Input(IN_SLN_BIAS).Exists()) {
      auto bias_dims = sp.GetMemory(node.Input(IN_SLN_BIAS)).get_desc().get_dims();
      if (bias_dims.size() != 1) {
        ORT_THROW("Bias is expected to have 1 dimension, got ", bias_dims.size());
      }
      if (bias_dims[0] != input_dims[2]) {
        ORT_THROW("Last dimension of bias and input does not match");
      }
    }

    // Define the input position when using SLN
    gamma_pos = IN_SLN_GAMMA;
    shift_pos = IN_BETA;

    // If the op is LayerNorm
  } else {
    // Define the input position when using LN
    gamma_pos = IN_LN_GAMMA;
    shift_pos = IN_LN_BIAS;
  }

  // Get gamma and evaluate
  auto gamma_dims = sp.GetMemory(node.Input(gamma_pos)).get_desc().get_dims();
  if (gamma_dims.size() != 1) {
    ORT_THROW("Gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != input_dims[input_dims_size - 1]) {
    ORT_THROW("Last dimension of gamma and input does not match");
  }

  // Check if shift was provided and evaluate
  if (node.Input(shift_pos).Exists()) {
    auto beta_dims = sp.GetMemory(node.Input(shift_pos)).get_desc().get_dims();
    if (beta_dims.size() != 1) {
      ORT_THROW("Beta is expected to have 1 dimension, got ", beta_dims.size());
    }
    if (beta_dims[0] != input_dims[input_dims_size - 1]) {
      ORT_THROW("Last dimension of beta and input does not match");
    }
  }
}

float DnnlLayerNorm::GetEpsilon(DnnlNode& node) {
  auto attr = node.Attributes().find("epsilon");
  float epsilon = 1e-05f;  // Default value according to ONNX spec
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
    epsilon = attr->second().f();
  }
  return epsilon;
}

dnnl::memory DnnlLayerNorm::CastAndTransformMemory(DnnlSubgraphPrimitive& sp, dnnl::memory& src_mem, dnnl::memory::data_type dst_datatype, dnnl::memory::dims dst_strides) {
  dnnl::memory dst_mem;
  {
    auto eng = sp.GetEngine();

    // Make a new memory descriptor based on the source descriptor and given destination dataype and strides
    auto src_md = src_mem.get_desc();
    dnnl::memory::desc dst_md = dnnl::memory::desc(src_md.get_dims(), dst_datatype, dst_strides);
    dst_mem = dnnl::memory(dst_md, eng);

    // Reorder source memory to destination memory as per the given dataype and strides
    auto reorder_pd = dnnl::reorder::primitive_desc(eng, src_md, eng, dst_md);
    auto reorder = dnnl::reorder(reorder_pd);
    std::unordered_map<int, dnnl::memory> reorder_mem_map({{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, dst_mem}});
    sp.AddPrimitive(reorder, reorder_mem_map);
  }
  return dst_mem;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
