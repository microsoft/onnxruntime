import google.protobuf.text_format
import onnx
from numpy import array, float16

import onnxruntime as ort

# Run n times
N = 1

onnx_model_text = """
ir_version: 8
producer_name: "pytorch"
producer_version: "2.2.0"
graph {
  node {
    output: "_val_1"
    name: "Constant_0"
    op_type: "Constant"
    attribute {
      name: "value_ints"
      ints: -1
      type: INTS
    }
    doc_string: ""
  }
  node {
    input: "input_0"
    input: "_val_1"
    output: "_val_2"
    name: "Reshape_1"
    op_type: "Reshape"
    attribute {
      name: "allowzero"
      i: 0
      type: INT
    }
    doc_string: ""
  }
  node {
    input: "_val_2"
    output: "_val_3"
    name: "_aten_linalg_vector_norm_no_dim_onnx_2"
    op_type: "_aten_linalg_vector_norm_no_dim_onnx"
    attribute {
      name: "keepdim"
      i: 0
      type: INT
    }
    attribute {
      name: "ord"
      f: 2.0
      type: FLOAT
    }
    doc_string: ""
    domain: "pkg.onnxscript.torch_lib"
  }
  name: "main_graph"
  input {
    name: "input_0"
    type {
      tensor_type {
        elem_type: 10
        shape {
        }
      }
    }
  }
  output {
    name: "_val_3"
    type {
      tensor_type {
        elem_type: 10
        shape {
        }
      }
    }
  }
  value_info {
    name: "_val_1"
    type {
      tensor_type {
        elem_type: 7
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  value_info {
    name: "_val_2"
    type {
      tensor_type {
        elem_type: 10
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  domain: "pkg.onnxscript.torch_lib"
  version: 1
}
opset_import {
  domain: ""
  version: 18
}
opset_import {
  domain: "pkg.onnxscript.torch_lib.common"
  version: 1
}
functions {
  name: "_aten_linalg_vector_norm_no_dim_onnx"
  input: "self"
  output: "result_29"
  attribute: "ord"
  attribute: "keepdim"
  node {
    input: "self"
    output: "tmp"
    name: "n0"
    op_type: "Shape"
    domain: ""
  }
  node {
    input: "tmp"
    output: "self_rank"
    name: "n1"
    op_type: "Size"
    domain: ""
  }
  node {
    output: "int64_0"
    name: "n2"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 7
        int64_data: 0
        name: "int64_0"
      }
      type: TENSOR
    }
    domain: ""
  }
  node {
    input: "int64_0"
    input: "self_rank"
    output: "int64_0_cast"
    name: "n3"
    op_type: "CastLike"
    domain: ""
  }
  node {
    input: "self_rank"
    input: "int64_0_cast"
    output: "cond"
    name: "n4"
    op_type: "Equal"
    domain: ""
  }
  node {
    input: "cond"
    output: "self_2"
    name: "n5"
    op_type: "If"
    attribute {
      name: "then_branch"
      g {
        node {
          output: "int64_0_1d"
          name: "n0"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 1
              data_type: 7
              int64_data: 0
              name: "int64_0_1d"
            }
            type: TENSOR
          }
          domain: ""
        }
        node {
          input: "self"
          input: "int64_0_1d"
          output: "self_0"
          name: "n1"
          op_type: "Unsqueeze"
          domain: ""
        }
        name: "thenGraph_4"
        output {
          name: "self_0"
          type {
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "else_branch"
      g {
        node {
          input: "self"
          output: "self_1"
          name: "n0"
          op_type: "Identity"
          domain: ""
        }
        name: "elseGraph_4"
        output {
          name: "self_1"
          type {
          }
        }
      }
      type: GRAPH
    }
    domain: ""
  }
  node {
    input: "self_2"
    output: "self_3"
    name: "n6"
    op_type: "Abs"
    domain: ""
  }
  node {
    output: "ord"
    name: "n7"
    op_type: "Constant"
    attribute {
      name: "value_float"
      type: FLOAT
      ref_attr_name: "ord"
    }
    domain: ""
  }
  node {
    input: "ord"
    output: "ord_4"
    name: "n8"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 1
      type: INT
    }
    domain: ""
  }
  node {
    input: "ord_4"
    output: "cond_5"
    name: "n9"
    op_type: "IsInf"
    attribute {
      name: "detect_negative"
      i: 0
      type: INT
    }
    attribute {
      name: "detect_positive"
      i: 1
      type: INT
    }
    domain: ""
  }
  node {
    input: "cond_5"
    output: "result_24"
    name: "n10"
    op_type: "If"
    attribute {
      name: "then_branch"
      g {
        node {
          input: "self_3"
          output: "result"
          name: "n0"
          op_type: "ReduceMax"
          attribute {
            name: "keepdims"
            type: INT
            ref_attr_name: "keepdim"
          }
          domain: ""
        }
        name: "thenGraph_9"
        output {
          name: "result"
          type {
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "else_branch"
      g {
        node {
          input: "ord_4"
          output: "cond_6"
          name: "n0"
          op_type: "IsInf"
          attribute {
            name: "detect_negative"
            i: 1
            type: INT
          }
          attribute {
            name: "detect_positive"
            i: 0
            type: INT
          }
          domain: ""
        }
        node {
          input: "cond_6"
          output: "result_23"
          name: "n1"
          op_type: "If"
          attribute {
            name: "then_branch"
            g {
              node {
                input: "self_3"
                output: "result_7"
                name: "n0"
                op_type: "ReduceMin"
                attribute {
                  name: "keepdims"
                  type: INT
                  ref_attr_name: "keepdim"
                }
                domain: ""
              }
              name: "thenGraph_11"
              output {
                name: "result_7"
                type {
                }
              }
            }
            type: GRAPH
          }
          attribute {
            name: "else_branch"
            g {
              node {
                output: "const"
                name: "n0"
                op_type: "Constant"
                attribute {
                  name: "value"
                  t {
                    data_type: 1
                    float_data: 0.0
                    name: "const"
                  }
                  type: TENSOR
                }
                domain: ""
              }
              node {
                input: "const"
                input: "ord_4"
                output: "const_cast"
                name: "n1"
                op_type: "CastLike"
                domain: ""
              }
              node {
                input: "ord_4"
                input: "const_cast"
                output: "cond_8"
                name: "n2"
                op_type: "Equal"
                domain: ""
              }
              node {
                input: "cond_8"
                output: "result_22"
                name: "n3"
                op_type: "If"
                attribute {
                  name: "then_branch"
                  g {
                    node {
                      input: "self_3"
                      output: "self_bool"
                      name: "n0"
                      op_type: "Cast"
                      attribute {
                        name: "to"
                        i: 9
                        type: INT
                      }
                      domain: ""
                    }
                    node {
                      input: "self_bool"
                      input: "self_3"
                      output: "self_0_1"
                      name: "n1"
                      op_type: "CastLike"
                      domain: ""
                    }
                    node {
                      input: "self_0_1"
                      output: "result_9"
                      name: "n2"
                      op_type: "ReduceSum"
                      attribute {
                        name: "keepdims"
                        i: 0
                        type: INT
                      }
                      domain: ""
                    }
                    name: "thenGraph_13"
                    output {
                      name: "result_9"
                      type {
                      }
                    }
                  }
                  type: GRAPH
                }
                attribute {
                  name: "else_branch"
                  g {
                    node {
                      output: "const_10"
                      name: "n0"
                      op_type: "Constant"
                      attribute {
                        name: "value"
                        t {
                          data_type: 1
                          float_data: 1.0
                          name: "const_10"
                        }
                        type: TENSOR
                      }
                      domain: ""
                    }
                    node {
                      input: "const_10"
                      input: "ord_4"
                      output: "const_10_cast"
                      name: "n1"
                      op_type: "CastLike"
                      domain: ""
                    }
                    node {
                      input: "ord_4"
                      input: "const_10_cast"
                      output: "cond_11"
                      name: "n2"
                      op_type: "Equal"
                      domain: ""
                    }
                    node {
                      input: "cond_11"
                      output: "result_21"
                      name: "n3"
                      op_type: "If"
                      attribute {
                        name: "then_branch"
                        g {
                          node {
                            input: "self_3"
                            output: "result_12"
                            name: "n0"
                            op_type: "ReduceL1"
                            attribute {
                              name: "keepdims"
                              type: INT
                              ref_attr_name: "keepdim"
                            }
                            domain: ""
                          }
                          name: "thenGraph_18"
                          output {
                            name: "result_12"
                            type {
                            }
                          }
                        }
                        type: GRAPH
                      }
                      attribute {
                        name: "else_branch"
                        g {
                          node {
                            output: "const_13"
                            name: "n0"
                            op_type: "Constant"
                            attribute {
                              name: "value"
                              t {
                                data_type: 1
                                float_data: 2.0
                                name: "const_13"
                              }
                              type: TENSOR
                            }
                            domain: ""
                          }
                          node {
                            input: "const_13"
                            input: "ord_4"
                            output: "const_13_cast"
                            name: "n1"
                            op_type: "CastLike"
                            domain: ""
                          }
                          node {
                            input: "ord_4"
                            input: "const_13_cast"
                            output: "cond_14"
                            name: "n2"
                            op_type: "Equal"
                            domain: ""
                          }
                          node {
                            input: "cond_14"
                            output: "result_20"
                            name: "n3"
                            op_type: "If"
                            attribute {
                              name: "then_branch"
                              g {
                                node {
                                  input: "self_3"
                                  output: "result_15"
                                  name: "n0"
                                  op_type: "ReduceL2"
                                  attribute {
                                    name: "keepdims"
                                    type: INT
                                    ref_attr_name: "keepdim"
                                  }
                                  domain: ""
                                }
                                name: "thenGraph_20"
                                output {
                                  name: "result_15"
                                  type {
                                  }
                                }
                              }
                              type: GRAPH
                            }
                            attribute {
                              name: "else_branch"
                              g {
                                node {
                                  input: "ord_4"
                                  input: "self_3"
                                  output: "ord_float"
                                  name: "n0"
                                  op_type: "CastLike"
                                  domain: ""
                                }
                                node {
                                  input: "self_3"
                                  input: "ord_float"
                                  output: "self_pow"
                                  name: "n1"
                                  op_type: "Pow"
                                  domain: ""
                                }
                                node {
                                  input: "self_pow"
                                  output: "tmp_16"
                                  name: "n2"
                                  op_type: "ReduceSum"
                                  attribute {
                                    name: "keepdims"
                                    type: INT
                                    ref_attr_name: "keepdim"
                                  }
                                  domain: ""
                                }
                                node {
                                  output: "const_17"
                                  name: "n3"
                                  op_type: "Constant"
                                  attribute {
                                    name: "value"
                                    t {
                                      data_type: 1
                                      float_data: 1.0
                                      name: "const_17"
                                    }
                                    type: TENSOR
                                  }
                                  domain: ""
                                }
                                node {
                                  input: "const_17"
                                  input: "ord_float"
                                  output: "const_17_cast"
                                  name: "n4"
                                  op_type: "CastLike"
                                  domain: ""
                                }
                                node {
                                  input: "const_17_cast"
                                  input: "ord_float"
                                  output: "tmp_18"
                                  name: "n5"
                                  op_type: "Div"
                                  domain: ""
                                }
                                node {
                                  input: "tmp_16"
                                  input: "tmp_18"
                                  output: "result_19"
                                  name: "n6"
                                  op_type: "Pow"
                                  domain: ""
                                }
                                name: "elseGraph_20"
                                output {
                                  name: "result_19"
                                  type {
                                  }
                                }
                              }
                              type: GRAPH
                            }
                            domain: ""
                          }
                          name: "elseGraph_18"
                          output {
                            name: "result_20"
                            type {
                            }
                          }
                        }
                        type: GRAPH
                      }
                      domain: ""
                    }
                    name: "elseGraph_13"
                    output {
                      name: "result_21"
                      type {
                      }
                    }
                  }
                  type: GRAPH
                }
                domain: ""
              }
              name: "elseGraph_11"
              output {
                name: "result_22"
                type {
                }
              }
            }
            type: GRAPH
          }
          domain: ""
        }
        name: "elseGraph_9"
        output {
          name: "result_23"
          type {
          }
        }
      }
      type: GRAPH
    }
    domain: ""
  }
  node {
    output: "int64_0_25"
    name: "n11"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 7
        int64_data: 0
        name: "int64_0_25"
      }
      type: TENSOR
    }
    domain: ""
  }
  node {
    input: "int64_0_25"
    input: "self_rank"
    output: "int64_0_25_cast"
    name: "n12"
    op_type: "CastLike"
    domain: ""
  }
  node {
    input: "self_rank"
    input: "int64_0_25_cast"
    output: "cond_26"
    name: "n13"
    op_type: "Equal"
    domain: ""
  }
  node {
    input: "cond_26"
    output: "result_29"
    name: "n14"
    op_type: "If"
    attribute {
      name: "then_branch"
      g {
        node {
          input: "result_24"
          output: "result_27"
          name: "n0"
          op_type: "Squeeze"
          domain: ""
        }
        name: "thenGraph_27"
        output {
          name: "result_27"
          type {
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "else_branch"
      g {
        node {
          input: "result_24"
          output: "result_28"
          name: "n0"
          op_type: "Identity"
          domain: ""
        }
        name: "elseGraph_27"
        output {
          name: "result_28"
          type {
          }
        }
      }
      type: GRAPH
    }
    domain: ""
  }
  opset_import {
    domain: ""
    version: 18
  }
  domain: "pkg.onnxscript.torch_lib"
}
functions {
  name: "Rank"
  input: "input"
  output: "return_val"
  node {
    input: "input"
    output: "tmp"
    name: "n0"
    op_type: "Shape"
    domain: ""
  }
  node {
    input: "tmp"
    output: "return_val"
    name: "n1"
    op_type: "Size"
    domain: ""
  }
  doc_string: "Take the rank of the input tensor."
  opset_import {
    domain: ""
    version: 18
  }
  domain: "pkg.onnxscript.torch_lib.common"
}
functions {
  name: "IsScalar"
  input: "input"
  output: "return_val"
  node {
    input: "input"
    output: "tmp"
    name: "n0"
    op_type: "Shape"
    domain: ""
  }
  node {
    input: "tmp"
    output: "tmp_0"
    name: "n1"
    op_type: "Size"
    domain: ""
  }
  node {
    output: "tmp_1"
    name: "n2"
    op_type: "Constant"
    attribute {
      name: "value_int"
      i: 0
      type: INT
    }
    domain: ""
  }
  node {
    input: "tmp_0"
    input: "tmp_1"
    output: "return_val"
    name: "n3"
    op_type: "Equal"
    domain: ""
  }
  doc_string: "Return whether the input has rank 0, or is a scalar."
  opset_import {
    domain: ""
    version: 18
  }
  domain: "pkg.onnxscript.torch_lib.common"
}

"""

ort_inputs = {"input_0": array(0.8965, dtype=float16)}

# Set up the inference session
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
onnx_model = onnx.ModelProto()
google.protobuf.text_format.Parse(onnx_model_text, onnx_model)

# Uncomment this line to save the model to a file for examination
# onnx.save_model(onnx_model, "transform_nested_ifs_toplogical_sorted_nodes.onnx")

onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(onnx_model.SerializeToString(), session_options, providers=("CPUExecutionProvider",))

# Run the model
for _ in range(N):
    ort_outputs = session.run(None, ort_inputs)
