import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

#
# TWO SUBGRAPHS: These should all be fused together before quantization!
#    SkipLayerNormGelu
#    SkipLayerNormAttention


#   <SkipLayerNormalization/EmbedLayerNormalization>   <MatMul>
#                              \                        /
#                               \                      /
#                                \                    /
#                               <SkipLayerNormalization>
#                                   /             \
#                                  /               \
#                                 /                 \
#                   <SkipLayerNormalization>   <Attention/MatMul>


def generate_nodes(model_name, has_cast, suffix=''):
    #
    # TODO(kreeger): Write me!
    #

    # TODO(kreeger): LEFT OFF RIGHT HERE. NEED TO STITCH TOGETHER A TRANSFORMER SUBGRAPH TO PREP
    #                for fusing!
    nodes = [
        # make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
        helper.make_node("EmbedLayerNormalization", ["input_ids" + suffix], ["shape1_out" + suffix], "shape1" + suffix),
        helper.make_node("SkipLayerNormalization"),
        helper.make_node("Attention"),
        helper.make_node("MatMul"),
        helper.make_node("SkipLayerNormalization"),
        helper.make_node("MatMul"),
        helper.make_node("BiasGelu"),
        helper.make_node("MatMul"),
        helper.make_node("SkipLayerNormalization"),
        helper.make_node("Attention"),
        helper.make_node("SkipLayerNormalization"),
        helper.make_node("MatMul"),
        helper.make_node("BiasGelu"),
        helper.make_node("MatMul"),
        helper.make_node("SkipLayerNormalization"),
        helper.make_node("Gather"),
        helper.make_node("MatMul"),
        helper.make_node("MatMul"),
        helper.make_node("MatMul"),
    ]
    return nodes


def generate_initializers():
    #
    # TODO(kreeger): Write me!
    #
    initializers = [
    ]
    return initializers
    pass

def generate_model(model_name):
    #
    # TODO(kreeger): Write me!
    #
    pass


generate_model('skip_layer_norm_subgraph.py')