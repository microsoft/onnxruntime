#!/usr/bin/env python3
"""Standalone minimal problem for minimal_repro.onnx — the ORT 1.26 ConstantFolding
HasExternalDataInMemory reproducer. Generated from the model; run to recreate
an identical file.

    python3 onnxruntime/test/testdata/gh_issue_29071_if_constant_folding.py [out.onnx]
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build():
    graph_nodes = []
    graph_nodes.append(helper.make_node("Shape", ["/ScatterND_4_output_0"], ["/Shape_10_output_0"], name="/Shape_10"))
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_46_output_0"],
            name="/Constant_46",
            value=numpy_helper.from_array(
                np.array([-1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Gather", ["/Shape_10_output_0", "/Constant_46_output_0"], ["/Gather_5_output_0"], name="/Gather_5"
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Reshape",
            ["/Gather_5_output_0", "/TopK_K_shape"],
            ["/Gather_5_output_0_1d"],
            name="/Gather_5_output_0_reshape_1d",
        )
    )
    graph_nodes.append(
        helper.make_node(
            "TopK",
            ["/ScatterND_4_output_0", "/Gather_5_output_0_1d"],
            ["/TopK_output_0", "/TopK_output_1"],
            name="/TopK",
            axis=-1,
            largest=1,
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_47_output_0"],
            name="/Constant_47",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_48_output_0"],
            name="/Constant_48",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_49_output_0"],
            name="/Constant_49",
            value=numpy_helper.from_array(
                np.array([20], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_50_output_0"],
            name="/Constant_50",
            value=numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Slice",
            [
                "/TopK_output_1",
                "/Constant_48_output_0",
                "/Constant_49_output_0",
                "/Constant_47_output_0",
                "/Constant_50_output_0",
            ],
            ["/Slice_5_output_0"],
            name="/Slice_5",
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Gather", ["/ScatterND_4_output_0", "/Slice_5_output_0"], ["/Gather_7_output_0"], name="/Gather_7", axis=0
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_51_output_0"],
            name="/Constant_51",
            value=numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Unsqueeze", ["/Gather_7_output_0", "/Constant_51_output_0"], ["/Unsqueeze_6_output_0"], name="/Unsqueeze_6"
        )
    )
    graph_nodes.append(
        helper.make_node("Mul", ["/Unsqueeze_6_output_0", "/Gemm_output_0"], ["/Mul_9_output_0"], name="/Mul_9")
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_52_output_0"],
            name="/Constant_52",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Unsqueeze", ["/Gather_7_output_0", "/Constant_52_output_0"], ["/Unsqueeze_7_output_0"], name="/Unsqueeze_7"
        )
    )
    graph_nodes.append(
        helper.make_node("Mul", ["/Mul_9_output_0", "/Unsqueeze_7_output_0"], ["/Mul_10_output_0"], name="/Mul_10")
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_54_output_0"],
            name="/Constant_54",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(helper.make_node("Shape", ["/Mul_10_output_0"], ["/Shape_11_output_0"], name="/Shape_11"))
    graph_nodes.append(
        helper.make_node(
            "Gather", ["/Shape_11_output_0", "/Constant_54_output_0"], ["/Gather_8_output_0"], name="/Gather_8", axis=0
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_55_output_0"],
            name="/Constant_55",
            value=numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(helper.make_node("Shape", ["/Mul_10_output_0"], ["/Shape_12_output_0"], name="/Shape_12"))
    graph_nodes.append(
        helper.make_node(
            "Gather", ["/Shape_12_output_0", "/Constant_55_output_0"], ["/Gather_9_output_0"], name="/Gather_9", axis=0
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_57_output_0"],
            name="/Constant_57",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node("Sub", ["/Gather_9_output_0", "/Constant_57_output_0"], ["/Sub_5_output_0"], name="/Sub_5")
    )
    graph_nodes.append(
        helper.make_node("Min", ["/Gather_8_output_0", "/Sub_5_output_0"], ["/Min_output_0"], name="/Min")
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_58_output_0"],
            name="/Constant_58",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node("Max", ["/Min_output_0", "/Constant_58_output_0"], ["/Max_1_output_0"], name="/Max_1")
    )
    graph_nodes.append(
        helper.make_node("Concat", ["/Max_1_output_0"], ["/Concat_3_output_0"], name="/Concat_3", axis=0)
    )
    graph_nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["/Concat_3_output_0"],
            ["/ConstantOfShape_1_output_0"],
            name="/ConstantOfShape_1",
            value=numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_59_output_0"],
            name="/Constant_59",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "CumSum", ["/ConstantOfShape_1_output_0", "/Constant_59_output_0"], ["/CumSum_output_0"], name="/CumSum"
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_60_output_0"],
            name="/Constant_60",
            value=numpy_helper.from_array(
                np.array([-1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node("Add", ["/CumSum_output_0", "/Constant_60_output_0"], ["/Add_3_output_0"], name="/Add_3")
    )
    graph_nodes.append(
        helper.make_node("Concat", ["/Concat_3_output_0"], ["/Concat_4_output_0"], name="/Concat_4", axis=0)
    )
    graph_nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["/Concat_4_output_0"],
            ["/ConstantOfShape_2_output_0"],
            name="/ConstantOfShape_2",
            value=numpy_helper.from_array(
                np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    graph_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_61_output_0"],
            name="/Constant_61",
            value=numpy_helper.from_array(np.array([0], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([]), name=""),
        )
    )
    graph_nodes.append(
        helper.make_node("Equal", ["/Concat_3_output_0", "/Constant_61_output_0"], ["/Equal_output_0"], name="/Equal")
    )
    graph_nodes.append(helper.make_node("Not", ["/Equal_output_0"], ["/Not_2_output_0"], name="/Not_2"))
    g1_nodes = []
    g1_nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["/Concat_4_output_0"],
            ["/ConstantOfShape_3_output_0"],
            name="/ConstantOfShape_3",
            value=numpy_helper.from_array(
                np.array([0.0], dtype=helper.tensor_dtype_to_np_dtype(1)).reshape([1]), name=""
            ),
        )
    )
    g1 = helper.make_graph(
        g1_nodes,
        "sub_graph1",
        [],
        [
            helper.make_tensor_value_info(
                "/ConstantOfShape_3_output_0", 1, ["ConstantOfShape/ConstantOfShape_3_output_0_dim_0"]
            )
        ],
        initializer=[],
    )
    g2_nodes = []
    g2_nodes.append(
        helper.make_node("Add", ["/ConstantOfShape_2_output_0", "/Add_3_output_0"], ["/Add_4_output_0"], name="/Add_4")
    )
    g2_nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/Constant_62_output_0"],
            name="/Constant_62",
            value=numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name=""
            ),
        )
    )
    g2_nodes.append(
        helper.make_node(
            "Unsqueeze", ["/Add_4_output_0", "/Constant_62_output_0"], ["/Unsqueeze_8_output_0"], name="/Unsqueeze_8"
        )
    )
    g2_nodes.append(
        helper.make_node(
            "GatherND",
            ["/ReduceSum_3_output_0", "/Unsqueeze_8_output_0"],
            ["/GatherND_3_output_0"],
            name="/GatherND_3",
            batch_dims=0,
        )
    )
    g2 = helper.make_graph(
        g2_nodes,
        "sub_graph",
        [],
        [helper.make_tensor_value_info("/GatherND_3_output_0", 1, ["GatherND/GatherND_3_output_0_dim_0"])],
        initializer=[],
    )
    graph_nodes.append(
        helper.make_node("If", ["/Not_2_output_0"], ["/If_output_0"], name="/If", else_branch=g1, then_branch=g2)
    )
    graph = helper.make_graph(
        graph_nodes,
        "main_graph",
        [
            helper.make_tensor_value_info("/ScatterND_4_output_0", 1, [2000]),
            helper.make_tensor_value_info("/Gemm_output_0", 1, ["unk__14", "unk__14"]),
            helper.make_tensor_value_info("/ReduceSum_3_output_0", 1, ["unk__17"]),
        ],
        [
            helper.make_tensor_value_info("/If_output_0", 1, ["unk__23"]),
            helper.make_tensor_value_info("/TopK_output_0", 1, ["unk__12"]),
        ],
        initializer=[
            numpy_helper.from_array(
                np.array([1], dtype=helper.tensor_dtype_to_np_dtype(7)).reshape([1]), name="/TopK_K_shape"
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    import onnxruntime as ort

    path = sys.argv[1] if len(sys.argv) > 1 else "gh_issue_29071_if_constant_folding.onnx"
    onnx.save(build(), path)
    print("wrote", path)

    print("onnxruntime", ort.__version__)
    for avoid in (True, False):
        print("loading", path, "(disabled ConstantFolding)" if avoid else "(default options)")
        kw = {"disabled_optimizers": ["ConstantFolding"]} if avoid else {}
        _ = ort.InferenceSession(path, **kw)
        print("LOAD OK")
