from onnx_model import OnnxModel
import onnx
from fusion_utils import NumpyHelper

model_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small.onnx"
# model = OnnxModel(onnx.load(model_path))
# for node in model.nodes():
#     if "model_modeling_mixformer_sequential_ParallelBlock_sub1" in node.name:
#         for input in node.input:
#             print(input)
#             tensor = model.get_initializer(input)
#             if tensor is None:
#                 print("None")
#             else:
#                 np_array = NumpyHelper.to_array(tensor)
#                 print(np_array.shape)
#                 print(np_array)
#     print(node.name)

# from onnx import inliner

# model_proto = onnx.load(model_path)
# inlined_model = inliner.inline_local_functions(model_proto)
# onnx.save(inlined_model, "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small_inlined.onnx")

# Authors: Aaron Bockover, Ganesan Ramalingam, Kunal Vaishnavi
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
from onnx import inliner
from typing import Callable, Sequence


def inline_into_function(
    main: onnx.FunctionProto,
    inlined_functions: Sequence[onnx.FunctionProto],
    opset_imports,
):
    dummy_type = onnx.helper.make_tensor_type_proto(1, ["N"])

    def typed(names):
        return [onnx.helper.make_value_info(name, dummy_type) for name in names]

    graph = onnx.helper.make_graph(main.node, main.name, typed(main.input), typed(main.output))
    model = onnx.helper.make_model(graph)

    model.functions.extend(inlined_functions)
    model.opset_import.extend(opset_imports)

    model = onnx.inliner.inline_local_functions(model, False)
    del main.node[:]
    main.node.extend(model.graph.node)

    return main


def inline_local_functions(
    model: onnx.ModelProto,
    should_inline_function: Callable[[onnx.FunctionProto], bool],
) -> onnx.ModelProto:

    skip_inlining: list[onnx.FunctionProto] = []

    if should_inline_function is not None:
        skip_inlining = [fn for fn in model.functions if not should_inline_function(fn)]

    for func in skip_inlining:
        model.functions.remove(func)

    inlined_model = onnx.inliner.inline_local_functions(model, False)

    skipped = [inline_into_function(f, model.functions, model.opset_import) for f in skip_inlining]
    inlined_model.functions.extend(skipped)

    return inlined_model


def should_inline_function(
    function: onnx.FunctionProto,
) -> bool:

    functions_to_inline = {
        "model_modeling_mixformer_sequential_Embedding_layers_0_1",
        "model_modeling_mixformer_sequential_CausalLMHead_layers__1_1",
    }
    return function.name in functions_to_inline


if __name__ == "__main__":
    print(onnx.__version__)
    model = onnx.load_model(model_path, load_external_data=False)
    inlined_model = inline_local_functions(model, should_inline_function)
    onnx.checker.check_model(inlined_model)
    onnx.shape_inference.infer_shapes(inlined_model)
    onnx.save_model(
        inlined_model,
        "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small_inlined.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    )
