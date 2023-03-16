import torch
import torch.onnx.symbolic_helper
import torch.onnx.utils


def _export_jit_graph_to_onnx_model_proto(graph: torch._C.Graph, operator_export_type: int):
    from torch.onnx.symbolic_helper import (  # noqa: F401
        _set_onnx_shape_inference,
        _set_operator_export_type,
        _set_opset_version,
    )

    _set_onnx_shape_inference(True)
    _set_operator_export_type(operator_export_type)
    torch._C._jit_pass_run_decompositions(graph)
    graph = torch.onnx.utils._optimize_graph(graph, operator_export_type, params_dict={})
    proto, _, _, _ = graph._export_onnx(
        {},
        torch.onnx._globals.GLOBALS.export_onnx_opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto
