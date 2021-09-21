from onnxruntime.capi._pybind_state import GradientGraphBuilder

from ..ortmodule._custom_op_symbolic_registry import CustomOpSymbolicRegistry
import torch

def export_gradient_graph(model: torch.nn.Module):
    # Make sure that loss nodes that expect multiple outputs are set up.        
    CustomOpSymbolicRegistry.register_all()
    
