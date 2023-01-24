import numpy as np
import onnx
from mpi4py import MPI
from onnx import AttributeProto, GraphProto, TensorProto, helper

import onnxruntime as ort

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def build_allreduce_model():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [128, 128])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [128, 128])
    node_def = helper.make_node("AllReduce", ["X"], ["Y"], domain="com.microsoft")
    graph_def = helper.make_graph(
        [node_def],
        "",
        [X],
        [Y],
    )
    return helper.make_model(graph_def, producer_name="ort-distributed-inference-unittest")


print(f"!!!!!Running for rank {rank}")
model = build_allreduce_model()
ort_sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"device_id": str(rank)}, {}],
)
input = np.ones((128, 128), dtype=np.float32)
outputs = ort_sess.run(None, {"X": input})
assert np.allclose(outputs[0], size * input)
