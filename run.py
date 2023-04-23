import os, sys

# don't load from . relative from __file__
to_remove = []
d_to_remove = os.path.dirname(os.path.abspath(__file__))
for i, p in enumerate(sys.path):
    try:
        if p == "":
            to_remove.append(i)
        elif os.path.samefile(p, d_to_remove):
            to_remove.append(i)
    except:
        pass

for i in reversed(to_remove):
    try:
        sys.path.pop(i)
    except:
        pass

import onnxruntime as ort
from pprint import pprint
import tempfile
import torch

ort.set_default_logger_severity(0)
# ort.set_default_logger_verbosity(1000)

so = ort.SessionOptions()
ort.log_severity_level = 0
# ort.log_verbosity_level = 1000


class ToOnnxModel(torch.nn.Module):
    def forward(self, x):
        return torch.matmul(x[0], x[1])
    def dummy_inputs(self):
        return [torch.zeros([4096, 768]), torch.zeros([768, 1024])]

m = ToOnnxModel()

with tempfile.TemporaryDirectory() as tmp_dir:
    path = os.path.join(tmp_dir, "tmp.onnx")
    torch.onnx.export(m, m.dummy_inputs(), path)
    model_bytes = open(path, "rb").read()

sess = ort.InferenceSession(
    model_bytes, providers=[
    ("ROCMExecutionProvider", {"tunable_op_enabled": "1"})
])

sess.run(
    output_names = [node.name for node in sess.get_outputs()],
    input_feed = { sess.get_inputs()[i].name: d.numpy()  for i, d in enumerate(m.dummy_inputs())}
)

pprint(sess.get_provider_options())

all_options = sess.get_provider_options()
rocm_options = all_options["ROCMExecutionProvider"]
rocm_options["tunable_op_enabled"] = "0"
sess.set_providers(["ROCMExecutionProvider"], [rocm_options])
pprint(sess.get_provider_options())
