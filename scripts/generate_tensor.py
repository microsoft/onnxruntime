import onnx
import numpy as np
from onnx import numpy_helper

def write_tensor(f,tensor,input_name=None):
    if input_name:
       tensor.name = input_name
    body = tensor.SerializeToString()
    f.write(body)

x = np.random.randn(1,224,288,1)
with  open("input_0.pb", "wb") as f:
      t = numpy_helper.from_array(x.astype(np.float32))
      write_tensor(f,t,'Placeholder:0')