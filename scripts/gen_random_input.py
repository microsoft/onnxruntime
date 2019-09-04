#!/usr/bin/python3

import numpy as np
from onnx import numpy_helper

t = np.random.rand(1,3,60,128).astype(np.float32)
im = numpy_helper.from_array(t)
with  open("input_0.pb", "wb") as f:
    f.write(im.SerializeToString())