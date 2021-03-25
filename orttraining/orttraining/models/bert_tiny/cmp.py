import onnx
from onnx import numpy_helper
import os
import numpy as np

cpu_dump_dir = os.path.join(os.getcwd(), "cpu_dump")
cuda_dump_dir = os.path.join(os.getcwd(), "cuda_dump")

name_diff = {}
for entry in os.scandir(cuda_dump_dir):
    if not "tensorproto1" in entry.name:
        continue
    cuda_pb = onnx.load_tensor(os.path.join(cuda_dump_dir, entry.name))
    filename = entry.name.replace("_CUDAExecutionProvider", "")
    cpu_filename = os.path.join(cpu_dump_dir, filename)
    if not os.path.exists(cpu_filename):
        continue
    cpu_pb = onnx.load_tensor(cpu_filename)
    cuda_arr = numpy_helper.to_array(cuda_pb)
    cpu_arr = numpy_helper.to_array(cpu_pb)
    if cpu_arr.dtype != np.bool:
        abs_diff = np.abs(cuda_arr - cpu_arr)
        avg_abs_diff = np.average(abs_diff)
        max_abs_diff = np.max(abs_diff)
        name_diff[filename] = (avg_abs_diff, max_abs_diff)

names = list(name_diff.keys())
names.sort()
max_values = {"max_avg":0, "max_avg_name":"", "max_max":0, "max_max_name":""}
for key in names:
    print(key + ": " + str(name_diff[key]))
    t_avg, t_max = name_diff[key]
    if max_values["max_avg"] < t_avg:
        max_values["max_avg"] = t_avg
        max_values["max_avg_name"] = key
    if max_values["max_max"] < t_max:
        max_values["max_max"] = t_max
        max_values["max_max_name"] = key
print("max avg_abs_diff: " + str(max_values["max_avg"]) + " name:" + max_values["max_avg_name"])
print("max max_abs_diff: " + str(max_values["max_max"]) + " name:" + max_values["max_max_name"])