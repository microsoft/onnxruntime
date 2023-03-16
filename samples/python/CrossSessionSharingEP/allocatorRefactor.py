import numpy

import onnxruntime

idx = onnxruntime.register_execution_provider(True, 'CPUExecutionProvider', {})
idx2 = onnxruntime.register_execution_provider(True, 'XnnpackExecutionProvider', {})
print(f'idx:{idx},idx2:{idx2}')

input('Cpu EP:')

session1 = onnxruntime.InferenceSession('c:/share/models/Detection/model.onnx', providers=['CPUExecutionProvider'])
input_name = session1.get_inputs()[0].name
output_name = session1.get_outputs()[0].name
input_values = numpy.random.rand(1,3,256,256).astype(numpy.float32)
out1 = session1.run([output_name], {input_name: input_values})
print('out1:'+str(out1))

#session2 = onnxruntime.InferenceSession('c:/share/models/Detection2/model.onnx', providers=['CPUExecutionProvider'], provider_options=[{'UseGlobal':str(idx)}])
session2 = onnxruntime.InferenceSession('c:/share/models/Detection2/model.onnx', providers=['CPUExecutionProvider'], global_providers=[idx])
out2 = session2.run([output_name], {input_name: input_values})
print('out2:'+str(out2))

input('Cpu and Xnnpack EP:')

session3 = onnxruntime.InferenceSession('c:/share/models/Detection/model.onnx', providers=['XnnpackExecutionProvider', 'CPUExecutionProvider'])
out3 = session3.run([output_name], {input_name: input_values})
print('out3:'+str(out3))

#session4 = onnxruntime.InferenceSession('c:/share/models/Detection2/model.onnx', providers=['XnnpackExecutionProvider', 'CPUExecutionProvider'], provider_options=[{'UseGlobal':str(idx2)},{}])
session4 = onnxruntime.InferenceSession('c:/share/models/Detection2/model.onnx', providers=['XnnpackExecutionProvider', 'CPUExecutionProvider'], global_providers=[idx2,-1])
out4 = session4.run([output_name], {input_name: input_values})
print('out4:'+str(out4))
