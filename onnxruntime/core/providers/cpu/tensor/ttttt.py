import os
import sys

import numpy as np

sys.path.insert(0,'/home/stcadmin/work/onnxruntime/build/Linux/Debug/build/lib')
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static


class WavDataReader(CalibrationDataReader):
    def __init__(self):
        self.preprocess_flag = True
        self.enum_data_dicts = []

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            feature = np.random.rand(1, 500, 80).astype('float32')
            feature_len = np.array([500])
            self.enum_data_dicts = iter([{'x': feature, 'x_lens': feature_len}])

        return next(self.enum_data_dicts, None)


input_model = "/home/stcadmin/work/onnxruntime/encoder_en.onnx"
wav_data_reader = WavDataReader()

print(ort.__version__)

output_qt_model = f"/home/stcadmin/work/onnxruntime/exported_models/encoder_en_static_qt_rand.onnx"
#quantize_static(input_model,
#                output_qt_model,
#                wav_data_reader,
#                quant_format=QuantFormat.QOperator, # QOperator QDQ
#                optimize_model=True,
#                activation_type=QuantType.QUInt8,
#                weight_type=QuantType.QUInt8,
#                calibrate_method=CalibrationMethod.MinMax,
#                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
#                nodes_to_exclude=["softmax"]
#                )


print('ONNX full precision model size (MB):', os.path.getsize(input_model)/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize(output_qt_model)/(1024*1024))

feature = np.random.rand(1, 500, 80).astype('float32')
feature_len = np.array([500])
enum_data_dicts = {'x': feature, 'x_lens': feature_len}
sess_q = ort.InferenceSession(output_qt_model,providers=['CPUExecutionProvider'])
sess_f = ort.InferenceSession(input_model,providers=['CPUExecutionProvider'])
q_res = sess_q.run([], input_feed=enum_data_dicts)
f_res = sess_f.run([], input_feed=enum_data_dicts)
q_f_d = q_res[0]-f_res[0]
