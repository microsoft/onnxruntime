import os
import sys
import numpy as np
import onnxruntime
import subprocess
import onnx
from onnx import numpy_helper
from BaseModel import *

class Resnet50(BaseModel):
    def __init__(self, model_name='Resnet 50'): 
        BaseModel.__init__(self, model_name)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.outputs_ = []

        if not os.path.exists("resnet50v2"):
            subprocess.run("wget https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz", shell=True, check=True)
            subprocess.run("wget https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json", shell=True, check=True)
            subprocess.run("tar zxf resnet50v2.tar.gz", shell=True, check=True)

        self.load_input_and_output()
        self.session_ = onnxruntime.InferenceSession('resnet50v2/resnet50v2.onnx', None)


    def load_input_and_output(self):
        import glob

        test_data_dir = 'resnet50v2/test_data_set'
        test_data_num = 3

        # Load inputs
        inputs = self.inputs_ 
        for i in range(test_data_num):
            input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
                inputs.append(numpy_helper.to_array(tensor))

        print('Loaded {} inputs successfully.'.format(test_data_num))

        # Load reference outputs
        ref_outputs = self.ref_outputs_ 
        for i in range(test_data_num):
            output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
            tensor = onnx.TensorProto()
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
                ref_outputs.append(numpy_helper.to_array(tensor))

        print('Loaded {} reference outputs successfully.'.format(test_data_num))

    def inference(self):
        test_data_num = 3
        inputs = self.inputs_
        session = self.session_
        # get the name of the first input of the model
        input_name = session.get_inputs()[0].name
        self.outputs_ = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

        self.validate()

    def postprocess(self):
        return

    def validate(self):
        print('Predicted {} results.'.format(len(self.outputs_)))

        # Compare the results with reference outputs up to 4 decimal places
        for ref_o, o in zip(self.ref_outputs_, self.outputs_):
            np.testing.assert_almost_equal(ref_o, o, 4)

        print('ONNX Runtime outputs are similar to reference outputs!')
