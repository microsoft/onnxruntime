import os
import sys

import numpy as np
import onnxruntime as ort
import json
import tokenization
from run_onnx_squad import *
import subprocess
from BaseModel import *

class GPT2(BaseModel):
    def __init__(self, model_name='GPT2', providers=None): 
        BaseModel.__init__(self, model_name, providers)
        self.input_file_ = 'inputs.json'
        self.all_results_ = []
        self.input_ids_ =  None
        self.input_mask_ = None 
        self.segment_ids_ = None 
        self.extra_data_ = None 
        self.eval_examples_ = None 
        # self.validate_decimal_ = 2 

        self.model_path_ = os.path.join(os.getcwd(), "GPT2", "model.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz", shell=True, check=True)     
            subprocess.run("tar zxf gpt2-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "GPT2") 

    def preprocess(self):
        return

    def inference(self, input_list=None):
        session = self.session_
        for input_meta in session.get_inputs():
            print(input_meta)
        for output_meta in session.get_outputs():
            print(output_meta)

        if input_list:

            outputs = []
            for test_data in input_list:
                input_ids = test_data[0] 

                n = len(input_ids)
                batch_size = 1
                bs = batch_size

                for idx in range(0, n):
                    data = {"input1": test_data[0]}

                    result = session.run(["output1","output2","output3", "output4", "output5", "output6", "output7", "output8", "output9", "output10", "output11", "output12", "output13"], data)
                outputs.append(result)
            self.outputs_ = outputs

    def postprocess(self):
        return

