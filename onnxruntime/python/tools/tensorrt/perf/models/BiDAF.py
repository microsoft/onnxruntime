import os
import subprocess
from BaseModel import *

class BiDAF(BaseModel):
    def __init__(self, model_name='BiDAF', providers=None):
        BaseModel.__init__(self, model_name, providers)

        self.model_path_ = os.path.join(os.getcwd(), "bidaf", "bidaf.onnx")

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf bidaf-9.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "bidaf")

    def get_ort_inputs(self, inputs):
        data = {"context_word": inputs[0],
                "context_char": inputs[2],
                "query_word": inputs[1],
                "query_char": inputs[3]}
        return data

    def get_ort_outputs(self):
        return ["start_pos", "end_pos"]

    def inference(self):
        self.outputs_ = []
        for test_data in self.inputs_:
            unique_ids_raw_output = test_data[0]
            input_ids = test_data[1]
            input_mask = test_data[2]
            segment_ids = test_data[3]

            n = len(input_ids)
            batch_size = 1
            bs = batch_size

            for idx in range(0, n):
                data = {"context_word": test_data[0],
                        "context_char": test_data[2],
                        "query_word": test_data[1],
                        "query_char": test_data[3]}

                result = self.session_.run(["start_pos","end_pos"], data)
            self.outputs_.append([result])

    # not use for perf
    def inference_old(self, input_list=None):
        session = self.session_
        for input_meta in session.get_inputs():
            print(input_meta)

        if input_list:
            outputs = []
            for test_data in input_list:
                unique_ids_raw_output = test_data[0]
                input_ids = test_data[1]
                input_mask = test_data[2]
                segment_ids = test_data[3]

                n = len(input_ids)
                batch_size = 1
                bs = batch_size

                for idx in range(0, n):
                    data = {"context_word": test_data[0],
                            "context_char": test_data[2],
                            "query_word": test_data[1],
                            "query_char": test_data[3]}

                    result = session.run(["start_pos","end_pos"], data)
                outputs.append([result])
            self.outputs_ = outputs

