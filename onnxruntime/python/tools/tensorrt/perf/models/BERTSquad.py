import os
import sys
sys.path.append('./models/bert-squad/dependencies/')

import numpy as np
import onnxruntime as ort
import json
import tokenization
from run_onnx_squad import *
import subprocess
from BaseModel import *

class BERTSquad(BaseModel):
    def __init__(self, model_name='BERT-Squad', providers=None):
        BaseModel.__init__(self, model_name, providers)
        self.input_file_ = 'inputs.json'
        self.all_results_ = []
        self.input_ids_ =  None
        self.input_mask_ = None
        self.segment_ids_ = None
        self.extra_data_ = None
        self.eval_examples_ = None

        self.model_path_ = os.path.join(os.getcwd(), "download_sample_10", "bertsquad10.onnx")

        if not os.path.exists("uncased_L-12_H-768_A-12/bert_config.json"):
            subprocess.run("wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip", shell=True, check=True)
            subprocess.run("unzip uncased_L-12_H-768_A-12.zip", shell=True, check=True)

        if not os.path.exists(self.model_path_):
            subprocess.run("wget https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf bertsquad-10.tar.gz", shell=True, check=True)

        self.onnx_zoo_test_data_dir_ = os.path.join(os.getcwd(), "download_sample_10")

    def get_ort_inputs(self, inputs):
        unique_ids_raw_output = inputs[0]
        input_ids = inputs[1]
        input_mask = inputs[2]
        segment_ids = inputs[3]

        n = len(input_ids)
        batch_size = 1
        bs = batch_size

        data = {"unique_ids_raw_output___9:0": unique_ids_raw_output,
                "input_ids:0": input_ids[0:1],
                "input_mask:0": input_mask[0:1],
                "segment_ids:0": segment_ids[0:1]}

        return data

    def get_ort_outputs(self):
        return ["unique_ids:0", "unstack:0", "unstack:1"]

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
                data = {"unique_ids_raw_output___9:0": unique_ids_raw_output,
                        "input_ids:0": input_ids[idx:idx+bs],
                        "input_mask:0": input_mask[idx:idx+bs],
                        "segment_ids:0": segment_ids[idx:idx+bs]}

                result = self.session_.run(["unique_ids:0","unstack:0", "unstack:1"], data)
            self.outputs_.append([result])

    # not use for perf
    def preprocess(self):
        with open(self.input_file_) as json_file:
            test_data = json.load(json_file)
            print(json.dumps(test_data, indent=2))

        eval_examples = self.eval_examples_
        # Use read_squad_examples method from run_onnx_squad to read the input file
        eval_examples = read_squad_examples(input_file=self.input_file_)

        max_seq_length = 256
        doc_stride = 128
        max_query_length = 64

        vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        my_list = []

        # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
        input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
                                                                              max_seq_length, doc_stride, max_query_length)

        self.input_ids_ = input_ids
        self.input_mask_ = input_mask
        self.segment_ids_ = segment_ids
        self.extra_data_ = extra_data
        self.eval_examples_ = eval_examples

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
                    data = {"unique_ids_raw_output___9:0": unique_ids_raw_output,
                            "input_ids:0": input_ids[idx:idx+bs],
                            "input_mask:0": input_mask[idx:idx+bs],
                            "segment_ids:0": segment_ids[idx:idx+bs]}

                    result = session.run(["unique_ids:0","unstack:0", "unstack:1"], data)
                outputs.append([result])
            self.outputs_ = outputs

        else:
            input_ids = self.input_ids_
            input_mask = self.input_mask_
            segment_ids = self.segment_ids_

            extra_data = self.extra_data_
            eval_examples = self.eval_examples_
            all_results = self.all_results_
            batch_size = 1

            n = len(input_ids)
            bs = batch_size

            for idx in range(0, n):
                item = eval_examples[idx]
                data = {"unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64),
                        "input_ids:0": input_ids[idx:idx+bs],
                        "input_mask:0": input_mask[idx:idx+bs],
                        "segment_ids:0": segment_ids[idx:idx+bs]}

                result = session.run(["unique_ids:0","unstack:0", "unstack:1"], data)
                in_batch = result[1].shape[0]
                start_logits = [float(x) for x in result[1][0].flat]
                end_logits = [float(x) for x in result[2][0].flat]
                for i in range(0, in_batch):
                    unique_id = len(all_results)
                    all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
            self.outputs_ = [result]

    # not use for perf
    def postprocess(self):
        n_best_size = 20
        max_answer_length = 30
        all_results = self.all_results_
        eval_examples = self.eval_examples_
        extra_data = self.extra_data_

        # postprocessing
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        output_prediction_file = os.path.join(output_dir, "predictions.json")
        output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
        write_predictions(eval_examples, extra_data, all_results,
                          n_best_size, max_answer_length,
                          True, output_prediction_file, output_nbest_file)


        # print results
        with open(output_prediction_file) as json_file:
            test_data = json.load(json_file)
            print(json.dumps(test_data, indent=2))

