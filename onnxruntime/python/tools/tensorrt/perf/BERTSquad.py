import os
import sys
sys.path.append('./bert-squad/dependencies/')

import numpy as np
import onnxruntime as ort
import json
import tokenization
from run_onnx_squad import *
import subprocess
from BaseModel import *

class BERTSquad(BaseModel):
    def __init__(self, model_name='BERT Squad'): 
        BaseModel.__init__(self, model_name)
        self.input_file_ = 'inputs.json'
        self.all_results_ = []
        self.input_ids_ =  None
        self.input_mask_ = None 
        self.segment_ids_ = None 
        self.extra_data_ = None 
        self.eval_examples_ = None 

        if not os.path.exists("uncased_L-12_H-768_A-12"):
            subprocess.run("wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip", shell=True, check=True)     
            subprocess.run("unzip uncased_L-12_H-768_A-12.zip", shell=True, check=True)

        if not os.path.exists("download_sample_10"):
            subprocess.run("wget https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz", shell=True, check=True)     
            subprocess.run("tar zxf bertsquad-10.tar.gz", shell=True, check=True)

        self.preprocess()

        try: 
            self.session_ = ort.InferenceSession('./download_sample_10/bertsquad10.onnx')
        except:
            subprocess.run("python3 ../symbolic_shape_infer.py --input ./download_sample_10/bertsquad10.onnx --output ./download_sample_10/bertsquad10_new.onnx --auto_merge", shell=True, check=True)     
            self.session_ = ort.InferenceSession('./download_sample_10/bertsquad10_new.onnx')

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

    def inference(self):
        input_ids = self.input_ids_
        input_mask = self.input_mask_
        segment_ids = self.segment_ids_
        extra_data = self.extra_data_
        eval_examples = self.eval_examples_
        all_results = self.all_results_
        batch_size = 1

        session = self.session_

        for input_meta in session.get_inputs():
            print(input_meta)

        n = len(input_ids)
        bs = batch_size
        start = timer()

        for idx in range(0, n):
            item = eval_examples[idx]
            # this is using batch_size=1
            # feed the input data as int64
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

