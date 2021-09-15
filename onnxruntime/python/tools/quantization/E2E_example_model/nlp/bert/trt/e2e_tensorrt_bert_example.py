import os
import onnx
import glob
import scipy.io
import numpy as np
import logging
from PIL import Image
import json
import collections
import six
import unicodedata
import onnx
import onnxruntime
import data_processing as dp
import tokenization
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table, QuantType, QuantizationMode, QLinearOpsRegistry, optimize_model, QDQQuantizer

class BertDataReader(CalibrationDataReader):
    def __init__(self, model_path, squad_json, vocab_file, cache_file, batch_size, max_seq_length, num_inputs):
        self.model_path = model_path
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        self.num_inputs = num_inputs
        self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = 128
        self.max_query_length = 64
        self.enum_data_dicts = iter([])
        self.feature_list = []
        self.token_list = []
        self.example_id_list = []

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        input_ids = []
        input_mask = []
        segment_ids = []
        for i in range(self.batch_size):
            example = self.data[self.current_index + i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.feature_list.append(features)
            self.token_list.append(example.doc_tokens)
            if len(input_ids) and len(segment_ids) and len(input_mask):
                input_ids = np.vstack([input_ids, features[0].input_ids])
                input_mask = np.vstack([input_mask, features[0].input_mask])
                segment_ids = np.vstack([segment_ids, features[0].segment_ids])
            else:
                input_ids = np.expand_dims(features[0].input_ids, axis=0)
                input_mask = np.expand_dims(features[0].input_mask, axis=0)
                segment_ids = np.expand_dims(features[0].segment_ids, axis=0)
        data = [{"input_ids": input_ids, "input_mask": input_mask, "segment_ids":segment_ids}]
        print(input_ids.shape)
        self.current_index += self.batch_size
        self.enum_data_dicts = iter(data)
        return next(self.enum_data_dicts, None)

if __name__ == '__main__':
    '''
    QDQ Quantization of BERT model for TensorRT.

    There are two steps for the quantization,
    first, calibration is done based on SQuAD dataset to get dynamic range of each floating point tensor in the model
    second, Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

    The onnx model used in the script is converted from Hugging Face BERT model,
    https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package

    Some utility functions for dataset processing and data reader are from Nvidia TensorRT demo BERT repo,
    https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
    '''

    # Model and Dataset settings
    model_path = "./model.onnx"
    squad_json = "./dev-v1.1.json"
    vocab_file = "./vocab.txt"
    calibration_cache_file = "./bert.flatbuffers"
    augmented_model_path = "./augmented_model.onnx"
    qdq_model_path = "./qdq_model.onnx"
    sequence_lengths = [384]
    calib_num = 100

    # Generate INT8 calibration cache
    print("Calibration starts ...")
    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 
    data_reader = BertDataReader(model_path, squad_json, vocab_file, calibration_cache_file, 2, sequence_lengths[-1], calib_num)
    calibrator.collect_data(data_reader)
    compute_range = calibrator.compute_range()
    write_calibration_table(compute_range)
    print("Calibration is done. Calibration cache is saved to ", calibration_cache_file)

    # Generate QDQ model
    mode = QuantizationMode.QLinearOps
    op_types_to_quantize = ['MatMul', 'Transpose', 'Add']
    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())
    model = onnx.load_model(Path(model_path), optimize_model)
    quantizer = QDQQuantizer(
        model,
        False, #per_channel
        False, #reduce_range
        mode,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range,
        [], #nodes_to_quantize,
        [], #nodes_to_exclude,
        op_types_to_quantize,
        {'ActivationSymmetric' : True}) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(qdq_model_path, False)
    print("QDQ model is saved to ", qdq_model_path)
    
