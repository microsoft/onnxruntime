import copy
from functools import partial
import inspect
import math
from numpy.testing import assert_allclose
import onnx
import os
import pytest
import torch

import onnxruntime
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_commons, _test_helpers


###############################################################################
# Helper functions ############################################################
###############################################################################


def generate_random_input_from_model_desc(desc, seed=1, device = "cuda:0"):
    '''Generates a sample input for the BERT model using the model desc'''

    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    dtype = torch.int64
    vocab_size = 30528
    num_classes = [vocab_size, 2, 2, vocab_size, 2]
    dims = {"batch_size":16, "seq_len":1}
    sample_input = []
    for index, input in enumerate(desc['inputs']):
        size = []
        for s in input[1]:
            if isinstance(s, (int)):
                size.append(s)
            else:
                size.append(dims[s] if s in dims else 1)
        sample_input.append(torch.randint(0, num_classes[index], tuple(size), dtype=dtype).to(device))
    return sample_input

# EXPERIMENTAL HELPER FUNCTIONS

def bert_model_description(dynamic_shape=True):
    '''Creates the model description dictionary with static dimensions'''

    if dynamic_shape:
        model_desc = {'inputs': [('input_ids', ['batch_size', 'seq_len']),
                                 ('segment_ids', ['batch_size', 'seq_len'],),
                                 ('input_mask', ['batch_size', 'seq_len'],),
                                 ('masked_lm_labels', ['batch_size', 'seq_len'],),
                                 ('next_sentence_labels', ['batch_size', ],)],
                                 'outputs': [('loss', [], True)]}
    else:
        batch_size = 16
        seq_len = 1
        model_desc = {'inputs': [('input_ids', [batch_size, seq_len]),
                                ('segment_ids', [batch_size, seq_len],),
                                ('input_mask', [batch_size, seq_len],),
                                ('masked_lm_labels', [batch_size, seq_len],),
                                ('next_sentence_labels', [batch_size, ],)],
                    'outputs': [('loss', [], True)]}
    return model_desc


def optimizer_parameters(model):
    '''A method to assign different hyper parameters for different model parameter groups'''

    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay_param_group = []
    for initializer in model.graph.initializer:
        if any(key in initializer.name for key in no_decay_keys):
            no_decay_param_group.append(initializer.name)
    params = [{'params': no_decay_param_group, "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6, "do_bias_correction":False}]

    return params


def load_bert_onnx_model():
    bert_onnx_model_path = os.path.join('testdata', "bert_toy_postprocessed.onnx")
    model = onnx.load(bert_onnx_model_path)
    return model


class CustomLossScaler(amp.LossScaler):
    def __init__(self, loss_scale=float(1 << 16)):
        super().__init__(loss_scale)
        self._initial_loss_scale = loss_scale
        self.loss_scale = loss_scale

    def reset(self):
        self.loss_scale = self._initial_loss_scale

    def update(self, train_step_info):
        self.loss_scale *= 0.9
        return self.loss_scale

# LEGACY HELPER FUNCTIONS

class LegacyCustomLossScaler():
    def __init__(self, loss_scale=float(1 << 16)):
        self._initial_loss_scale = loss_scale
        self.loss_scale_ = loss_scale

    def reset(self):
        self.loss_scale_ = self._initial_loss_scale

    def update_loss_scale(self, is_all_finite):
        self.loss_scale_ *= 0.9


def legacy_model_params(lr, device = torch.device("cuda", 0)):
    legacy_model_desc = legacy_bert_model_description()
    learning_rate_description = legacy_ort_trainer_learning_rate_description()
    learning_rate = torch.tensor([lr]).to(device)
    return (legacy_model_desc, learning_rate_description, learning_rate)

def legacy_ort_trainer_learning_rate_description():
    return Legacy_IODescription('Learning_Rate', [1, ], torch.float32)


def legacy_bert_model_description():
    vocab_size = 30528
    input_ids_desc = Legacy_IODescription('input_ids', ['batch', 'max_seq_len_in_batch'])
    segment_ids_desc = Legacy_IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'])
    input_mask_desc = Legacy_IODescription('input_mask', ['batch', 'max_seq_len_in_batch'])
    masked_lm_labels_desc = Legacy_IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'])
    next_sentence_labels_desc = Legacy_IODescription('next_sentence_labels', ['batch', ])
    loss_desc = Legacy_IODescription('loss', [])

    return Legacy_ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc,
                             next_sentence_labels_desc], [loss_desc])


def legacy_optim_params_a(name):
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6, "do_bias_correction": False}


def legacy_optim_params_b(name):
    params = ['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight']
    if name in params:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6, "do_bias_correction": False}
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6, "do_bias_correction": False}


def legacy_optim_params_c(name):
    params_group = optimizer_parameters(load_bert_onnx_model())
    if name in params_group[0]['params']:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6, "do_bias_correction": False}
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6, "do_bias_correction": False}
