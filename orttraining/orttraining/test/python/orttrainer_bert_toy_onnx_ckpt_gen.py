import copy
from functools import partial
import inspect
import math
import numpy as np
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


def load_bert_onnx_model():
    bert_onnx_model_path = os.path.join('testdata', "bert_toy_postprocessed.onnx")
    model = onnx.load(bert_onnx_model_path)
    return model



###############################################################################
# This method is used to generate checkpoints for ZeRO
# eg: mpirun -n 4 --tag-output python3 orttrainer_bert_toy_onnx_ckpt_gen.py
###############################################################################

def testToyBERTModelMixedPrecisionLossScaler():
    # Common setup
    from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size
    local_rank = max(0, get_mpi_context_local_rank())
    world_size = max(1, get_mpi_context_world_size())
    total_steps = int(np.ceil(float(10)/world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = 1
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': str(device),
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': None
        },
        'distributed': {
            'world_rank': local_rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'allreduce_post_accumulation': True,
            'deepspeed_zero_optimization': {'stage': 1}},
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, world_size * i+local_rank)
        losses.append(trainer.train_step(*sample_input).cpu().item())
    
    ckpt_dir = _test_helpers._get_name("ort_ckpt")
    checkpoint.experimental_save_checkpoint(trainer, ckpt_dir, 'bert_toy_lamb')

if __name__ == "__main__":
    testToyBERTModelMixedPrecisionLossScaler()