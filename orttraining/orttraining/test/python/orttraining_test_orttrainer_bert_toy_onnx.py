
# generate sample input for our example
import inspect
import onnx
import os
import pytest
import torch

from numpy.testing import assert_allclose

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.experimental import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options


###############################################################################
# Helper functions ############################################################
###############################################################################


def generate_random_input_from_model_desc(desc, device='cuda'):
    num_classes = [30528, 2, 2, 30528, 2]
    sample_input = []
    for index, input in enumerate(desc['inputs']):
        size = [s if isinstance(s, (int)) else 1 for s in input[1]]
        sample_input.append(torch.randint(0,
                                          num_classes[index],
                                          tuple(size),
                                          dtype=torch.int64).to(device))
    return sample_input


def bert_model_description():
    model_desc = {'inputs': [('input_ids', ['batch_size', 'seq_len']),
                             ('segment_ids', ['batch_size', 'seq_len'],),
                             ('input_mask', ['batch_size', 'seq_len'],),
                             ('masked_lm_labels', ['batch_size', 'seq_len'],),
                             ('next_sentence_labels', ['batch_size', ],)],
                  'outputs': [('loss', [], True)]}
    return model_desc


###############################################################################
# Testing starts here #########################################################
###############################################################################


def testORTTrainerToyBERTModel():
    # Common setup
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)

    # Modeling
    pytorch_transformer_path = os.path.join('..', '..', '..', 'onnxruntime', 'test', 'testdata')
    bert_onnx_model_path = os.path.join(pytorch_transformer_path, "bert_toy_postprocessed.onnx")
    model = onnx.load(bert_onnx_model_path)
    model_desc = bert_model_description()
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True}})
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Generating fake input
    sample_input = generate_random_input_from_model_desc(model_desc)

    # Train
    output = trainer.train_step(*sample_input)

    # Check output
    assert output.shape == torch.Size([])
