
# generate sample input for our example
import inspect
import onnx
import os
import pytest
import torch

from numpy.testing import assert_allclose

from onnxruntime.capi._pybind_state import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.capi.training import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options


###############################################################################
# Helper functions ############################################################
###############################################################################


def generate_random_input_from_model_desc(desc):
    dtype = torch.int64
    vocab_size = 30528
    num_classes = [vocab_size, 2, 2, vocab_size, 2]
    device = "cuda:0"
    sample_input = []
    for index, input in enumerate(desc['inputs']):
        sample_input.append(torch.randint(0, num_classes[index], tuple(input[1]), dtype=dtype).to(device))
    return sample_input

def bert_model_description():
    vocab_size = 30528
    batch_size = 16
    seq_len = 1
    model_desc = {'inputs': [('input_ids', [batch_size, seq_len]),
                             ('segment_ids', [batch_size, seq_len],),
                             ('input_mask', [batch_size, seq_len],),
                             ('masked_lm_labels', [batch_size, seq_len],),
                             ('next_sentence_labels', [batch_size, ],)],
                  'outputs': [('loss', [], True)]}
    return model_desc


###############################################################################
# Testing starts here #########################################################
###############################################################################


def testToyBERTModel():
    #print(bert_model_description())
    model_desc = bert_model_description()
    device = torch.device("cuda", 0)

    pytorch_transformer_path = os.path.join('..', '..', '..', 'onnxruntime', 'test', 'testdata')
    bert_onnx_model_path = os.path.join(pytorch_transformer_path, "bert_toy_postprocessed.onnx")
    model = onnx.load(bert_onnx_model_path)

    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device' : {
            'id' : "cuda:0",
        }

    })
    
    torch.manual_seed(1)
    set_seed(1)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    sample_input = generate_random_input_from_model_desc(model_desc)

    output = trainer.train_step(*sample_input)
    #print(output)
    assert output.shape == torch.Size([]) 

