import copy
from functools import partial
import inspect
import math
from numpy.testing import assert_allclose
import onnx
import os
import pytest
import torch
import subprocess

import onnxruntime
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_commons, _test_helpers

from orttrainer_bert_toy_onnx import generate_random_input_from_model_desc,\
                                     bert_model_description, load_bert_onnx_model


# Test ZeRO checkpoint loading from a distributed Zero stage 1 mixedprecision run to the
# following configs. This test ensures that the weights as well as the optimizer
# state are aggregated and loaded correctly into fp32 and fp16 single-node
# trainers respectively, for Adam and Lamb optimizers.
@pytest.mark.parametrize("optimizer, mixedprecision_enabled, expected_eval_loss", [
    (optim.LambConfig(), False, [11.011026]),
    (optim.AdamConfig(), False, [10.998348]),
    (optim.LambConfig(), True, [11.011026]),
    (optim.AdamConfig(), True, [10.998348]),
])
def testToyBertCheckpointLoadZero(optimizer, mixedprecision_enabled, expected_eval_loss):
    # Common setup
    rtol = 1e-03
    device = 'cuda'
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optimizer
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True},
                                         'device' : {'id' : device},
                                         'mixed_precision': {
                                                'enabled': mixedprecision_enabled,
                                            },
                                         'distributed' : {'allreduce_post_accumulation' : True}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    ckpt_dir = _test_helpers._get_name("ort_ckpt")
    ckpt_prefix = _test_helpers._get_bert_ckpt_prefix(optimizer.name)
    try:
        checkpoint_files = sorted(checkpoint._list_checkpoint_files(ckpt_dir, ckpt_prefix))
    except(AssertionError):
        print("No checkpoint files found. Attempting to generate...")
        assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python',
                                'orttrainer_bert_toy_onnx_ckpt_gen.py']) == 0
        checkpoint_files = sorted(checkpoint._list_checkpoint_files(ckpt_dir, ckpt_prefix))

    ########################################
    # Test the aggregation code individually
    ########################################
    ckpt_agg = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    expected_state_name = 'aggregated_' + ckpt_prefix + '.pt'
    expected_state_dict = torch.load(os.path.join(ckpt_dir, expected_state_name),
                                     map_location=torch.device("cpu"))

    assert expected_state_dict.keys() == aggregate_state_dict.keys()

    for k,v in aggregate_state_dict.items():
        assert_allclose(v, expected_state_dict[k], rtol=1e-3, atol=1e-4)

    #############################################
    # Test the load_state_dict functionality that
    # does the aggregation uner the hood
    #############################################
    checkpoint.experimental_load_checkpoint(trainer, ckpt_dir, ckpt_prefix)

    # input values
    input_ids = torch.tensor([[26598],[21379],[19922],[ 5219],[ 5644],[20559],[23777],[25672],[22969],[16824],[16822],[635],[27399],[20647],[18519],[15546]], device=device)
    segment_ids = torch.tensor([[0],[1],[0],[1],[0],[0],[1],[0],[0],[1],[1],[0],[0],[1],[1],[1]], device=device)
    input_mask = torch.tensor([[0],[0],[0],[0],[1],[1],[1],[0],[1],[1],[0],[0],[0],[1],[0],[0]], device=device)
    masked_lm_labels = torch.tensor([[25496],[16184],[11005],[16228],[14884],[21660],[ 8678],[23083],[ 4027],[ 8397],[11921],[ 1333],[26482],[ 1666],[17925],[27978]], device=device)
    next_sentence_labels = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], device=device)

    # Actual values
    actual_eval_loss = trainer.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    actual_eval_loss = actual_eval_loss.cpu().numpy().item(0)

    checkpoint_files = sorted(checkpoint._list_checkpoint_files(ckpt_dir, ckpt_prefix))
    loaded_state_dict = checkpoint.experimental_state_dict(trainer)

    # check if the loaded state dict is as expected
    if mixedprecision_enabled:
        assert expected_state_dict.keys() == loaded_state_dict.keys()
    for k,v in loaded_state_dict.items():
        assert_allclose(v, expected_state_dict[k], rtol=1e-3, atol=1e-4)

    # compare loaded state dict to rank state dicts
    loaded_state_dict = checkpoint._split_state_dict(loaded_state_dict)

    for f in checkpoint_files:
        rank_state_dict = torch.load(f, map_location=torch.device("cpu"))
        rank_state_dict = checkpoint._split_state_dict(rank_state_dict['model'])

        for k,v in rank_state_dict['fp16_param'].items():
            fp32_name = k.split('_fp16')[0]
            assert_allclose(v, loaded_state_dict['fp32_param'][fp32_name], rtol=1e-02, atol=1e-02)
            if mixedprecision_enabled:
                assert_allclose(v, loaded_state_dict['fp16_param'][k])
        
        for k,v in rank_state_dict['optimizer'].items():
            if k in loaded_state_dict['optimizer']:
                assert_allclose(v, loaded_state_dict['optimizer'][k])
            else:
                assert '_view_' in k 

    # Check results
    assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol)
