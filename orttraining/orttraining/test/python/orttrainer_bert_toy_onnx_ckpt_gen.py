import os
import torch
import pytest

import onnxruntime
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_commons
import _test_helpers

from orttraining_test_orttrainer_bert_toy_onnx import bert_model_description, \
                                                      load_bert_onnx_model


###############################################################################
# This method is used to generate bert checkpoints for ZeRO
# eg: mpirun -n 4 python3 orttrainer_bert_toy_onnx_ckpt_gen.py
###############################################################################
# @pytest.mark.parametrize("optimizer, mixedprecision_enabled", [
#     (optim.LambConfig(), False),
#     (optim.AdamConfig(), False),
#     (optim.LambConfig(), True),
#     (optim.AdamConfig(), True),
# ])
def testToyBertLoadOptimStateZero(optimizer, mixedprecision_enabled):
    # Common setup
    local_rank = max(0, int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']))
    world_size = max(1, int(os.environ["OMPI_COMM_WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optimizer
    opts = orttrainer.ORTTrainerOptions({
        'debug': {
            'deterministic_compute': True
        },
        'device': {
            'id': str(device),
        },
        'mixed_precision': {
            'enabled': mixedprecision_enabled,
        },
        'distributed': {
            'world_rank': local_rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'allreduce_post_accumulation': True,
            'deepspeed_zero_optimization': {'stage': 1}},
    })
    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    dummy_init_state = _test_commons.generate_dummy_optim_state(model, optimizer)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    checkpoint._experimental_load_optimizer_state(trainer, dummy_init_state)

    # Input values
    input_ids = torch.tensor([[26598],[21379],[19922],[ 5219],[ 5644],[20559],[23777],[25672],[22969],[16824],[16822],[635],[27399],[20647],[18519],[15546]], device=device)
    segment_ids = torch.tensor([[0],[1],[0],[1],[0],[0],[1],[0],[0],[1],[1],[0],[0],[1],[1],[1]], device=device)
    input_mask = torch.tensor([[0],[0],[0],[0],[1],[1],[1],[0],[1],[1],[0],[0],[0],[1],[0],[0]], device=device)
    masked_lm_labels = torch.tensor([[25496],[16184],[11005],[16228],[14884],[21660],[ 8678],[23083],[ 4027],[ 8397],[11921],[ 1333],[26482],[ 1666],[17925],[27978]], device=device)
    next_sentence_labels = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], device=device)

    # Actual values
    _ = trainer.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)

    actual_state = checkpoint.experimental_state_dict(trainer)
    actual_state = checkpoint._split_state_dict(actual_state)
    fp16_param = actual_state['fp16_param']
    fp32_param = actual_state['fp32_param']

    split_info = dict()
    if (mixedprecision_enabled):
        # get split info
        for weight_name, v in fp32_param.items():
            if '_view_' in weight_name:
                clean_name = weight_name.split('_view_')[0]
                split_info[clean_name] = v.size()

    actual_optim_state = _test_commons.get_optim_state_from_state_dict(actual_state['optimizer'], optimizer)
    _test_helpers.assert_optim_state(dummy_init_state, actual_optim_state, split_info)


testToyBertLoadOptimStateZero(optim.LambConfig(), True)
testToyBertLoadOptimStateZero(optim.AdamConfig(), True)
