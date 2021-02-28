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

###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize("dynamic_shape", [
    (True),
    (False)
])
def testToyBERTModelBasicTraining(dynamic_shape):
    model_desc = bert_model_description(dynamic_shape)
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({})
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(10):
        sample_input = generate_random_input_from_model_desc(model_desc)
        output = trainer.train_step(*sample_input)
        assert output.shape == torch.Size([])


@pytest.mark.parametrize("expected_losses", [
    ([10.991958, 10.975625, 11.032847, 11.034771, 10.987653,
      11.039469, 10.971498, 11.101391, 11.047601, 11.077588])
])
def testToyBERTDeterministicCheck(expected_losses):
    # Common setup
    train_steps = 10
    device = 'cuda'
    seed = 1
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    params = optimizer_parameters(model)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(experimental_losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("initial_lr, lr_scheduler, expected_learning_rates, expected_losses", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler,\
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [10.988012313842773, 10.99213981628418, 120.79301452636719, 36.11647033691406, 95.83200073242188,\
         221.2766571044922, 208.40316772460938, 279.5332946777344, 402.46380615234375, 325.79254150390625]),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler,\
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [10.988012313842773, 10.99213981628418, 52.69743347167969, 19.741533279418945, 83.88340759277344,\
         126.39848327636719, 91.53898620605469, 63.62016296386719, 102.21206665039062, 180.1424560546875]),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler,\
        [0.0, 0.9931806517013612, 0.9397368756032445, 0.8386407858128706, 0.7008477123264848, 0.5412896727361662,\
         0.37725725642960045, 0.22652592093878665, 0.10542974530180327, 0.02709137914968268],
        [10.988012313842773, 10.99213981628418, 120.6441650390625, 32.152557373046875, 89.63705444335938,\
         138.8782196044922, 117.57748413085938, 148.01927185058594, 229.60403442382812, 110.2930908203125]),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler,\
        [0.0, 0.9473684210526315, 0.8421052631578947, 0.7368421052631579, 0.631578947368421, 0.5263157894736842,\
         0.42105263157894735, 0.3157894736842105, 0.21052631578947367, 0.10526315789473684],
        [10.988012313842773, 10.99213981628418, 112.89633178710938, 31.114538192749023, 80.94029235839844,\
         131.34490966796875, 111.4329605102539, 133.74252319335938, 219.37344360351562, 109.67041015625]),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler,\
        [0.0, 0.9473684263157895, 0.8421052789473684, 0.7368421315789474, 0.6315789842105263, 0.5263158368421054,
         0.42105268947368424, 0.31578954210526317, 0.21052639473684212, 0.10526324736842106],
        [10.988012313842773, 10.99213981628418, 112.89633178710938, 31.114538192749023, 80.9402847290039,\
         131.3447265625, 111.43253326416016, 133.7415008544922, 219.37147521972656, 109.66986083984375])
])
def testToyBERTModelLRScheduler(initial_lr, lr_scheduler, expected_learning_rates, expected_losses):
    return # TODO: re-enable after nondeterminism on backend is fixed
    # Common setup
    device = 'cuda'
    total_steps = 10
    seed = 1
    warmup = 0.05
    cycles = 0.5
    power = 1.
    lr_end = 1e-7
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Setup LR Schedulers
    if lr_scheduler == optim.lr_scheduler.ConstantWarmupLRScheduler or lr_scheduler == optim.lr_scheduler.LinearWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup)
    elif lr_scheduler == optim.lr_scheduler.CosineWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif lr_scheduler == optim.lr_scheduler.PolyWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid lr_scheduler")

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    learning_rates = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())
        learning_rates.append(trainer.options.lr_scheduler.get_last_lr()[0])

    # Check output
    _test_helpers.assert_model_outputs(learning_rates, expected_learning_rates, rtol=rtol)
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("loss_scaler, expected_losses", [
    (None, [10.992018, 10.975699, 11.032809, 11.034765, 10.987625,
            11.039452, 10.971539, 11.10148, 11.047551, 11.077468]),
    (amp.DynamicLossScaler(), [10.992018, 10.975699, 11.032809, 11.034765,
                               10.987625, 11.039452, 10.971539, 11.10148, 11.047551, 11.077468]),
    (CustomLossScaler(), [10.992018, 10.975699, 11.032791, 11.034729,
                          10.987614, 11.039479, 10.971532, 11.101475, 11.04761, 11.077413])
])
def testToyBERTModelMixedPrecisionLossScaler(loss_scaler, expected_losses):
    # Common setup
    total_steps = 10
    device = 'cuda'
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
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("gradient_accumulation_steps, expected_losses", [
    (1, [10.991958, 10.975625, 11.032847, 11.034771, 10.987653,
         11.039469, 10.971498, 11.101391, 11.047601, 11.077588]),
    (4, [10.991958, 10.97373, 11.033534, 11.028931, 10.988836,
         11.04126, 10.969865, 11.085526, 11.036701, 11.0628]),
    (7, [10.991958, 10.97373, 11.033534, 11.028931, 10.994967,
         11.043544, 10.974638, 11.085087, 11.034944, 11.059022])
])
def testToyBERTModelGradientAccumulation(gradient_accumulation_steps, expected_losses):
    # Common setup
    total_steps = 10
    device = "cuda"
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
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)


def testToyBertCheckpointBasic():
    # Common setup
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    sd = trainer.state_dict()

    ## All initializers must be present in the state_dict
    ##  when the specified model for ORTTRainer is an ONNX model
    for param in trainer._onnx_model.graph.initializer:
        assert param.name in sd['model']['full_precision']

    ## Modify one of the state values and load into ORTTrainer
    sd['model']['full_precision']['bert.encoder.layer.0.attention.output.LayerNorm.weight'] += 10
    trainer.load_state_dict(sd)

    ## Save a checkpoint
    ckpt_dir = 'testdata'
    trainer.save_checkpoint(os.path.join(ckpt_dir, 'bert_toy_save_test.ortcp'))
    del trainer
    del model

    # Create a new ORTTrainer and load the checkpoint from previous ORTTrainer
    model2 = load_bert_onnx_model()
    model_desc2 = bert_model_description()
    trainer2 = orttrainer.ORTTrainer(model2, model_desc2, optim_config, options=opts)
    trainer2.load_checkpoint(os.path.join(ckpt_dir, 'bert_toy_save_test.ortcp'))
    loaded_sd = trainer2.state_dict()

    # Assert whether original state and the one loaded from checkpoint matches
    _test_commons.assert_all_states_close_ort(sd, loaded_sd)


def testToyBertCheckpointFrozenWeights():
    # Common setup
    seed = 1
    total_steps = 10
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True},
                                         'utils' : {'frozen_weights' : ['bert.encoder.layer.0.attention.self.value.weight']}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    optim_config = optim.LambConfig()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train for a few steps
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, seed)
        _ = trainer.train_step(*sample_input)
    sample_input = generate_random_input_from_model_desc(model_desc, seed + total_steps + 1)
    # Evaluate once to get a base loss
    loss = trainer.eval_step(*sample_input)
    # Save checkpoint
    state_dict = trainer.state_dict()

    # Load previous state into another instance of ORTTrainer
    model2 = load_bert_onnx_model()
    model_desc2 = bert_model_description()
    optim_config2 = optim.LambConfig()
    trainer2 = orttrainer.ORTTrainer(model2, model_desc2, optim_config2, options=opts)
    trainer2.load_state_dict(state_dict)
    # Evaluate once to get a base loss
    ckpt_loss = trainer2.eval_step(*sample_input)

    # Must match as both trainers have the same dict state
    assert_allclose(loss.cpu(), ckpt_loss.cpu())
    loaded_state_dict = trainer2.state_dict()
    _test_commons.assert_all_states_close_ort(state_dict, loaded_state_dict)

@pytest.mark.parametrize("optimizer, mixedprecision_enabled", [
    (optim.LambConfig(), False),
    (optim.AdamConfig(), False),
    (optim.LambConfig(), True),
    (optim.AdamConfig(), True),
])
def testToyBertLoadOptimState(optimizer, mixedprecision_enabled):
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
    dummy_init_state = _test_commons.generate_dummy_optim_state(model, optimizer)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    trainer.load_state_dict(dummy_init_state)
    
    # Expected values
    input_ids = torch.tensor([[26598],[21379],[19922],[ 5219],[ 5644],[20559],[23777],[25672],[22969],[16824],[16822],[635],[27399],[20647],[18519],[15546]], device=device)
    segment_ids = torch.tensor([[0],[1],[0],[1],[0],[0],[1],[0],[0],[1],[1],[0],[0],[1],[1],[1]], device=device)
    input_mask = torch.tensor([[0],[0],[0],[0],[1],[1],[1],[0],[1],[1],[0],[0],[0],[1],[0],[0]], device=device)
    masked_lm_labels = torch.tensor([[25496],[16184],[11005],[16228],[14884],[21660],[ 8678],[23083],[ 4027],[ 8397],[11921],[ 1333],[26482],[ 1666],[17925],[27978]], device=device)
    next_sentence_labels = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], device=device)

    # Actual values
    _ = trainer.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    
    actual_state_dict = trainer.state_dict()
    del actual_state_dict['model']
    _test_commons.assert_all_states_close_ort(actual_state_dict, dummy_init_state)

@pytest.mark.parametrize("model_params", [
    (['bert.embeddings.LayerNorm.bias']),
    (['bert.embeddings.LayerNorm.bias',
      'bert.embeddings.LayerNorm.weight',
      'bert.encoder.layer.0.attention.output.LayerNorm.bias']),
])
def testORTTrainerFrozenWeights(model_params):
    device = 'cuda'
    total_steps = 10
    seed = 1

    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    # Setup ORTTrainer WITHOUT frozen weights
    opts_dict = {
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    }
    opts =  orttrainer.ORTTrainerOptions(opts_dict)

    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        trainer.train_step(*sample_input)

    # All model_params must be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert all([param in session_state for param in model_params])

    # Setup ORTTrainer WITH frozen weights
    opts_dict.update({'utils' : {'frozen_weights' : model_params}})
    opts =  orttrainer.ORTTrainerOptions(opts_dict)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        trainer.train_step(*sample_input)

    # All model_params CANNOT be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert not any([param in session_state for param in model_params])

def testToyBERTSaveAsONNX():
    device = 'cuda'
    onnx_file_name = '_____temp_toy_bert_onnx_model.onnx'
    if os.path.exists(onnx_file_name):
        os.remove(onnx_file_name)
    assert not os.path.exists(onnx_file_name)

    # Load trainer
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })

    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    trainer.save_as_onnx(onnx_file_name)
    assert os.path.exists(onnx_file_name)

    with open(onnx_file_name, "rb") as f:
        bin_str = f.read()
        reload_onnx_model = onnx.load_model_from_string(bin_str)
    os.remove(onnx_file_name)

    # Create a new trainer from persisted ONNX model and compare with original ONNX model
    trainer_from_onnx = orttrainer.ORTTrainer(reload_onnx_model, model_desc, optim_config, options=opts)
    assert trainer_from_onnx._onnx_model is not None
    assert (id(trainer_from_onnx._onnx_model) != id(trainer._onnx_model))
    for initializer, loaded_initializer in zip(trainer._onnx_model.graph.initializer, trainer_from_onnx._onnx_model.graph.initializer):
        assert initializer.name == loaded_initializer.name
    assert (onnx.helper.printable_graph(trainer_from_onnx._onnx_model.graph) == onnx.helper.printable_graph(trainer._onnx_model.graph))
    _test_helpers.assert_onnx_weights(trainer, trainer_from_onnx)


###############################################################################
# Temporary tests comparing Legacy vs Experimental ORTTrainer APIs ############
###############################################################################
@pytest.mark.parametrize("optimizer_config", [
    (optim.AdamConfig),
#    (optim.LambConfig), # TODO: re-enable after nondeterminism on backend is fixed
    (optim.SGDConfig)
])
def testToyBERTModelLegacyExperimentalBasicTraining(optimizer_config):
    # Common setup
    train_steps = 512

    device = 'cuda'
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    optim_config = optimizer_config(lr=0.01)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    if optimizer_config == optim.AdamConfig:
        legacy_optimizer = 'AdamOptimizer'
    elif optimizer_config == optim.LambConfig:
        legacy_optimizer = 'LambOptimizer'
    elif optimizer_config == optim.SGDConfig:
        legacy_optimizer = 'SGDOptimizer'
    else:
        raise RuntimeError("Invalid optimizer_config")

    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(lr=optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, legacy_optimizer,
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True)
    legacy_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, True)


@pytest.mark.parametrize("initial_lr, lr_scheduler, legacy_lr_scheduler", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, _test_commons.legacy_cosine_lr_scheduler),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, _test_commons.legacy_linear_lr_scheduler),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, _test_commons.legacy_poly_lr_scheduler),
])
def testToyBERTModelLegacyExperimentalLRScheduler(initial_lr, lr_scheduler, legacy_lr_scheduler):
    ############################################################################
    # These tests require hard-coded values for 'total_steps' and 'initial_lr' #
    ############################################################################

    # Common setup
    total_steps = 128
    device = 'cuda'
    seed = 1
    warmup = 0.05
    cycles = 0.5
    power = 1.
    lr_end = 1e-7

    # Setup both Experimental and Legacy LR Schedulers before the experimental loop
    if legacy_lr_scheduler == _test_commons.legacy_constant_lr_scheduler or legacy_lr_scheduler == _test_commons.legacy_linear_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup)
    elif legacy_lr_scheduler == _test_commons.legacy_cosine_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif legacy_lr_scheduler == _test_commons.legacy_poly_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid legacy_lr_scheduler")
    if lr_scheduler == optim.lr_scheduler.ConstantWarmupLRScheduler or lr_scheduler == optim.lr_scheduler.LinearWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup)
    elif lr_scheduler == optim.lr_scheduler.CosineWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif lr_scheduler == optim.lr_scheduler.PolyWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid lr_scheduler")


    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optim.AdamConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())
        assert_allclose(trainer.options.lr_scheduler.get_last_lr()[0], legacy_lr_scheduler(i))

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(initial_lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       get_lr_this_step=legacy_lr_scheduler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)


@pytest.mark.parametrize("loss_scaler, legacy_loss_scaler", [
    (None, Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (amp.DynamicLossScaler(), Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (CustomLossScaler(), LegacyCustomLossScaler())
])
def testToyBERTModelMixedPrecisionLossScalerLegacyExperimental(loss_scaler, legacy_loss_scaler):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig(lr=0.001)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       use_mixed_precision=True,
                       loss_scaler = legacy_loss_scaler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)


@pytest.mark.parametrize("gradient_accumulation_steps", [
    (1),
    (4),
    (7)
])
def testToyBERTModelGradientAccumulationLegacyExperimental(gradient_accumulation_steps):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        loss = trainer.train_step(*sample_input)
        experimental_losses.append(loss.cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute = True,
                       gradient_accumulation_steps = gradient_accumulation_steps)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)

@pytest.mark.parametrize("params, legacy_optim_map", [
    # Change the hyper parameters for all parameters
    ([], legacy_optim_params_a),
    # Change the hyperparameters for a subset of hardcoded parameters
    ([{'params':['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight'], "alpha": 0.9,
        "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6, "do_bias_correction":False}], legacy_optim_params_b),
    # Change the hyperparameters for a generated set of paramers
    (optimizer_parameters(load_bert_onnx_model()), legacy_optim_params_c)
])
def testToyBERTModelLegacyExperimentalCustomOptimParameters(params, legacy_optim_map):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL API
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.AdamConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6, do_bias_correction=False)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(trainer.optim_config.lr)

    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                                       legacy_optim_map,
                                       learning_rate_description,
                                       device,
                                       _use_deterministic_compute=True)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]
        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

     # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)
