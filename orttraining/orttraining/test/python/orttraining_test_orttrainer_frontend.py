from functools import partial
import inspect
import math
from numpy.testing import assert_allclose
import onnx
import os
import pytest
import tempfile
import torch
import torch.nn.functional as F

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options
import _test_commons,_test_helpers
from onnxruntime import SessionOptions


###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize("test_input", [
    ({}),
    ({'batch': {},
      'device': {},
      'distributed': {},
      'mixed_precision': {},
      'utils': {},
      '_internal_use': {}})
])
def testORTTrainerOptionsDefaultValues(test_input):
    ''' Test different ways of using default values for incomplete input'''

    expected_values = {
        'batch': {
            'gradient_accumulation_steps': 1
        },
        'device': {
            'id': 'cuda',
            'mem_limit': 0
        },
        'distributed': {
            'world_rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'data_parallel_size': 1,
            'horizontal_parallel_size': 1,
            'pipeline_parallel' : {
                'pipeline_parallel_size': 1,
                'num_pipeline_micro_batches':1,
                'pipeline_cut_info_string': '',
                'sliced_schema' : {},
                'sliced_axes' : {},
                'sliced_tensor_names': []
            },
            'allreduce_post_accumulation': False,
            'data_parallel_size': 1,
            'horizontal_parallel_size':1,
            'deepspeed_zero_optimization': {
                'stage' : 0,
            },
            'enable_adasum': False,
        },
        'lr_scheduler': None,
        'mixed_precision': {
            'enabled': False,
            'loss_scaler': None
        },
        'graph_transformer': {
            'attn_dropout_recompute': False,
            'gelu_recompute': False,
            'transformer_layer_recompute': False,
            'number_recompute_layers': 0
        },
        'utils': {
            'frozen_weights': [],
            'grad_norm_clip': True,
            'invertible_layer_norm_gradient': False,
            'run_symbolic_shape_infer': False
        },
        'debug': {
            'deterministic_compute': False,
            'check_model_export': False,
            'graph_save_paths' : {
                'model_after_graph_transforms_path': '',
                'model_with_gradient_graph_path': '',
                'model_with_training_graph_path': '',
                'model_with_training_graph_after_optimization_path': ''
            }
        },
        '_internal_use': {
            'enable_internal_postprocess': True,
            'extra_postprocess': None,
            'onnx_opset_version' : 12,
            'enable_onnx_contrib_ops': True,
        },
        'provider_options':{},
        'session_options': None,
    }

    actual_values = orttrainer_options.ORTTrainerOptions(test_input)
    assert actual_values._validated_opts == expected_values


@pytest.mark.parametrize("input,error_msg", [
    ({'mixed_precision': {'enabled': 1}},\
        "Invalid options: {'mixed_precision': [{'enabled': ['must be of boolean type']}]}")
])
def testORTTrainerOptionsInvalidMixedPrecisionEnabledSchema(input, error_msg):
    '''Test an invalid input based on schema validation error message'''

    with pytest.raises(ValueError) as e:
        orttrainer_options.ORTTrainerOptions(input)
    assert str(e.value) == error_msg


@pytest.mark.parametrize("input_dict,input_dtype,output_dtype", [
    ({'inputs': [('in0', [])],
      'outputs': [('out0', []), ('out1', [])]},(torch.int,),(torch.float,torch.int32,)),
    ({'inputs': [('in0', ['batch', 2, 3])],
      'outputs': [('out0', [], True)]}, (torch.int8,), (torch.int16,)),
    ({'inputs': [('in0', []), ('in1', [1]), ('in2', [1, 2]), ('in3', [1000, 'dyn_ax1']), ('in4', ['dyn_ax1', 'dyn_ax2', 'dyn_ax3'])],
      'outputs': [('out0', [], True), ('out1', [1], False), ('out2', [1, 'dyn_ax1', 3])]},
        (torch.float,torch.uint8,torch.bool,torch.double,torch.half,), (torch.float,torch.float,torch.int64))
])
def testORTTrainerModelDescValidSchemas(input_dict, input_dtype, output_dtype):
    r''' Test different ways of using default values for incomplete input'''

    model_description = md_val._ORTTrainerModelDesc(input_dict)

    # Validating hard-coded learning rate description
    assert model_description.learning_rate.name == md_val.LEARNING_RATE_IO_DESCRIPTION_NAME
    assert model_description.learning_rate.shape == [1]
    assert model_description.learning_rate.dtype == torch.float32

    # Validating model description from user
    for idx, i_desc in enumerate(model_description.inputs):
        assert isinstance(i_desc, model_description._InputDescription)
        assert len(i_desc) == 2
        assert input_dict['inputs'][idx][0] == i_desc.name
        assert input_dict['inputs'][idx][1] == i_desc.shape
    for idx, o_desc in enumerate(model_description.outputs):
        assert isinstance(o_desc, model_description._OutputDescription)
        assert len(o_desc) == 3
        assert input_dict['outputs'][idx][0] == o_desc.name
        assert input_dict['outputs'][idx][1] == o_desc.shape
        is_loss = input_dict['outputs'][idx][2] if len(input_dict['outputs'][idx]) == 3 else False
        assert is_loss == o_desc.is_loss

    # Set all_finite name and check its description
    model_description.all_finite = md_val.ALL_FINITE_IO_DESCRIPTION_NAME
    assert model_description.all_finite.name == md_val.ALL_FINITE_IO_DESCRIPTION_NAME
    assert model_description.all_finite.shape == [1]
    assert model_description.all_finite.dtype == torch.bool

    # Set loss_scale_input and check its description
    model_description.loss_scale_input = md_val.LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME
    assert model_description.loss_scale_input.name == md_val.LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME
    assert model_description.loss_scale_input.shape == []
    assert model_description.loss_scale_input.dtype == torch.float32

    # Append type to inputs/outputs tuples
    for idx, i_desc in enumerate(model_description.inputs):
        model_description.add_type_to_input_description(idx, input_dtype[idx])
    for idx, o_desc in enumerate(model_description.outputs):
        model_description.add_type_to_output_description(idx, output_dtype[idx])

    # Verify inputs/outputs tuples are replaced by the typed counterparts
    for idx, i_desc in enumerate(model_description.inputs):
        assert isinstance(i_desc, model_description._InputDescriptionTyped)
        assert input_dtype[idx] == i_desc.dtype
    for idx, o_desc in enumerate(model_description.outputs):
        assert isinstance(o_desc, model_description._OutputDescriptionTyped)
        assert output_dtype[idx] == o_desc.dtype


@pytest.mark.parametrize("input_dict,error_msg", [
    ({'inputs': [(True, [])],
      'outputs': [(True, [])]},
      "Invalid model_desc: {'inputs': [{0: ['the first element of the tuple (aka name) must be a string']}], "
                           "'outputs': [{0: ['the first element of the tuple (aka name) must be a string']}]}"),
    ({'inputs': [('in1', None)],
      'outputs': [('out1', None)]},
      "Invalid model_desc: {'inputs': [{0: ['the second element of the tuple (aka shape) must be a list']}], "
                           "'outputs': [{0: ['the second element of the tuple (aka shape) must be a list']}]}"),
    ({'inputs': [('in1', [])],
     'outputs': [('out1', [], None)]},
     "Invalid model_desc: {'outputs': [{0: ['the third element of the tuple (aka is_loss) must be a boolean']}]}"),
    ({'inputs': [('in1', [True])],
      'outputs': [('out1', [True])]},
      "Invalid model_desc: {'inputs': [{0: ['each shape must be either a string or integer']}], "
                           "'outputs': [{0: ['each shape must be either a string or integer']}]}"),
    ({'inputs': [('in1', [])],
      'outputs': [('out1', [], True), ('out2', [], True)]},
      "Invalid model_desc: {'outputs': [{1: ['only one is_loss can bet set to True']}]}"),
    ({'inputz': [('in1', [])],
      'outputs': [('out1', [], True)]},
      "Invalid model_desc: {'inputs': ['required field'], 'inputz': ['unknown field']}"),
    ({'inputs': [('in1', [])],
      'outputz': [('out1', [], True)]},
      "Invalid model_desc: {'outputs': ['required field'], 'outputz': ['unknown field']}"),
])
def testORTTrainerModelDescInvalidSchemas(input_dict, error_msg):
    r''' Test different ways of using default values for incomplete input'''
    with pytest.raises(ValueError) as e:
        md_val._ORTTrainerModelDesc(input_dict)
    assert str(e.value) == error_msg


def testDynamicLossScaler():
    rtol = 1e-7
    default_scaler = amp.loss_scaler.DynamicLossScaler()

    # Initial state
    train_step_info = orttrainer.TrainStepInfo(optim.LambConfig())
    assert_allclose(default_scaler.loss_scale, float(1 << 16),
                    rtol=rtol, err_msg="loss scale mismatch")
    assert default_scaler.up_scale_window == 2000
    assert_allclose(default_scaler.min_loss_scale, 1.0,
                    rtol=rtol, err_msg="min loss scale mismatch")
    assert_allclose(default_scaler.max_loss_scale, float(
        1 << 24), rtol=rtol, err_msg="max loss scale mismatch")

    # Performing 9*2000 updates to cover all branches of LossScaler.update(train_step_info.all_finite=True)
    loss_scale = float(1 << 16)
    for cycles in range(1, 10):

        # 1999 updates without overflow produces 1999 stable steps
        for i in range(1, 2000):
            new_loss_scale = default_scaler.update(train_step_info)
            assert default_scaler._stable_steps_count == i
            assert_allclose(new_loss_scale, loss_scale,
                            rtol=rtol, err_msg=f"loss scale mismatch at update {i}")

        # 2000th update without overflow doubles the loss and zero stable steps until max_loss_scale is reached
        new_loss_scale = default_scaler.update(train_step_info)
        if cycles <= 8:
            loss_scale *= 2
        assert default_scaler._stable_steps_count == 0
        assert_allclose(new_loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # After 8 cycles, loss scale should be float(1 << 16)*(2**8)
    assert_allclose(new_loss_scale, float(1 << 16)
                    * (2**8), rtol=rtol, err_msg="loss scale mismatch")

    # After 9 cycles, loss scale reaches max_loss_scale and it is not doubled from that point on
    loss_scale = float(1 << 16)*(2**8)
    for count in range(1, 2050):
        new_loss_scale = default_scaler.update(train_step_info)
        assert default_scaler._stable_steps_count == (count % 2000)
        assert_allclose(new_loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # Setting train_step_info.all_finite = False to test down scaling
    train_step_info.all_finite = False

    # Performing 24 updates to half the loss scale each time
    loss_scale = float(1 << 16)*(2**8)
    for count in range(1, 25):
        new_loss_scale = default_scaler.update(train_step_info)
        loss_scale /= 2
        assert default_scaler._stable_steps_count == 0
        assert_allclose(new_loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # After 24 updates with gradient overflow, loss scale is 1.0
    assert_allclose(new_loss_scale, 1.,
                    rtol=rtol, err_msg="loss scale mismatch")

    # After 25 updates, min_loss_scale is reached and loss scale is not halfed from that point on
    for count in range(1, 5):
        new_loss_scale = default_scaler.update(train_step_info)
        assert default_scaler._stable_steps_count == 0
        assert_allclose(new_loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")


def testDynamicLossScalerCustomValues():
    rtol = 1e-7
    scaler = amp.loss_scaler.DynamicLossScaler(automatic_update=False,
                                               loss_scale=3,
                                               up_scale_window=7,
                                               min_loss_scale=5,
                                               max_loss_scale=10)
    assert scaler.automatic_update == False
    assert_allclose(scaler.loss_scale, 3, rtol=rtol,
                    err_msg="loss scale mismatch")
    assert_allclose(scaler.min_loss_scale, 5, rtol=rtol,
                    err_msg="min loss scale mismatch")
    assert_allclose(scaler.max_loss_scale, 10, rtol=rtol,
                    err_msg="max loss scale mismatch")
    assert scaler.up_scale_window == 7


def testTrainStepInfo():
    '''Test valid initializations of TrainStepInfo'''

    optimizer_config = optim.LambConfig()
    fetches=['out1','out2']
    step_info = orttrainer.TrainStepInfo(optimizer_config=optimizer_config,
                                         all_finite=False,
                                         fetches=fetches,
                                         optimization_step=123,
                                         step=456)
    assert step_info.optimizer_config == optimizer_config
    assert step_info.all_finite == False
    assert step_info.fetches == fetches
    assert step_info.optimization_step == 123
    assert step_info.step == 456

    step_info = orttrainer.TrainStepInfo(optimizer_config)
    assert step_info.optimizer_config == optimizer_config
    assert step_info.all_finite == True
    assert step_info.fetches == []
    assert step_info.optimization_step == 0
    assert step_info.step == 0


@pytest.mark.parametrize("invalid_input", [
    (-1),
    ('Hello'),
])
def testTrainStepInfoInvalidInput(invalid_input):
    '''Test invalid initialization of TrainStepInfo'''
    optimizer_config = optim.LambConfig()
    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config=invalid_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config, all_finite=invalid_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config, fetches=invalid_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config, optimization_step=invalid_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config, step=invalid_input)


@pytest.mark.parametrize("optim_name,lr,alpha,default_alpha", [
    ('AdamOptimizer', .1, .2, None),
    ('LambOptimizer', .2, .3, None),
    ('SGDOptimizer', .3, .4, None),
    ('SGDOptimizer', .3, .4, .5)
])
def testOptimizerConfig(optim_name, lr, alpha, default_alpha):
    '''Test initialization of _OptimizerConfig'''
    defaults = {'lr': lr, 'alpha': alpha}
    params = [{'params': ['fc1.weight', 'fc2.weight']}]
    if default_alpha is not None:
        params[0].update({'alpha': default_alpha})
    else:
        params[0].update({'alpha': alpha})
    cfg = optim.config._OptimizerConfig(
        name=optim_name, params=params, defaults=defaults)

    assert cfg.name == optim_name
    rtol = 1e-07
    assert_allclose(defaults['lr'],
                    cfg.lr, rtol=rtol, err_msg="lr mismatch")

    # 1:1 mapping between defaults and params's hyper parameters
    for param in params:
        for k, _ in param.items():
            if k != 'params':
                assert k in cfg.defaults, "hyper parameter {k} not present in one of the parameter params"
    for k, _ in cfg.defaults.items():
        for param in cfg.params:
            assert k in param, "hyper parameter {k} not present in one of the parameter params"


@pytest.mark.parametrize("optim_name,defaults,params", [
    ('AdamOptimizer', {'lr': -1}, []),  # invalid lr
    ('FooOptimizer', {'lr': 0.001}, []),  # invalid name
    ('SGDOptimizer', [], []),  # invalid type(defaults)
    (optim.AdamConfig, {'lr': 0.003}, []),  # invalid type(name)
    ('AdamOptimizer', {'lr': None}, []),  # missing 'lr' hyper parameter
    ('SGDOptimizer', {'lr': 0.004}, {}),  # invalid type(params)
    # invalid type(params[i])
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [[]]),
    # missing 'params' at 'params'
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [{'alpha': 1}]),
    # missing 'alpha' at 'defaults'
    ('AdamOptimizer', {'lr': 0.005}, [{'params': 'param1', 'alpha': 1}]),
])
def testOptimizerConfigInvalidInputs(optim_name, defaults, params):
    '''Test invalid initialization of _OptimizerConfig'''

    with pytest.raises(AssertionError):
        optim.config._OptimizerConfig(
            name=optim_name, params=params, defaults=defaults)


def testOptimizerConfigSGD():
    '''Test initialization of SGD'''
    cfg = optim.SGDConfig()
    assert cfg.name == 'SGDOptimizer'

    rtol = 1e-07
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    cfg = optim.SGDConfig(lr=0.002)
    assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    # SGD does not support params
    with pytest.raises(AssertionError) as e:
        params = [{'params': ['layer1.weight'], 'lr': 0.1}]
        optim.SGDConfig(params=params, lr=0.002)
        assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert str(e.value) == "'params' must be an empty list for SGD optimizer"


def testOptimizerConfigAdam():
    '''Test initialization of Adam'''
    cfg = optim.AdamConfig()
    assert cfg.name == 'AdamOptimizer'

    rtol = 1e-7
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol,
                    err_msg="lambda_coef mismatch")
    assert_allclose(1e-8, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert_allclose(1.0, cfg.max_norm_clip, rtol=rtol, err_msg="max_norm_clip mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"
    assert cfg.weight_decay_mode == optim.AdamConfig.DecayMode.BEFORE_WEIGHT_UPDATE, "weight_decay_mode mismatch"


def testOptimizerConfigLamb():
    '''Test initialization of Lamb'''
    cfg = optim.LambConfig()
    assert cfg.name == 'LambOptimizer'
    rtol = 1e-7
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol,
                    err_msg="lambda_coef mismatch")
    assert cfg.ratio_min == float('-inf'), "ratio_min mismatch"
    assert cfg.ratio_max == float('inf'), "ratio_max mismatch"
    assert_allclose(1e-6, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert_allclose(1.0, cfg.max_norm_clip, rtol=rtol, err_msg="max_norm_clip mismatch")
    assert cfg.do_bias_correction == False, "do_bias_correction mismatch"


@pytest.mark.parametrize("optim_name", [
    ('Adam'),
    ('Lamb')
])
def testOptimizerConfigParams(optim_name):
    rtol = 1e-7
    params = [{'params': ['layer1.weight'], 'alpha': 0.1}]
    if optim_name == 'Adam':
        cfg = optim.AdamConfig(params=params, alpha=0.2)
    elif optim_name == 'Lamb':
        cfg = optim.LambConfig(params=params, alpha=0.2)
    else:
        raise ValueError('invalid input')
    assert len(cfg.params) == 1, "params should have length 1"
    assert_allclose(cfg.params[0]['alpha'], 0.1,
                    rtol=rtol, err_msg="invalid lr on params[0]")


@pytest.mark.parametrize("optim_name", [
    ('Adam'),
    ('Lamb')
])
def testOptimizerConfigInvalidParams(optim_name):
    # lr is not supported within params
    with pytest.raises(AssertionError) as e:
        params = [{'params': ['layer1.weight'], 'lr': 0.1}]
        if optim_name == 'Adam':
            optim.AdamConfig(params=params, lr=0.2)
        elif optim_name == 'Lamb':
            optim.LambConfig(params=params, lr=0.2)
        else:
            raise ValueError('invalid input')
    assert str(e.value) == "'lr' is not supported inside params"


def testLinearLRSchedulerCreation():
    total_steps = 10
    warmup = 0.05

    lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(total_steps,
                                                              warmup)

    # Initial state
    assert lr_scheduler.total_steps == total_steps
    assert lr_scheduler.warmup == warmup


@pytest.mark.parametrize("lr_scheduler,expected_values", [
    (optim.lr_scheduler.ConstantWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (optim.lr_scheduler.CosineWarmupLRScheduler,
        [0.0, 0.9763960957919413, 0.9059835861602854, 0.7956724530494887, 0.6563036824392345,\
         0.5015739416158049, 0.34668951940611276, 0.2068719061737831, 0.09586187986225325, 0.0245691111902418]),
    (optim.lr_scheduler.LinearWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]),
    (optim.lr_scheduler.PolyWarmupLRScheduler,
        [0.0, 0.9509018036072144, 0.9008016032064128, 0.8507014028056112, 0.8006012024048097,\
         0.750501002004008, 0.7004008016032064, 0.6503006012024048, 0.6002004008016032, 0.5501002004008015])
])
def testLRSchedulerUpdateImpl(lr_scheduler, expected_values):
    # Test tolerance
    rtol = 1e-03

    # Initial state
    initial_lr = 1
    total_steps = 10
    warmup = 0.5
    optimizer_config = optim.SGDConfig(lr=initial_lr)
    lr_scheduler = lr_scheduler(total_steps, warmup)

    # First half is warmup
    for optimization_step in range(total_steps):
        # Emulate ORTTRainer.train_step() call that updates its train_step_info
        train_step_info = TrainStepInfo(optimizer_config=optimizer_config, optimization_step=optimization_step)

        lr_scheduler._step(train_step_info)
        lr_list = lr_scheduler.get_last_lr()
        assert len(lr_list) == 1
        assert_allclose(lr_list[0],
                        expected_values[optimization_step], rtol=rtol, err_msg="lr mismatch")

def testInstantiateORTTrainerOptions():
    session_options = SessionOptions()
    session_options.enable_mem_pattern = False
    provider_options = {'EP1': {'key':'val'}}
    opts = {'session_options' : session_options, 
            'provider_options' : provider_options}
    opts = orttrainer.ORTTrainerOptions(opts)
    assert(opts.session_options.enable_mem_pattern is False)
    assert(opts._validated_opts['provider_options']['EP1']['key'] == 'val')

@pytest.mark.parametrize("step_fn, lr_scheduler, expected_lr_values, device", [
    ('train_step', None, None, 'cuda'),
    ('eval_step', None, None, 'cpu'),
    ('train_step', optim.lr_scheduler.ConstantWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], 'cpu'),
    ('train_step', optim.lr_scheduler.CosineWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9045084971874737, 0.6545084971874737, 0.34549150281252633, 0.09549150281252633],
        'cuda'),
    ('train_step', optim.lr_scheduler.LinearWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2], 'cpu'),
    ('train_step', optim.lr_scheduler.PolyWarmupLRScheduler,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.80000002, 0.60000004, 0.40000006000000005, 0.20000007999999997], 'cuda')
])
def testInstantiateORTTrainer(step_fn, lr_scheduler, expected_lr_values, device):
    total_steps = 1
    initial_lr = 1.
    rtol = 1e-3

    # PyTorch Transformer model as example
    opts = {'device' : {'id' : device}}
    if lr_scheduler:
        total_steps = 10
        opts.update({'lr_scheduler' : lr_scheduler(total_steps=total_steps, warmup=0.5)})
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=initial_lr)
    model, model_desc, my_loss, batcher_fn, train_data, val_data, _ = _test_commons._load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)

    # Run a train or evaluation step
    if step_fn == 'eval_step':
        data, targets = batcher_fn(val_data, 0)
    elif step_fn == 'train_step':
        data, targets = batcher_fn(train_data, 0)
    else:
        raise ValueError('Invalid step_fn')

    # Export model to ONNX
    if step_fn == 'eval_step':
        step_fn = trainer.eval_step
        output = trainer.eval_step(data, targets)
    elif step_fn == 'train_step':
        step_fn = trainer.train_step
        for i in range(total_steps):
            output = trainer.train_step(data, targets)
            if lr_scheduler:
                lr_list = trainer.options.lr_scheduler.get_last_lr()
                assert_allclose(lr_list[0], expected_lr_values[i], rtol=rtol, err_msg="lr mismatch")
    else:
        raise ValueError('Invalid step_fn')
    assert trainer._onnx_model is not None

    # Check output shape after train/eval step
    for out, desc in zip(output, trainer.model_desc.outputs):
        if trainer.loss_fn and desc.is_loss:
            continue
        assert list(out.size()) == desc.shape

    # Check name, shape and dtype of the first len(forward.parameters) ORT graph inputs
    sig = inspect.signature(model.forward)
    for i in range(len(sig.parameters.keys())):
        input_name = trainer.model_desc.inputs[i][0]
        input_dim = trainer.model_desc.inputs[i][1]
        input_type = trainer.model_desc.inputs[i][2]

        assert trainer._onnx_model.graph.input[i].name == input_name
        for dim_idx, dim in enumerate(trainer._onnx_model.graph.input[i].type.tensor_type.shape.dim):
            assert input_dim[dim_idx] == dim.dim_value
            assert input_type == _utils.dtype_onnx_to_torch(
                trainer._onnx_model.graph.input[i].type.tensor_type.elem_type)

    # Check name, shape and dtype of the ORT graph outputs
    for i in range(len(trainer.model_desc.outputs)):
        output_name = trainer.model_desc.outputs[i][0]
        output_dim = trainer.model_desc.outputs[i][1]
        output_type = trainer.model_desc.outputs[i][3]

        assert trainer._onnx_model.graph.output[i].name == output_name
        for dim_idx, dim in enumerate(trainer._onnx_model.graph.output[i].type.tensor_type.shape.dim):
            assert output_dim[dim_idx] == dim.dim_value
            assert output_type == _utils.dtype_onnx_to_torch(
                trainer._onnx_model.graph.output[i].type.tensor_type.elem_type)

    # Save current model as ONNX as a file
    file_name = os.path.join('_____temp_onnx_model.onnx')
    trainer.save_as_onnx(file_name)
    assert os.path.exists(file_name)
    with open(file_name, "rb") as f:
        bin_str = f.read()
        reload_onnx_model = onnx.load_model_from_string(bin_str)
    os.remove(file_name)

    # Create a new trainer from persisted ONNX model and compare with original ONNX model
    trainer_from_onnx = orttrainer.ORTTrainer(reload_onnx_model, model_desc, optim_config)
    step_fn(data, targets)
    assert trainer_from_onnx._onnx_model is not None
    assert (id(trainer_from_onnx._onnx_model) != id(trainer._onnx_model))
    assert (trainer_from_onnx._onnx_model == trainer._onnx_model)
    assert (trainer_from_onnx._onnx_model.graph == trainer._onnx_model.graph)
    assert (onnx.helper.printable_graph(trainer_from_onnx._onnx_model.graph) == onnx.helper.printable_graph(trainer._onnx_model.graph))


@pytest.mark.parametrize("seed, device", [
    (0, 'cpu'),
    (24, 'cuda')
])
def testORTDeterministicCompute(seed, device):
    # Common setup
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device' : {
            'id' : device,
            'mem_limit' : 10*1024*1024
        }
    })

    # Setup for the first ORTTRainer run
    torch.manual_seed(seed)
    set_seed(seed)
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    first_trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    data, targets = batcher_fn(train_data, 0)
    _ = first_trainer.train_step(data, targets)
    assert first_trainer._onnx_model is not None

    # Setup for the second ORTTRainer run
    torch.manual_seed(seed)
    set_seed(seed)
    model, _, _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device)
    second_trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    _ = second_trainer.train_step(data, targets)
    assert second_trainer._onnx_model is not None

    # Compare two different instances with identical setup
    assert id(first_trainer._onnx_model) != id(second_trainer._onnx_model)
    _test_helpers.assert_onnx_weights(first_trainer, second_trainer)


@pytest.mark.parametrize("seed,device,expected_loss,fetches", [
    (321, 'cuda', [10.5774, 10.4403, 10.4175, 10.2886, 10.2760], False),
    (321, 'cuda', [10.5774, 10.4403, 10.4175, 10.2886, 10.2760], True),
])
def testORTTrainerMixedPrecisionLossScaler(seed, device, expected_loss, fetches):
    return # TODO: re-enable after nondeterminism on backend is fixed. update numbers

    rtol = 1e-3
    total_steps = len(expected_loss)
    torch.manual_seed(seed)
    set_seed(seed)

    # Setup ORTTrainer
    loss_scaler = amp.DynamicLossScaler()
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'mixed_precision' : {
                                                'enabled' : True,
                                                'loss_scaler' : loss_scaler},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, val_data, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    actual_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        if fetches:
            trainer._train_step_info.fetches=['loss']
            loss = trainer.train_step(data, targets)
        else:
            loss, _ = trainer.train_step(data, targets)
        actual_loss.append(loss.cpu())

    # Eval once just to test fetches in action
    val_data, val_targets = batcher_fn(val_data, 0)
    if fetches:
        trainer._train_step_info.fetches=['loss']
        loss = trainer.eval_step(val_data, val_targets)
        trainer._train_step_info.fetches=[]
    loss, _ = trainer.eval_step(val_data, val_targets)

    # Compare loss to ground truth computed from current ORTTrainer API
    _test_helpers.assert_model_outputs(expected_loss, actual_loss, True, rtol=rtol)
    assert trainer._onnx_model is not None


def _recompute_data():
    device_capability_major = torch.cuda.get_device_capability()[0]
    if device_capability_major == 7:    # V100 for Dev machine
        expected_loss = [10.5732, 10.4407, 10.3701, 10.2778, 10.1824]
        return [
            (False, False, False, 0, expected_loss),    # no recompute
            (True, False, False, 0, expected_loss),     # attn_dropout recompute
            (False, True, False, 0, expected_loss),     # gelu recompute
            (False, False, True, 0, expected_loss),     # transformer_layer recompute
            (False, False, True, 1, expected_loss),     # transformer_layer recompute with 1 layer
        ]
    elif device_capability_major == 5:  # M60 for CI machines
        expected_loss = [10.5445, 10.4389, 10.3480, 10.2627, 10.2113]
        return [
            (False, False, False, 0, expected_loss),    # no recompute
            (True, False, False, 0, expected_loss),     # attn_dropout recompute
            (False, True, False, 0, expected_loss),     # gelu recompute
            (False, False, True, 0, expected_loss),     # transformer_layer recompute
            (False, False, True, 1, expected_loss),     # transformer_layer recompute with 1 layer
        ]
@pytest.mark.parametrize("attn_dropout, gelu, transformer_layer, number_layers, expected_loss", _recompute_data())
def testORTTrainerRecompute(attn_dropout, gelu, transformer_layer, number_layers, expected_loss):
    seed = 321
    device = 'cuda'
    rtol = 1e-3
    total_steps = len(expected_loss)
    torch.manual_seed(seed)
    set_seed(seed)

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'graph_transformer' : {
                                                'attn_dropout_recompute': attn_dropout,
                                                'gelu_recompute': gelu,
                                                'transformer_layer_recompute': transformer_layer,
                                                'number_recompute_layers': number_layers
                                            },
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, val_data, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    actual_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        actual_loss.append(loss.cpu())

    # Compare loss to ground truth computed from current ORTTrainer API
    _test_helpers.assert_model_outputs(expected_loss, actual_loss, True, rtol=rtol)
    assert trainer._onnx_model is not None


@pytest.mark.parametrize("seed,device,gradient_accumulation_steps,total_steps,expected_loss", [
    (0, 'cuda', 1, 12, [10.5368022919, 10.4146203995, 10.3635568619, 10.2650547028, 10.2284049988, 10.1304626465,\
        10.0853414536, 9.9987659454, 9.9472427368, 9.8832416534, 9.8223171234, 9.8222122192]),
    (42, 'cuda', 3, 12, [10.6455879211, 10.6247081757, 10.6361322403, 10.5187482834, 10.5345087051, 10.5487670898,\
        10.4833698273, 10.4600019455, 10.4535751343, 10.3774127960, 10.4144191742, 10.3757553101]),
    (123, 'cuda', 7, 12, [10.5353469849, 10.5261383057, 10.5240392685, 10.5013713837, 10.5678377151, 10.5452117920,\
        10.5184345245, 10.4271221161, 10.4458627701, 10.4864749908, 10.4416503906, 10.4467563629]),
    (321, 'cuda', 12, 12, [10.5773944855, 10.5428829193, 10.5974750519, 10.5416746140, 10.6009902954, 10.5684127808,\
        10.5759754181, 10.5636739731, 10.5613927841, 10.5825119019, 10.6031589508, 10.6199369431]),
])
def testORTTrainerGradientAccumulation(seed, device, gradient_accumulation_steps, total_steps, expected_loss):
    return # TODO: re-enable after nondeterminism on backend is fixed. update numbers
    rtol = 1e-3
    torch.manual_seed(seed)
    set_seed(seed)

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'batch' : {'gradient_accumulation_steps' : gradient_accumulation_steps},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    actual_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        actual_loss.append(loss.cpu())

    # Compare legacy vs experimental APIs
    _test_helpers.assert_model_outputs(expected_loss, actual_loss, rtol=rtol)


@pytest.mark.parametrize("dynamic_axes", [
    (True),
    (False),
])
def testORTTrainerDynamicShape(dynamic_axes):
    # Common setup
    device = 'cuda'

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({})
    model, model_desc, my_loss, batcher_fn,\
        train_data, _, _ = _test_commons._load_pytorch_transformer_model(device, dynamic_axes=dynamic_axes)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    total_steps = 10
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        if dynamic_axes:
            # Forcing batches with different sizes to exercise dynamic shapes
            data = data[:-(i+1)]
            targets = targets[:-(i+1)*data.size(1)]
        _, _ = trainer.train_step(data, targets)

    assert trainer._onnx_model is not None


@pytest.mark.parametrize('enable_onnx_contrib_ops', [
    (True),
    (False),
])
def testORTTrainerInternalUseContribOps(enable_onnx_contrib_ops):
    # Common setup
    device = 'cuda'

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({"_internal_use": {"enable_onnx_contrib_ops": enable_onnx_contrib_ops}})
    model, model_desc, my_loss, batcher_fn,\
        train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    data, targets = batcher_fn(train_data, 0)
    if not enable_onnx_contrib_ops:
        with pytest.raises(Exception) as e_info:
            _, _ = trainer.train_step(data, targets)
    else:
        _, _ = trainer.train_step(data, targets)


@pytest.mark.parametrize("model_params", [
    (['decoder.weight',
      'transformer_encoder.layers.0.linear1.bias',
      'transformer_encoder.layers.0.linear2.weight',
      'transformer_encoder.layers.1.self_attn.out_proj.weight',
      'transformer_encoder.layers.1.self_attn.out_proj.bias']),
])
def testORTTrainerFrozenWeights(model_params):
    # Common setup
    device = 'cuda'
    total_steps = 10

    # Setup ORTTrainer WITHOUT frozen weights
    options = orttrainer.ORTTrainerOptions({})
    model, model_desc, my_loss, batcher_fn,\
        train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        _, _ = trainer.train_step(data, targets)

    # All model_params must be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert all([param in session_state for param in model_params])


    # Setup ORTTrainer WITH frozen weights
    options = orttrainer.ORTTrainerOptions({'utils' : {'frozen_weights' : model_params}})
    model, _, _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        _, _ = trainer.train_step(data, targets)

    # All model_params CANNOT be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert not all([param in session_state for param in model_params])


@pytest.mark.parametrize("loss_scaler, optimizer_config, gradient_accumulation_steps", [
    (None, optim.AdamConfig(), 1),
    (None, optim.LambConfig(), 1),
    (None, optim.SGDConfig(), 1),
    (amp.DynamicLossScaler(), optim.AdamConfig(), 1),
    (amp.DynamicLossScaler(), optim.LambConfig(), 5),
    #(amp.DynamicLossScaler(), optim.SGDConfig(), 1), # SGD doesnt support fp16
])
def testORTTrainerStateDictWrapModelLossFn(loss_scaler, optimizer_config, gradient_accumulation_steps):
    # Common setup
    seed = 1
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4)
        def forward(self, y=None, x=None):
            if y is not None:
                return self.linear(x) + y
            else:
                return self.linear(x) + torch.ones(2, 4)
    model_desc = {'inputs' : [('x', [2, 2]),
                              ('label', [2, ])],
                  'outputs' : [('loss', [], True),
                               ('output', [2, 4])]}

    # Dummy data
    data1 = torch.randn(2, 2)
    label1 = torch.tensor([0, 1], dtype=torch.int64)
    data2 = torch.randn(2, 2)
    label2 = torch.tensor([0, 1], dtype=torch.int64)

    # Setup training based on test parameters
    opts =  {'debug' : {'deterministic_compute': True},
             'batch' : { 'gradient_accumulation_steps' : gradient_accumulation_steps}}
    if loss_scaler:
        opts['mixed_precision'] = { 'enabled': True, 'loss_scaler': loss_scaler}
    opts =  orttrainer.ORTTrainerOptions(opts)

    # Training session 1
    torch.manual_seed(seed)
    set_seed(seed)
    pt_model = LinearModel()
    def loss_fn(x, label):
        return F.nll_loss(F.log_softmax(x, dim=1), label)
    trainer = orttrainer.ORTTrainer(pt_model, model_desc, optimizer_config, loss_fn=loss_fn, options=opts)

    # Check state_dict keys before train. Must be empty
    state_dict = trainer.state_dict()
    assert state_dict == {}

    # Train once and check initial state
    trainer.train_step(x=data1, label=label1)
    state_dict = trainer.state_dict()
    assert all([weight in state_dict['model']['full_precision'].keys() for weight in ['linear.bias', 'linear.weight']])

    # Initialize training session 2 from state of Training 1
    torch.manual_seed(seed)
    set_seed(seed)
    trainer2 = orttrainer.ORTTrainer(pt_model, model_desc, optimizer_config, loss_fn=loss_fn, options=opts)
    trainer2.load_state_dict(state_dict)

    # Verify state was loaded properly
    _test_commons.assert_all_states_close_ort(state_dict, trainer2._load_state_dict.args[0])

    # Perform a second step in both training session 1 and 2 and verify they match
    trainer.train_step(x=data2, label=label2)
    state_dict = trainer.state_dict()
    trainer2.train_step(x=data2, label=label2)
    state_dict2 = trainer2.state_dict()
    _test_commons.assert_all_states_close_ort(state_dict, state_dict2)


def testORTTrainerNonPickableModel():
    # Common setup
    import threading
    seed = 1
    class UnpickableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4)
            self._lock = threading.Lock()

        def forward(self, y=None, x=None):
            with self._lock:
                if y is not None:
                    return self.linear(x) + y
                else:
                    return self.linear(x) + torch.ones(2, 4)

    model_desc = {'inputs' : [('x', [2, 2]),
                              ('label', [2, ])],
                  'outputs' : [('loss', [], True),
                               ('output', [2, 4])]}

    # Dummy data
    data = torch.randn(2, 2)
    label = torch.tensor([0, 1], dtype=torch.int64)

    # Setup training based on test parameters
    opts =  orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True}})

    # Training session
    torch.manual_seed(seed)
    set_seed(seed)
    pt_model = UnpickableModel()
    def loss_fn(x, label):
        return F.nll_loss(F.log_softmax(x, dim=1), label)
    optim_config = optim.AdamConfig()
    trainer = orttrainer.ORTTrainer(pt_model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # Train must succeed despite warning
    _, _ = trainer.train_step(data, label)

###############################################################################
# Temporary tests comparing Legacy vs Experimental ORTTrainer APIs ############
###############################################################################


@pytest.mark.parametrize("seed,device", [
    (1234, 'cuda')
])
def testORTTrainerLegacyAndExperimentalWeightsCheck(seed, device):
    # Common data
    rtol = 1e-7
    total_steps = 5

    # Setup for the experimental ORTTRainer run
    torch.manual_seed(seed)
    set_seed(seed)
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({
        'device' : {
            'id' : device
        },
        'debug' : {
            'deterministic_compute': True
        },
    })
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    # Training loop
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        _ = trainer.train_step(data, targets)

    # Setup for the legacy ORTTrainer run
    torch.manual_seed(seed)
    set_seed(seed)
    model, (model_desc, lr_desc), _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device, legacy_api=True)
    legacy_trainer = Legacy_ORTTrainer(model, my_loss, model_desc, "LambOptimizer", None, lr_desc,
                                       device, _use_deterministic_compute=True)
    # Training loop
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        _, _ = legacy_trainer.train_step(data, targets, torch.tensor([optim_config.lr]))

    # Compare legacy vs experimental APIs
    _test_helpers.assert_legacy_onnx_weights(trainer, legacy_trainer, rtol=rtol)


@pytest.mark.parametrize("seed,device", [
    (321, 'cuda'),
])
def testORTTrainerLegacyAndExperimentalPrecisionLossScaler(seed, device):
    # Common data
    total_steps = 128

    # Setup experimental API
    torch.manual_seed(seed)
    set_seed(seed)
    loss_scaler = amp.DynamicLossScaler()
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'mixed_precision' : {
                                                'enabled' : True,
                                                'loss_scaler' : loss_scaler},
                                            'debug' : {'deterministic_compute' : True,}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    # Training loop
    experimental_loss = []
    experimental_preds_dtype = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        exp_loss, exp_preds = trainer.train_step(data, targets)
        experimental_loss.append(exp_loss.cpu())
        experimental_preds_dtype.append(exp_preds.dtype)

    # Setup legacy API
    torch.manual_seed(seed)
    set_seed(seed)
    model, (model_desc, lr_desc), _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device, legacy_api=True)
    loss_scaler = Legacy_LossScaler('ort_test_input_loss_scalar', True)
    legacy_trainer = Legacy_ORTTrainer(model, my_loss, model_desc, "LambOptimizer",
                                       None, lr_desc, device=device,
                                       _use_deterministic_compute=True,
                                       use_mixed_precision=True,
                                       loss_scaler=loss_scaler)
    # Training loop
    legacy_loss = []
    legacy_preds_dtype = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        leg_loss, leg_preds = legacy_trainer.train_step(data, targets, torch.tensor([optim_config.lr]))
        legacy_loss.append(leg_loss.cpu())
        legacy_preds_dtype.append(leg_preds.dtype)

    # Compare legacy vs experimental APIs
    assert experimental_preds_dtype == legacy_preds_dtype
    _test_helpers.assert_legacy_onnx_weights(trainer, legacy_trainer)
    _test_helpers.assert_model_outputs(legacy_loss, experimental_loss)


@pytest.mark.parametrize("seed,device,gradient_accumulation_steps,total_steps", [
    (0, 'cuda', 1, 12),
    (42, 'cuda', 3, 12),
    (123, 'cuda', 7, 12),
    (321, 'cuda', 12, 12),
])
def testORTTrainerLegacyAndExperimentalGradientAccumulation(seed, device, gradient_accumulation_steps, total_steps):
    # Common data
    torch.set_printoptions(precision=10)

    # Setup experimental API
    torch.manual_seed(seed)
    set_seed(seed)
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'batch' : {'gradient_accumulation_steps' : gradient_accumulation_steps},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    # Training loop
    experimental_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        exp_loss, _ = trainer.train_step(data, targets)
        experimental_loss.append(exp_loss.cpu())

    # Setup legacy API
    torch.manual_seed(seed)
    set_seed(seed)
    model, (model_desc, lr_desc), _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device, legacy_api=True)
    legacy_trainer = Legacy_ORTTrainer(model, my_loss, model_desc, "LambOptimizer",
                                       None, lr_desc, device=device,
                                       _use_deterministic_compute=True,
                                       gradient_accumulation_steps=gradient_accumulation_steps)
    # Training loop
    legacy_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        leg_loss, _ = legacy_trainer.train_step(data, targets, torch.tensor([optim_config.lr]))
        legacy_loss.append(leg_loss.cpu())

    # Compare legacy vs experimental APIs
    _test_helpers.assert_model_outputs(legacy_loss, experimental_loss)


@pytest.mark.parametrize("seed,device,optimizer_config,lr_scheduler, get_lr_this_step", [
    (0, 'cuda', optim.AdamConfig, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (0, 'cuda', optim.LambConfig, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (0, 'cuda', optim.SGDConfig, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (42, 'cuda', optim.AdamConfig, optim.lr_scheduler.LinearWarmupLRScheduler, _test_commons.legacy_linear_lr_scheduler),
    (42, 'cuda', optim.LambConfig, optim.lr_scheduler.LinearWarmupLRScheduler, _test_commons.legacy_linear_lr_scheduler),
    (42, 'cuda', optim.SGDConfig, optim.lr_scheduler.LinearWarmupLRScheduler, _test_commons.legacy_linear_lr_scheduler),
    (123, 'cuda', optim.AdamConfig, optim.lr_scheduler.CosineWarmupLRScheduler, _test_commons.legacy_cosine_lr_scheduler),
    (123, 'cuda', optim.LambConfig, optim.lr_scheduler.CosineWarmupLRScheduler, _test_commons.legacy_cosine_lr_scheduler),
    (123, 'cuda', optim.SGDConfig, optim.lr_scheduler.CosineWarmupLRScheduler, _test_commons.legacy_cosine_lr_scheduler),
    (321, 'cuda', optim.AdamConfig, optim.lr_scheduler.PolyWarmupLRScheduler, _test_commons.legacy_poly_lr_scheduler),
    (321, 'cuda', optim.LambConfig, optim.lr_scheduler.PolyWarmupLRScheduler, _test_commons.legacy_poly_lr_scheduler),
    (321, 'cuda', optim.SGDConfig, optim.lr_scheduler.PolyWarmupLRScheduler, _test_commons.legacy_poly_lr_scheduler),
])
def testORTTrainerLegacyAndExperimentalLRScheduler(seed, device, optimizer_config, lr_scheduler, get_lr_this_step):
    # Common data
    total_steps = 10
    lr = 0.001
    warmup = 0.5
    cycles = 0.5
    power = 1.
    lr_end = 1e-7
    torch.set_printoptions(precision=10)

    # Setup experimental API
    torch.manual_seed(seed)
    set_seed(seed)
    if lr_scheduler == optim.lr_scheduler.ConstantWarmupLRScheduler or lr_scheduler == optim.lr_scheduler.LinearWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup)
    elif lr_scheduler == optim.lr_scheduler.CosineWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif lr_scheduler == optim.lr_scheduler.PolyWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid lr_scheduler")

    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'debug' : {'deterministic_compute' : True},
                                            'lr_scheduler' : lr_scheduler})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optimizer_config(lr=lr)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    # Training loop
    experimental_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        exp_loss, exp_preds = trainer.train_step(data, targets)
        experimental_loss.append(exp_loss.cpu())

    # Setup legacy API
    torch.manual_seed(seed)
    set_seed(seed)

    if optimizer_config == optim.AdamConfig:
        legacy_optimizer_config = 'AdamOptimizer'
    elif optimizer_config == optim.LambConfig:
        legacy_optimizer_config = 'LambOptimizer'
    elif optimizer_config == optim.SGDConfig:
        legacy_optimizer_config = 'SGDOptimizer'
    else:
        raise RuntimeError("Invalid optimizer_config")

    if get_lr_this_step == _test_commons.legacy_constant_lr_scheduler or get_lr_this_step == _test_commons.legacy_linear_lr_scheduler:
        get_lr_this_step = partial(get_lr_this_step, initial_lr=lr, total_steps=total_steps, warmup=warmup)
    elif get_lr_this_step == _test_commons.legacy_cosine_lr_scheduler:
        get_lr_this_step = partial(get_lr_this_step, initial_lr=lr, total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif get_lr_this_step == _test_commons.legacy_poly_lr_scheduler:
        get_lr_this_step = partial(get_lr_this_step, initial_lr=lr, total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid get_lr_this_step")

    model, (model_desc, lr_desc), _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device, legacy_api=True)
    legacy_trainer = Legacy_ORTTrainer(model, my_loss, model_desc, legacy_optimizer_config,
                                       None, lr_desc, device=device,
                                       _use_deterministic_compute=True,
                                       get_lr_this_step=get_lr_this_step)
    # Training loop
    legacy_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        leg_loss, leg_preds = legacy_trainer.train_step(data, targets)
        legacy_loss.append(leg_loss.cpu())

    # Compare legacy vs experimental APIs
    _test_helpers.assert_model_outputs(legacy_loss, experimental_loss)


def testLossScalerLegacyAndExperimentalFullCycle():
    info = orttrainer.TrainStepInfo(optimizer_config=optim.LambConfig(lr=0.001), all_finite=True, fetches=[], optimization_step=0, step=0)
    new_ls = amp.DynamicLossScaler()
    old_ls = Legacy_LossScaler("ort_test_input_loss_scaler", True)

    # Initial state
    train_step_info = orttrainer.TrainStepInfo(optim.LambConfig())
    assert_allclose(new_ls.loss_scale, old_ls.loss_scale_)
    assert new_ls.up_scale_window == old_ls.up_scale_window_
    assert_allclose(new_ls.min_loss_scale, old_ls.min_loss_scale_)
    assert_allclose(new_ls.max_loss_scale, old_ls.max_loss_scale_)

    # Performing 9*2000 updates to cover all branches of LossScaler.update(train_step_info.all_finite=True)
    for cycles in range(1, 10):

        # 1999 updates without overflow produces 1999 stable steps
        for i in range(1, 2000):
            new_loss_scale = new_ls.update(train_step_info)
            old_ls.update_loss_scale(train_step_info.all_finite)
            old_loss_scale = old_ls.loss_scale_
            assert new_ls._stable_steps_count == old_ls.stable_steps_
            # import pdb; pdb.set_trace()
            assert_allclose(new_loss_scale, old_loss_scale)

        # 2000th update without overflow doubles the loss and zero stable steps until max_loss_scale is reached
        new_loss_scale = new_ls.update(train_step_info)
        old_ls.update_loss_scale(train_step_info.all_finite)
        old_loss_scale = old_ls.loss_scale_
        assert new_ls._stable_steps_count == old_ls.stable_steps_
        assert_allclose(new_loss_scale, old_loss_scale)

    # After 8 cycles, loss scale should be float(1 << 16)*(2**8)
    assert_allclose(new_loss_scale, old_loss_scale)

    # After 9 cycles, loss scale reaches max_loss_scale and it is not doubled from that point on
    for count in range(1, 2050):
        new_loss_scale = new_ls.update(train_step_info)
        old_ls.update_loss_scale(train_step_info.all_finite)
        old_loss_scale = old_ls.loss_scale_
        assert new_ls._stable_steps_count == old_ls.stable_steps_
        assert_allclose(new_loss_scale, old_loss_scale)

    # Setting train_step_info.all_finite = False to test down scaling
    train_step_info.all_finite = False

    # Performing 24 updates to half the loss scale each time
    for count in range(1, 25):
        new_loss_scale = new_ls.update(train_step_info)
        old_ls.update_loss_scale(train_step_info.all_finite)
        old_loss_scale = old_ls.loss_scale_
        assert new_ls._stable_steps_count == old_ls.stable_steps_
        assert_allclose(new_loss_scale, old_loss_scale)

    # After 24 updates with gradient overflow, loss scale is 1.0
    assert_allclose(new_loss_scale, old_loss_scale)

    # After 25 updates, min_loss_scale is reached and loss scale is not halfed from that point on
    for count in range(1, 5):
        new_loss_scale = new_ls.update(train_step_info)
        old_ls.update_loss_scale(train_step_info.all_finite)
        old_loss_scale = old_ls.loss_scale_
        assert new_ls._stable_steps_count == old_ls.stable_steps_
        assert_allclose(new_loss_scale, old_loss_scale)


def testLossScalerLegacyAndExperimentalRandomAllFinite():
    new_ls = amp.DynamicLossScaler()
    old_ls = Legacy_LossScaler("ort_test_input_loss_scaler", True)

    # Initial state
    train_step_info = orttrainer.TrainStepInfo(optim.LambConfig())
    assert_allclose(new_ls.loss_scale, old_ls.loss_scale_)
    assert new_ls.up_scale_window == old_ls.up_scale_window_
    assert_allclose(new_ls.min_loss_scale, old_ls.min_loss_scale_)
    assert_allclose(new_ls.max_loss_scale, old_ls.max_loss_scale_)

    import random
    out = []
    for _ in range(1, 64):
        train_step_info.all_finite = bool(random.getrandbits(1))
        new_loss_scale = new_ls.update(train_step_info)
        old_ls.update_loss_scale(train_step_info.all_finite)
        old_loss_scale = old_ls.loss_scale_
        assert new_ls._stable_steps_count == old_ls.stable_steps_
        assert_allclose(new_loss_scale, old_loss_scale)
        out.append(new_loss_scale)
        assert new_loss_scale > 1e-7

def testORTTrainerRunSymbolicShapeInfer():
    # Common data
    seed = 0
    total_steps = 12
    device = 'cuda'
    torch.set_printoptions(precision=10)

    # Setup without symbolic shape inference
    torch.manual_seed(seed)
    set_seed(seed)
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    # Training loop
    expected_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        expected_loss.append(loss.cpu())

    # Setup with symbolic shape inference
    torch.manual_seed(seed)
    set_seed(seed)
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001)
    options.utils.run_symbolic_shape_infer = True
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)
    # Training loop
    new_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        new_loss.append(loss.cpu())

    # Setup with symbolic shape inference in legacy API
    torch.manual_seed(seed)
    set_seed(seed)
    model, (model_desc, lr_desc), _, _, _, _, _ = _test_commons._load_pytorch_transformer_model(device, legacy_api=True)
    legacy_trainer = Legacy_ORTTrainer(model, my_loss, model_desc, "LambOptimizer",
                                       None, lr_desc, device=device,
                                       run_symbolic_shape_infer=True,
                                       _use_deterministic_compute=True)
    # Training loop
    legacy_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = legacy_trainer.train_step(data, targets, torch.tensor([optim_config.lr]))
        legacy_loss.append(loss.cpu())

    # Compare losses
    _test_helpers.assert_model_outputs(new_loss, expected_loss)
    _test_helpers.assert_model_outputs(legacy_loss, expected_loss)

@pytest.mark.parametrize("test_input", [
    ({
      'distributed': {'enable_adasum': True},
    })
])
def testORTTrainerOptionsEnabledAdasumFlag(test_input):
    ''' Test the enabled_adasum flag values when set enabled'''

    actual_values = orttrainer_options.ORTTrainerOptions(test_input)
    assert actual_values.distributed.enable_adasum == True

@pytest.mark.parametrize("test_input", [
    ({
      'distributed': {'enable_adasum': False},
    })
])
def testORTTrainerOptionsDisabledAdasumFlag(test_input):
    ''' Test the enabled_adasum flag values when set disabled'''

    actual_values = orttrainer_options.ORTTrainerOptions(test_input)
    assert actual_values.distributed.enable_adasum == False

def testORTTrainerUnusedInput():
    class UnusedInputModel(torch.nn.Module):
        def __init__(self):
            super(UnusedInputModel, self).__init__()
        def forward(self, x, y):
            return torch.mean(x)

    model = UnusedInputModel()
    model_desc = {'inputs': [('x', [1]), ('y', [1])], 'outputs': [('loss', [], True)]}
    optim_config = optim.LambConfig(lr=0.001)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config)

    # Run just one step to make sure there are no iobinding errors for the unused input.
    try:
        trainer.train_step(torch.FloatTensor([1.0]), torch.FloatTensor([1.0]))
    except RuntimeError:
        pytest.fail("RuntimeError doing train_step with an unused input.")

@pytest.mark.parametrize("debug_files", [
    {'model_after_graph_transforms_path': 'transformed.onnx',
      'model_with_gradient_graph_path': 'transformed_grad.onnx',
      'model_with_training_graph_path': 'training.onnx',
      'model_with_training_graph_after_optimization_path': 'training_optimized.onnx'
    },
    {'model_after_graph_transforms_path': 'transformed.onnx',
      'model_with_training_graph_path': ''
    },
    ])
def testTrainingGraphExport(debug_files):
    device = 'cuda'
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)

    with tempfile.TemporaryDirectory() as tempdir:
        debug_paths = {}
        for k,v in debug_files.items():
            debug_paths[k] = os.path.join(tempdir, v)
        opts =  orttrainer.ORTTrainerOptions(
            {
                "device": {"id": device},
                "debug": {"graph_save_paths": debug_paths}
            }
        )
        optim_config = optim.AdamConfig()
        trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
        data, targets = batcher_fn(train_data, 0)
        trainer.train_step(data, targets)
        for k,v in debug_files.items():
            path = debug_paths[k]
            if len(v) > 0:
                assert os.path.isfile(path)
                saved_graph = onnx.load(path).graph
                if k == 'model_with_training_graph_path':
                    assert any("AdamOptimizer" in n.op_type for n in saved_graph.node)
                elif k == 'model_with_gradient_graph_path':
                    assert any("Grad" in n.name for n in saved_graph.node)
                elif k == 'model_after_graph_transforms_path':
                    assert any("LayerNormalization" in n.op_type for n in saved_graph.node)
                elif k == 'model_with_training_graph_after_optimization_path':
                    assert any("FusedMatMul" in n.op_type for n in saved_graph.node)
                # remove saved file
                os.remove(path)
            else:
                assert not os.path.isfile(path)


def _adam_max_norm_clip_data():
    device_capability_major = torch.cuda.get_device_capability()[0]
    if device_capability_major == 7:    # V100 for Dev machine
        return [
            (0, 'cuda', 1.0, 1, 12, [10.596329, 10.087329, 9.625324, 9.254117, 8.914067,\
                8.557245, 8.296672, 8.040311, 7.780754, 7.499548, 7.229341, 7.036769]),
            (0, 'cuda', 0.1, 1, 12, [10.596329, 10.088068, 9.626670, 9.256137, 8.916809,\
                8.560838, 8.301097, 8.045413, 7.786527, 7.505644, 7.236132, 7.043610]),
            (42, 'cuda', 1.0, 1, 12, [10.659752, 10.149531, 9.646378, 9.273719, 8.938648,\
                8.595006, 8.344718, 8.100259, 7.828771, 7.541266, 7.269467, 7.083140]),
            (42, 'cuda', 0.1, 1, 12, [10.659752, 10.150211, 9.647715, 9.275835, 8.941610,\
                8.598876, 8.349401, 8.105709, 7.834774, 7.547812, 7.276530, 7.090215]),
        ]
    elif device_capability_major == 5:  # M60 for CI machines (Python Packaging Pipeline)
        return [
            (0, 'cuda', 1.0, 1, 12, [10.618382, 10.08292 ,  9.603334,  9.258133,  8.917768,  8.591574,
                                     8.318401,  8.042292,  7.783608,  7.50226 ,  7.236041,  7.035602]),
            (0, 'cuda', 0.1, 1, 12, [10.618382, 10.083632,  9.604639,  9.260109,  8.920504,  8.595082,
                                     8.322799,  8.047493,  7.78929 ,  7.508382,  7.242587,  7.042367]),
            (42, 'cuda', 1.0, 1, 12, [10.68639 , 10.102986,  9.647681,  9.293091,  8.958928,  8.625297,
                                      8.351107,  8.079577,  7.840723,  7.543044,  7.284141,  7.072688]),
            (42, 'cuda', 0.1, 1, 12, [10.68639 , 10.103672,  9.649025,  9.295167,  8.961777,  8.629059,
                                      8.355571,  8.084871,  7.846589,  7.549438,  7.290722,  7.079446]),
        ]
@pytest.mark.parametrize("seed,device,max_norm_clip,gradient_accumulation_steps,total_steps,expected_loss", _adam_max_norm_clip_data())
def testORTTrainerAdamMaxNormClip(seed, device, max_norm_clip, gradient_accumulation_steps, total_steps, expected_loss):
    rtol = 1e-5
    torch.manual_seed(seed)
    set_seed(seed)

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'batch' : {'gradient_accumulation_steps' : gradient_accumulation_steps},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.AdamConfig(lr=0.001, max_norm_clip=max_norm_clip)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    actual_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        actual_loss.append(loss.cpu().item())

    # Compare legacy vs experimental APIs
    _test_helpers.assert_model_outputs(expected_loss, actual_loss, rtol=rtol)


def _lamb_max_norm_clip_data():
    device_capability_major = torch.cuda.get_device_capability()[0]
    if device_capability_major == 7:    # V100 for Dev machine
        return [
            (0, 'cuda', 1.0, 1, 12, [10.596329, 10.509530, 10.422451, 10.359101, 10.285673, 10.200603,\
                10.152860, 10.106999, 10.033828, 9.965749, 9.895924, 9.854723]),
            (0, 'cuda', 0.1, 1, 12, [10.596329, 10.474221, 10.350412, 10.253196, 10.148172, 10.032470,\
                9.958271, 9.885362, 9.788476, 9.696474, 9.601951, 9.542482]),
            (42, 'cuda', 1.0, 1, 12, [10.659752, 10.565927, 10.437677, 10.387601, 10.302234, 10.217105,\
                10.170007, 10.143104, 10.093051, 10.002419, 9.960327, 9.895797]),
            (42, 'cuda', 0.1, 1, 12, [10.659752, 10.531717, 10.367162, 10.284177, 10.168813, 10.053536,\
                9.980052, 9.926860, 9.852230, 9.738342, 9.673130, 9.590945]),
        ]
    elif device_capability_major == 5:  # M60 for CI machines (Python Packaging Pipeline)
        return [
            (0, 'cuda', 1.0, 1, 12, [10.618382, 10.50222 , 10.403347, 10.35298 , 10.288447, 10.237399,
                                     10.184225, 10.089048, 10.008952,  9.972644,  9.897674,  9.84524 ]),
            (0, 'cuda', 0.1, 1, 12, [10.618382, 10.466732, 10.330871, 10.24715 , 10.150972, 10.069127,
                                     9.98974 ,  9.870169,  9.763693,  9.704323,  9.605957,  9.533117]),
            (42, 'cuda', 1.0, 1, 12, [10.68639 , 10.511692, 10.447308, 10.405255, 10.334866, 10.261473,
                                      10.169422, 10.107138, 10.069889,  9.97798 ,  9.928105,  9.896435]),
            (42, 'cuda', 0.1, 1, 12, [10.68639 , 10.477489, 10.376671, 10.301725, 10.200718, 10.098477,
                                      9.97995 ,  9.890104,  9.828899,  9.713555,  9.639567,  9.589856]),
        ]
@pytest.mark.parametrize("seed,device,max_norm_clip, gradient_accumulation_steps,total_steps,expected_loss", _lamb_max_norm_clip_data())
def testORTTrainerLambMaxNormClip(seed, device, max_norm_clip, gradient_accumulation_steps, total_steps, expected_loss):
    rtol = 1e-3
    torch.manual_seed(seed)
    set_seed(seed)

    # Setup ORTTrainer
    options = orttrainer.ORTTrainerOptions({'device' : {'id' : device},
                                            'batch' : {'gradient_accumulation_steps' : gradient_accumulation_steps},
                                            'debug' : {'deterministic_compute' : True}})
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(device)
    optim_config = optim.LambConfig(lr=0.001, max_norm_clip=max_norm_clip)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

    # Training loop
    actual_loss = []
    for i in range(total_steps):
        data, targets = batcher_fn(train_data, i)
        loss, _ = trainer.train_step(data, targets)
        actual_loss.append(loss.cpu().item())

    # Compare legacy vs experimental APIs
    _test_helpers.assert_model_outputs(expected_loss, actual_loss, rtol=rtol)
