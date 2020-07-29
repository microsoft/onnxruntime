import inspect
import onnx
import os
import pytest
import torch

from numpy.testing import assert_allclose

from onnxruntime.capi.training import orttrainer_options as orttrainer_options
from onnxruntime.capi.training import model_desc_validation as md_val
from onnxruntime.capi.training import orttrainer, amp, optim, TrainStepInfo, _utils


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
            'gradient_accumulation_steps': 0
        },
        'device': {
            'id': 'cpu',
            'mem_limit': 0
        },
        'distributed': {
            'world_rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'allreduce_post_accumulation': False,
            'deepspeed_zero_stage': 0,
            'enable_adasum': False
        },
        'lr_scheduler': None,
        'mixed_precision': {
            'enabled': False,
            'loss_scaler': None
        },
        'utils': {
            'frozen_weights': [],
            'grad_norm_clip': False
        },
        'debug': {
            'deterministic_compute': False
        },
        '_internal_use': {
            'enable_internal_postprocess': True,
            'extra_postprocess': None,
            'onnx_opset_version' : 12
        }
    }

    actual_values = orttrainer_options.ORTTrainerOptions(test_input)
    assert actual_values._validated_opts == expected_values


def testORTTrainerOptionsInvalidMixedPrecisionEnabledSchema():
    '''Test an invalid input based on schema validation error message'''

    expected_msg = "Invalid options: {'mixed_precision': [{'enabled': ['must be of boolean type']}]}"
    with pytest.raises(ValueError) as e:
        orttrainer_options.ORTTrainerOptions(
            {'mixed_precision': {'enabled': 1}})
    assert str(e.value) == expected_msg


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
    assert model_description.learning_rate.name == "Learning_Rate"
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

    # Append type to inputs/outputs tuples
    for idx, i_desc in enumerate(model_description.inputs):
        model_description.add_type_to_input_description(idx, input_dtype[idx])
    for idx, o_desc in enumerate(model_description.outputs):
        model_description.add_type_to_output_description(idx, output_dtype[idx])

    # Verify inputs/outputs tuples are replaced by the typed counterparts
    for idx, i_desc in enumerate(model_description.inputs):
        assert len(i_desc) == 3
        assert isinstance(i_desc, model_description._InputDescriptionTyped)
        assert input_dtype[idx] == i_desc.dtype
    for idx, o_desc in enumerate(model_description.outputs):
        assert len(o_desc) == 4
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
    rtol = 1e-5
    default_scaler = amp.loss_scaler.DynamicLossScaler()

    # Initial state
    train_step_info = orttrainer.TrainStepInfo(all_finite=True, step=0,
                                               optimizer_config=None)
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
            default_scaler.update(train_step_info)
            assert default_scaler._stable_steps_count == i
            assert_allclose(default_scaler.loss_scale, loss_scale,
                            rtol=rtol, err_msg=f"loss scale mismatch at update {i}")

        # 2000th update without overflow doubles the loss and zero stable steps until max_loss_scale is reached
        default_scaler.update(train_step_info)
        if cycles <= 8:
            loss_scale *= 2
        assert default_scaler._stable_steps_count == 0
        assert_allclose(default_scaler.loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # After 8 cycles, loss scale should be float(1 << 16)*(2**8)
    assert_allclose(default_scaler.loss_scale, float(1 << 16)
                    * (2**8), rtol=rtol, err_msg="loss scale mismatch")

    # After 9 cycles, loss scale reaches max_loss_scale and it is not doubled from that point on
    loss_scale = float(1 << 16)*(2**8)
    for count in range(1, 2050):
        default_scaler.update(train_step_info)
        assert default_scaler._stable_steps_count == (count % 2000)
        assert_allclose(default_scaler.loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # Setting train_step_info.all_finite = False to test down scaling
    train_step_info.all_finite = False

    # Performing 24 updates to half the loss scale each time
    loss_scale = float(1 << 16)*(2**8)
    for count in range(1, 25):
        default_scaler.update(train_step_info)
        loss_scale /= 2
        assert default_scaler._stable_steps_count == 0
        assert_allclose(default_scaler.loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")

    # After 24 updates with gradient overflow, loss scale is 1.0
    assert_allclose(default_scaler.loss_scale, 1.,
                    rtol=rtol, err_msg="loss scale mismatch")

    # After 25 updates, min_loss_scale is reached and loss scale is not halfed from that point on
    for count in range(1, 5):
        default_scaler.update(train_step_info)
        assert default_scaler._stable_steps_count == 0
        assert_allclose(default_scaler.loss_scale, loss_scale,
                        rtol=rtol, err_msg="loss scale mismatch")


def testDynamicLossScalerCustomValues():
    rtol = 1e-5
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

    step_info = orttrainer.TrainStepInfo(all_finite=True, step=2, optimizer_config=optim.SGDConfig())
    assert step_info.all_finite is True
    assert step_info.step == 2
    assert isinstance(step_info.optimizer_config, optim._OptimizerConfig)

    step_info = orttrainer.TrainStepInfo()
    assert step_info.all_finite is None
    assert step_info.step is None
    assert step_info.optimizer_config is None


@pytest.mark.parametrize("test_input", [
    (-1),
    ('Hello'),
])
def testTrainStepInfoInvalidAllFinite(test_input):
    '''Test invalid initialization of TrainStepInfo'''
    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(all_finite=test_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(step=test_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(optimizer_config=test_input)


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
    rtol = 1e-03
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


def testSGDConfig():
    '''Test initialization of SGD'''
    cfg = optim.SGDConfig()
    assert cfg.name == 'SGDOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    cfg = optim.SGDConfig(lr=0.002)
    assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    # SGD does not support params
    with pytest.raises(AssertionError) as e:
        params = [{'params': ['layer1.weight'], 'lr': 0.1}]
        optim.SGDConfig(params=params, lr=0.002)
        assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert str(e.value) == "'params' must be an empty list for SGD optimizer"


def testAdamConfig():
    '''Test initialization of Adam'''
    cfg = optim.AdamConfig()
    assert cfg.name == 'AdamOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol,
                    err_msg="lambda_coef mismatch")
    assert_allclose(1e-8, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"
    assert cfg.weight_decay_mode == optim.AdamConfig.DecayMode.BEFORE_WEIGHT_UPDATE, "weight_decay_mode mismatch"


def testLambConfig():
    '''Test initialization of Lamb'''
    cfg = optim.LambConfig()
    assert cfg.name == 'LambOptimizer'
    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol,
                    err_msg="lambda_coef mismatch")
    assert cfg.ratio_min == float('-inf'), "ratio_min mismatch"
    assert cfg.ratio_max == float('inf'), "ratio_max mismatch"
    assert_allclose(1e-6, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"


@pytest.mark.parametrize("optim_name", [
    ('Adam'),
    ('Lamb')
])
def testParamparams(optim_name):
    rtol = 1e-5
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
def testInvalidParamparams(optim_name):
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
    (optim.lr_scheduler.ConstantWarmupLRScheduler, [0.181818, 0.066116, 0.036063, 0.026228, 0.023843,
                                                    0.023843, 0.023843, 0.023843, 0.023843, 0.023843]),
    (optim.lr_scheduler.CosineWarmupLRScheduler, [0.181818, 0.066116, 0.036063, 0.026228, 0.023843,
                                                  0.010225, 0.002989, 0.0005158, 0.000040937, 0.0000008291]),
    (optim.lr_scheduler.LinearWarmupLRScheduler, [0.181818, 0.066116, 0.036063, 0.026228, 0.023843,
                                                  0.021675, 0.0157636, 0.0085983, 0.0031266, 0.00056847]),
    (optim.lr_scheduler.PolyWarmupLRScheduler, [0.181818, 0.066116, 0.036063, 0.026228, 0.023843,
                                                0.0160749, 0.0096935, 0.0050622, 0.0021585, 0.000650833])
])
def testLRSchedulerUpdateImpl(lr_scheduler, expected_values):
    rtol = 1e-04

    # Initial state
    initial_lr = 1
    total_steps = 10
    warmup = 0.5
    optimizer_config = optim.SGDConfig(lr=initial_lr)
    lr_scheduler = lr_scheduler(total_steps,
                                warmup)

    # First half is warmup
    for step in range(total_steps):
        # Emulate ORTTRainer.train_step() call that updates its train_step_info
        train_step_info = TrainStepInfo(step=step, optimizer_config=optimizer_config)

        lr_scheduler._step(train_step_info)
        lr_list = lr_scheduler.get_last_lr()
        assert len(lr_list) == 1
        assert_allclose(lr_list[0],
                        expected_values[step], rtol=rtol, err_msg="lr mismatch")


@pytest.mark.parametrize("step_fn", [
    ('train_step'),
    ('eval_step')
])
def testInstantiateORTTrainer(step_fn):
    # Loading external TransformerModel model for testing
    # A manual import is done as this example is not part of onnxruntime package,
    # but resides on the onnxruntime repo
    pytorch_transformer_path = os.path.join('..', '..', '..', 'samples', 'python', 'pytorch_transformer')
    pt_model_path = os.path.join(pytorch_transformer_path, 'pt_model.py')
    pt_model_name = 'pt_model'
    pt_model = _utils.import_module_from_file(pt_model_path, pt_model_name)
    ort_utils_path = os.path.join(pytorch_transformer_path, 'ort_utils.py')
    ort_utils_name = 'ort_utils'
    ort_utils = _utils.import_module_from_file(ort_utils_path, ort_utils_name)
    utils_path = os.path.join(pytorch_transformer_path, 'utils.py')
    utils_name = 'utils'
    utils = _utils.import_module_from_file(utils_path, utils_name)

    # Modeling
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2)
    my_loss = ort_utils.my_loss
    model_desc = ort_utils.transformer_model_description()
    optim_config = optim.LambConfig()

    # Create ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss)

    # Preparing data
    train_data, val_data, _ = utils.prepare_data('cpu', 20, 20)

    # Export model to ONNX
    if step_fn == 'eval_step':
        step_fn = trainer.eval_step
        data, targets = utils.get_batch(val_data, 0)
        output = trainer.eval_step(data, targets)
        # TODO: move this test outside when train_step is implemented
        for out, desc in zip(output, trainer.model_desc.outputs):
            if trainer.loss_fn and desc.is_loss:
                continue
            assert list(out.size()) == desc.shape
    elif step_fn == 'train_step':
        step_fn = trainer.train_step
        data, targets = utils.get_batch(train_data, 0)
        _ = trainer.train_step(data, targets)
    else:
        raise ValueError('Invalid step_fn')
    assert trainer._onnx_model is not None

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
    file_name = os.path.join('..','..','..','temp_onnx_model.onnx')
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


def testEvalStep():
    # Loading external samples for testing
    pytorch_transformer_path = os.path.join('..','..','..','samples','python','pytorch_transformer')
    pt_model_path = os.path.join(pytorch_transformer_path,'pt_model.py')
    pt_model_name = 'pt_model'
    pt_model = _utils.import_module_from_file(pt_model_path, pt_model_name)
    ort_utils_path = os.path.join(pytorch_transformer_path,'ort_utils.py')
    ort_utils_name = 'ort_utils'
    ort_utils = _utils.import_module_from_file(ort_utils_path, ort_utils_name)
    utils_path = os.path.join(pytorch_transformer_path,'utils.py')
    utils_name = 'utils'
    utils = _utils.import_module_from_file(utils_path, utils_name)

    # Modeling
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2)
    my_loss = ort_utils.my_loss
    model_desc = ort_utils.transformer_model_description()
    optim_config = optim.LambConfig()

    device = "cpu"
    opts = {'device' : {'id' : device}}

    # Create ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)

    # Prep data
    train_data, val_data, test_data = utils.prepare_data('cpu', 20, 20)

    # Train
    for batch, i in enumerate(range(0, train_data.size(0)-1, 35)):
        data, targets = utils.get_batch(train_data, i)
        output = trainer.eval_step(data, targets) # removed learning rate here and in model desc
        break
    for output_elem, model_desc_output in zip(output, model_desc['outputs']):
        assert list(output_elem.size()) == model_desc_output[1]
