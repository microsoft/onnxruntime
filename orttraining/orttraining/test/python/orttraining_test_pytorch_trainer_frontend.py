import pytest
from numpy.testing import assert_allclose

from onnxruntime.capi.training import optim


@pytest.mark.parametrize("optim_name", [
    ('AdamOptimizer'),
    ('LambOptimizer'),
    ('SGDOptimizer')
])
def testOptimizerConfigs(optim_name):
    '''Test initialization of _OptimizerConfig'''
    hyper_parameters = {'lr': 0.001, 'alpha': 0.9}
    param_groups = [{'params': ['fc1.weight', 'fc2.weight'], 'alpha':.0}]
    cfg = optim.config._OptimizerConfig(
        name=optim_name, hyper_parameters=hyper_parameters, param_groups=param_groups)

    assert cfg.name == optim_name
    rtol = 1e-03
    assert_allclose(hyper_parameters['lr'],
                    cfg.lr, rtol=rtol, err_msg="lr mismatch")


@pytest.mark.parametrize("optim_name,hyper_parameters,param_groups", [
    ('AdamOptimizer', {'lr': -1}, []),  # invalid lr
    ('FooOptimizer', {'lr': 0.001}, []),  # invalid name
    ('SGDOptimizer', [], []),  # invalid type(hyper_parameters)
    (optim.config.Adam, {'lr': 0.003}, []),  # invalid type(name)
    ('AdamOptimizer', {'lr': None}, []),  # missing 'lr' hyper parameter
    ('SGDOptimizer', {'lr': 0.004}, {}),  # invalid type(param_groups)
    # invalid type(param_groups[i])
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [[]]),
    # missing 'params' at 'param_groups'
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [{'alpha': 1}]),
    # missing 'alpha' at 'hyper_parameters'
    ('AdamOptimizer', {'lr': 0.005}, [{'params': 'param1', 'alpha': 1}]),
])
def testOptimizerConfigsInvalidInputs(optim_name, hyper_parameters, param_groups):
    '''Test invalid initialization of _OptimizerConfig'''

    with pytest.raises(AssertionError):
        optim.config._OptimizerConfig(
            name=optim_name, hyper_parameters=hyper_parameters, param_groups=param_groups)


def testSGD():
    '''Test initialization of SGD'''
    cfg = optim.config.SGD()
    assert cfg.name == 'SGDOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    cfg = optim.config.SGD(lr=0.002)
    assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")


def testAdam():
    '''Test initialization of Adam'''
    cfg = optim.config.Adam()
    assert cfg.name == 'AdamOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol,
                    err_msg="lambda_coef mismatch")
    assert_allclose(1e-8, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"
    assert cfg.weight_decay_mode == True, "weight_decay_mode mismatch"


def testLamb():
    '''Test initialization of Lamb'''
    cfg = optim.config.Lamb()
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
