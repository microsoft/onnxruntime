import pytest
from numpy.testing import assert_allclose

from onnxruntime.capi.training import optim


@pytest.mark.parametrize("optim_name", [
    ('AdamOptimizer'),
    ('LambOptimizer'),
    ('SGDOptimizer')
])
def testOptimizerConfigs(optim_name):
    '''Test initialization of _OptimizerConfig and its extensions'''
    hyper_parameters={'lr':0.001}
    cfg = optim.config._OptimizerConfig(name=optim_name, hyper_parameters=hyper_parameters, param_groups=[])
    assert cfg.name == optim_name
    rtol = 1e-03
    assert_allclose(hyper_parameters['lr'], cfg.lr, rtol=rtol, err_msg="loss mismatch")

