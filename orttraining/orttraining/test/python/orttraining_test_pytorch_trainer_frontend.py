import pytest
from numpy.testing import assert_allclose

from onnxruntime.capi.training import optim, TrainStepInfo


@pytest.mark.parametrize("optim_name", [
    ('AdamOptimizer'),
    ('LambOptimizer'),
    ('SGDOptimizer')
])
def testLinearLRSchedulerCreation(optim_name):
    rtol = 1e-03
    initial_lr = 0.5
    total_steps = 10
    warmup = 0.05

    if optim_name == 'AdamOptimizer':
        optimizer_config = optim.AdamConfig(lr=initial_lr)
    elif optim_name == 'LambOptimizer':
        optimizer_config = optim.LambConfig(lr=initial_lr)
    elif optim_name == 'SGDOptimizer':
        optimizer_config = optim.SGDConfig(lr=initial_lr)

    lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(optimizer_config,
                                                              total_steps,
                                                              warmup)

    # Initial state
    assert lr_scheduler.optimizer_config == optimizer_config
    assert lr_scheduler.total_steps == total_steps
    assert lr_scheduler.warmup == warmup
    assert_allclose(lr_scheduler.optimizer_config.hyper_parameters['lr'],
                    initial_lr, rtol=rtol, err_msg="lr mismatch")


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
    lr_scheduler = lr_scheduler(optimizer_config,
                                total_steps,
                                warmup)

    # First half is warmup
    for step in range(total_steps):
        # Emulate train step call
        train_step_info = TrainStepInfo(step=step)

        lr_scheduler._step(train_step_info)
        lr_list = lr_scheduler.get_last_lr()
        assert len(lr_list) == 1
        assert_allclose(lr_list[0],
                        expected_values[step], rtol=rtol, err_msg="lr mismatch")
