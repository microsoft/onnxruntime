import pytest
from numpy.testing import assert_allclose

from onnxruntime.capi.training import pytorch_trainer
from onnxruntime.capi.training import amp


def testDynamicLossScaler():
    rtol = 1e-5
    default_scaler = amp.loss_scaler.DynamicLossScaler()

    # Initial state
    train_step_info = pytorch_trainer.TrainStepInfo(
        all_finite=True, epoch=0, step=0)
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
