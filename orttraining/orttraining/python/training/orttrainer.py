class TrainStepInfo(object):
    r"""Private class used to store runtime information from current train step.

    After every train step, :py:meth:`ORTTrainer.train_step` updates the internal instance of
    :py:class:`.TrainStepInfo` residing on :py:class:`.ORTTrainer` with relevant information
    from the forward pass.

    This class shouldn't be accessed directly by the user, unless they really know what they are doing.
    Instead, :py:class:`.ORTTrainer` passes it to relevant class methods automatically,
    such as :py:method:`._LRScheduler.get_lr` or :py:class:`.LossScaler.update`.

    Args:
        all_finite (bool): flag that indicates whether all gradients are still finite after last step
        step (int): indicates current step

    Example:
        .. code-block:: python

            info = TrainStepInfo(all_finite=True, step=0)
            if info.all_finite:
                print(f'Yay, all gradients are finite at {step} step!')

    """

    def __init__(self, all_finite=None, step=None):
        assert all_finite is None or isinstance(all_finite, bool),\
            "all_finite must be either None or a bool"
        assert step is None or (isinstance(
            step, int) and step >= 0), "step must be either None or a positive int"

        self.all_finite = all_finite
        self.step = step
