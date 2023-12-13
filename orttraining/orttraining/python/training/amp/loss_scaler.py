class LossScaler:
    r"""Base class for implementing custom loss scaler strategies

    Once the scaler is configured, no user intervention is needed to update loss scale during training.

    Note:
        This class should never be instantiated, but used as an abstract class for custom loss scaling strategy.
    """

    def __init__(self, loss_scale):
        assert isinstance(loss_scale, (int, float)) and loss_scale > 0, "'loss_scale' must be a positive float"
        self._input_name = None
        self._loss_scale = float(loss_scale)
        self._initial_loss_scale = float(loss_scale)

    @property
    def input_name(self):
        return self._input_name

    @input_name.setter
    def input_name(self, input_name):
        assert isinstance(input_name, str), "'input_name' must be a string"
        assert input_name is None or len(input_name) > 0, "'input_name' cannot be empty"
        self._input_name = input_name

    @property
    def loss_scale(self):
        return self._loss_scale

    @loss_scale.setter
    def loss_scale(self, loss_scale):
        assert isinstance(loss_scale, (int, float)) and loss_scale > 0, "'loss_scale' must be a positive float"
        self._loss_scale = float(loss_scale)

    def reset(self):
        r"""Resets loss scaler internal state"""
        self._loss_scale = self._initial_loss_scale

    def update(self, train_step_info):
        r"""Updates loss based on user input and training session info

        Args:
            train_step_info (TrainStepInfo): last step state information

        Returns:
            Updated loss scale (float)
        """
        raise NotImplementedError


class DynamicLossScaler(LossScaler):
    r"""Default implementation for :py:class:`.LossScaler` class used for mixed precision

    This loss scaler works by assuming an initial scale, which is doubled every time a certain number of
    (stable) training steps are performed without exploding gradients (overflow or reach infinity).
    When at least one of the gradients explode, loss scale is divided by 2.

    Users can use this class in two ways:

        1. Enable mixed precision and not setting a loss scaler class. Default values are used
        2. Enable mixed precision and instantiate this class to override default arguments

    Static loss scaling can be achieved by setting :py:attr:`.automatic_update` to :py:obj:`False`
    and not performing manual :py:meth:`update` in train loop.

    Args:
        automatic_update (bool, default is False): boolean switch that allows :py:meth:`ORTTrainer.train_step`
            to automatically perform loss scaling. If False, an explicit call to :py:meth:`.update` must be done by the user,
            otherwise static loss scaling is performed.
        loss_scale (default is 1 << 16): A float that represents current loss scale
        up_scale_window (int, default is 2000): number of stable train steps before doubling loss scale
        min_loss_scale (float, default is 1): min value for the loss scale. Used when loss scale is decreased
        max_loss_scale (float, default is 1 << 24): max value for the loss scale. Used when loss scale is increased

    Example with default values:
        .. code-block:: python

            scaler1 = amp.DynamicLossScaler()
            print(f'Default loss scale is {scaler1.loss_scale}')

    Example with user specified values:
        .. code-block:: python

            scaler2 = amp.DynamicLossScaler(loss_scale=1<<8)
            print(f'Custom loss scale is {scaler2.loss_scale}')
    """

    def __init__(
        self,
        automatic_update=True,
        loss_scale=float(1 << 16),
        up_scale_window=2000,
        min_loss_scale=1.0,
        max_loss_scale=float(1 << 24),
    ):
        super().__init__(loss_scale)
        self.automatic_update = automatic_update
        self.up_scale_window = up_scale_window
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self._stable_steps_count = 0

    def reset(self):
        super().reset()
        self._stable_steps_count = 0

    def update(self, train_step_info):
        if not self.automatic_update:
            return self.loss_scale

        if train_step_info.all_finite:
            self._stable_steps_count += 1

            if self._stable_steps_count >= self.up_scale_window:
                self.loss_scale = min(self.max_loss_scale, self.loss_scale * 2)
                self._stable_steps_count = 0
        else:
            self.loss_scale = max(self.min_loss_scale, self.loss_scale / 2)
            self._stable_steps_count = 0
        return self.loss_scale
