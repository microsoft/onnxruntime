import math


class _LRScheduler(object):
    r"""Base class for implementing custom learning rate schedulers

    Schedulers can be either stateful or stateless.
    Stateless implementation can only rely on information available at
    :py:class:`.TrainStepInfo`.
    Stateful implementation, on the other hand, can store additional parameters
    by overriding the constructor.

    In both cases, once the scheduler is configured, no user code is needed
    to update learning rate during each train step.

    NOTE: Current implementation doesn't support 'lr' within :py:attr:`param_groups` entries.
    """

    def __init__(self):
        self._last_lr = []

    def _step(self, train_step_info):
        r"""Internal method called to compute learning rate"""

        # Store last lr for future inquiry
        new_lr = self.get_lr(train_step_info)
        self._last_lr = new_lr

        return new_lr

    def get_lr(self, train_step_info):
        r"""Returns a list of learning rate

        Args:
            train_step_info (:py:class:`.TrainStepInfo`): runtime info for current training step

        Returns:
            ordered :py:obj:`list` of learning rates.
                The first entry is the default learning rate and
                    the remaining refer to each parameter group.
            NOTE: Currently, only default learning rate is supported and a single-valued list must be returned.
        """
        raise NotImplementedError

    def get_last_lr(self):
        r""" Return last computed learning rate by LR Scheduler"""
        return self._last_lr


class ConstantWarmupLRScheduler(_LRScheduler):
    r"""Constant warmup strategy for learning rate update based on HuggingFace's Transformers implementation

    Creates a schedule with constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Learning rate update strategy:
        When current_step < warmup
            lr = base_lr * (current_step / max(1, num_warmup_steps))
        Otherwise,
            lr = base_lr

    Args:
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup. Range is (0, 1]

    Example:
        .. code-block:: python

            # Initialize lr scheduler
            lr_scheduler = ConstantWarmupLRScheduler(total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, total_steps, warmup=0.002):
        super().__init__()
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0 and warmup < 1,\
            "warmup must be a float between (0, 1]"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup
        self._num_warmup_steps = warmup * total_steps

    def _warmup_constant(self, train_step_info):
        if train_step_info.optimization_step < self._num_warmup_steps:
            return float(train_step_info.optimization_step) / float(max(1, self._num_warmup_steps))
        return 1.0

    def get_lr(self, train_step_info):
        return [train_step_info.optimizer_config.lr * self._warmup_constant(train_step_info)]


class CosineWarmupLRScheduler(_LRScheduler):
    r"""Cosine warmup strategy for learning rate update based on HuggingFace's Transformers implementation

    Creates a schedule with learning rate that decreases following the values of the cosine function between the
    initial lr set in the :py:class`.optim._OptimizerConfig` to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the :py:class`.optim._OptimizerConfig`.

    Learning rate update strategy:
        When current_step < warmup
            lr = base_lr * (current_step / max(1, num_warmup_steps)), when
        Otherwise
            lr = base_lr * max(0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress))), where
                progress = current_step - num_warmup_steps / max(1, total_steps - num_warmup_steps)

    Args:
        total_steps (int): total training steps for learning.
        cycles (float, default is 0.5): number of waves in the cosine schedule.
            The default decreases from max value to 0, following a half-cosine
        warmup (float, default is 0.002): portion of total steps for warmup. Range is (0, 1]

    Example:
        .. code-block:: python

            # Initialize lr scheduler
            lr_scheduler = CosineWarmupLRScheduler(total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, total_steps, cycles=0.5, warmup=0.002):
        super().__init__()
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(cycles, float) and cycles > 0,\
            "cycles must be a positive float"
        assert isinstance(warmup, float) and warmup >= 0 and warmup < 1,\
            "warmup must be a float between (0, 1]"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.cycles = cycles
        self.warmup = warmup
        self._num_warmup_steps = warmup * total_steps

    def _warmup_cosine(self, train_step_info):
        if train_step_info.optimization_step < self._num_warmup_steps:
            return float(train_step_info.optimization_step) / float(max(1, self._num_warmup_steps))
        progress = float(train_step_info.optimization_step - self._num_warmup_steps) / float(max(1, self.total_steps - self._num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

    def get_lr(self, train_step_info):
        return [train_step_info.optimizer_config.lr * self._warmup_cosine(train_step_info)]


class LinearWarmupLRScheduler(_LRScheduler):
    r"""Linear warmup strategy for learning rate update based on HuggingFace's Transformers implementation

    Creates a schedule with a learning rate that decreases linearly from the initial lr
    set in the :py:class`.optim._OptimizerConfig` to 0, after a warmup period during which
    it increases linearly from 0 to the initial lr set in the :py:class`.optim._OptimizerConfig`.

    Learning rate update strategy:
        When current_step < warmup
            lr = base_lr * (current_step / max(1, num_warmup_steps))
        Otherwise
            lr = base_lr * (max(0, total_steps - current_step) / max(1, total_steps - num_warmup_steps)))

    Args:
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup. Range is (0, 1]

    Example:
        .. code-block:: python

            # Initialize lr scheduler
            lr_scheduler = LinearWarmupLRScheduler(total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, total_steps, warmup=0.002):
        super().__init__()
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0 and warmup < 1,\
            "warmup must be a float between (0, 1]"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup
        self._num_warmup_steps = warmup * total_steps

    def _warmup_linear(self, train_step_info):
        if train_step_info.optimization_step < self._num_warmup_steps:
            return float(train_step_info.optimization_step) / float(max(1, self._num_warmup_steps))
        return max(0.0, float(self.total_steps - train_step_info.optimization_step) / float(max(1, self.total_steps - self._num_warmup_steps)))

    def get_lr(self, train_step_info):
        return [train_step_info.optimizer_config.lr * self._warmup_linear(train_step_info)]


class PolyWarmupLRScheduler(_LRScheduler):
    r"""Polynomial warmup strategy for learning rate update based on HuggingFace's Transformers implementation

    Creates a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the :py:class`.optim._OptimizerConfig` to  lr_end,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the :py:class`.optim._OptimizerConfig`

    Learning rate update strategy:
        When current_step < warmup
            lr = base_lr * (current_step / max(1, num_warmup_steps))
        When current_step > total_steps
            lr = lr_end / lr
        Otherwise
            lr =  decay / lr, where decay is
                (lr - lr_end) * (1 - (current_step - num_warmup_steps) / (total_steps - num_warmup_steps)) ** power + lr_end

    Args:
        total_steps (int): total training steps for learning.
        lr_end (float, default 1e-7): final learning rate value.
            Applies to the default lr and parameter groups in :py:class:`.optim._OptimizerConfig`
        power (float, default is 1.0): polynomial factor
        warmup (float, default is 0.002): portion of total steps for warmup. Range is (0, 1]

    Example:
        .. code-block:: python

            # Initialize lr scheduler
            lr_scheduler = PolyWarmupLRScheduler(total_steps=512, warmup=0.002, degree=0.5)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, total_steps, lr_end=1e-7, power=1.0, warmup=0.002):
        super().__init__()
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(lr_end, float) and lr_end >= 0,\
            "lr_end must be a positive float"
        assert isinstance(warmup, float) and warmup >= 0 and warmup < 1,\
            "warmup must be a float between (0, 1]"
        assert isinstance(power, float) and power >= 0,\
            "power must be a positive float"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.lr_end = lr_end
        self.power = power
        self.warmup = warmup
        self._num_warmup_steps = warmup * total_steps

    def _warmup_poly(self, train_step_info):

        assert train_step_info.optimizer_config.lr > self.lr_end,\
            f"lr_end ({lr_end}) must be be smaller than initial lr ({train_step_info.optimizer_config.lr})"

        if train_step_info.optimization_step < self._num_warmup_steps:
            return float(train_step_info.optimization_step) / float(max(1, self._num_warmup_steps))
        elif train_step_info.optimization_step > self.total_steps:
            return self.lr_end / train_step_info.optimizer_config.lr
        else:
            lr_range = train_step_info.optimizer_config.lr - self.lr_end
            decay_steps = self.total_steps - self._num_warmup_steps
            pct_remaining = 1 - (train_step_info.optimization_step - self._num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / train_step_info.optimizer_config.lr


    def get_lr(self, train_step_info):
        return [train_step_info.optimizer_config.lr * self._warmup_poly(train_step_info)]
