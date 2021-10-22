from enum import IntEnum, unique


class _OptimizerConfig(object):
    r"""Base class for optimizer configuration

    This class is not an optimizer, but a means to configure existing ones from ORT backend.
    Once configured, no user intervention is needed to update weights or zero gradients during training.
    The 'parameter group' concept described at :py:attr:`.params` is borrowed from
    `Pytorch <https://pytorch.org/docs/stable/optim.html#per-parameter-options>`_.

    Args:
        name (str): optimizer names.
            One of 'SGDOptimizer', 'AdamOptimizer' and 'LambOptimizer'
        defaults (dict): optimizer parameters applied to all model parameters.
                         Used when a parameter group doesn’t specify them.
                         NOTE: Every optimizer must have 'lr'.
        params (list of dict, default is []): list of parameter groups.
            Each dict must contain a 'params' key with a list of names of model's parameter that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.defaults`
            for specific model parameters.
            Empty list means all the parameters of the model will use :py:attr:`.defaults` during optimization.

    NOTE: To prevent model parameters to be trained, refer to :py:attr:`.ORTTrainerOptions.utils.frozen_weights`.

    Example:

    .. code-block:: python

        lamb_optim = _OptimizerConfig(name = 'LambOptimizer',
                                      params = [ { 'params' : ['model_param0', 'model_param1'],
                                                   'epsilon' : 0.03, 'beta' : 0.5},
                                                 { 'params' : ['model_param2'],
                                                   'alpha' : 0.04},
                                               ],
                                      defaults = { 'lr': 0.001, 'alpha' : 0.01, 'beta' : 0.9})
    """

    def __init__(self, name, params, defaults):
        assert isinstance(name, str), "'name' must be a string"
        assert name in ['AdamOptimizer', 'LambOptimizer', 'SGDOptimizer'], \
            "'name' must be one of 'AdamOptimizer', 'LambOptimizer' or 'SGDOptimizer'"
        assert isinstance(defaults,
                          dict), "'defaults' must be a dict"
        assert 'lr' in defaults, "'defaults' must contain a {'lr' : positive number} entry"
        assert (isinstance(defaults['lr'], float) or
                isinstance(defaults['lr'], int)) and defaults['lr'] >= 0, "lr must be a positive number"
        assert isinstance(params, list), "'params' must be a list"
        for group in params:
            assert isinstance(group, dict) and len(group) > 1 and 'params' in group, \
                ("Each dict inside 'params' must contain a {'params' : [model parameter names]} entry"
                 " and additional entries for custom hyper parameter values")
            for k, _ in group.items():
                if k != 'params':
                    assert k in defaults or k.replace("_coef", "") in defaults, f"'params' has {k} hyper parameter not present at 'defaults'"

        self.name = name
        self.lr = float(defaults['lr'])
        self.defaults = defaults
        self.params = []

        # TODO: monitor this for perf issues
        # Maybe we don't have to do this to populate TrainingParameters,
        # but it does make code easier to maintain
        for param_group in params:
            self._add_param_group(param_group)

    def _add_param_group(self, param_group):
        r"""Add a parameter group to the :py:class:`_OptimizerConfig` s `params`."""
        assert isinstance(param_group, dict), "param group must be a dict"

        # Each parameter group must have all hyper parameters set
        for name, value in self.defaults.items():
            if name not in param_group:
                param_group.setdefault(name, value)

        if "lambda_coef" in param_group:
            param_group["lambda"] = param_group.pop("lambda_coef")

        self.params.append(param_group)


class SGDConfig(_OptimizerConfig):
    r"""SGD optimizer configuration

    NOTE: Current implementation does not support :py:attr:`params`, and must be
    passed as an empty list.

    Args:
        params (list of dict, default is []): list of parameter groups.
            Each dict must contain a 'params' key with a list of names of model's parameter that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.defaults`
            for specific model parameters.
            Empty list means all the parameters of the model will use :py:attr:`.defaults` during optimization.
        lr (float, default is 0.001): Learning rate

    NOTE: To prevent model parameters to be trained, refer to :py:attr:`.ORTTrainerOptions.utils.frozen_weights`.

    Example:

    .. code-block:: python

        sgd_optim1 = SGDConfig(lr=0.001)
    """

    def __init__(self, params=[], lr=0.001):
        super().__init__(name='SGDOptimizer',
                         params=params,
                         defaults={'lr': lr})
        assert isinstance(params, list) and len(params) == 0, "'params' must be an empty list for SGD optimizer"


class AdamConfig(_OptimizerConfig):
    r"""Adam optimizer configuration

    Args:
        params (list of dict, default is []): list of parameter groups.
            Each dict must contain a 'params' key with a list of names of model's parameter that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.defaults`
            for specific model parameters.
            Empty list means all the parameters of the model will use :py:attr:`.defaults` during optimization.
        lr (float, default is 0.001): Learning rate
        alpha (float, default is 0.9): Coefficient of previous gradient in running average of row 1.
        beta (float, default is 0.999):  Coefficient of previous squared gradient in running average.
        lambda_coef (float, default is 0): Regularization coefficient.
        epsilon (float, default is 1e-8): Small scalar to avoid dividing by zero.
        do_bias_correction (bool, default is True): Compute unbiased 1st and 2nd momentums.
        weight_decay_mode (DecayMode, default is BEFORE_WEIGHT_UPDATE): Selects weight decay update strategy.

    NOTE: To prevent model parameters to be trained, refer to :py:attr:`.ORTTrainerOptions.utils.frozen_weights`.

    Example:

    .. code-block:: python

        # User-specified lr, alpha and weight_decay_mode for all model parameters
        adam_optim1 = AdamConfig(lr=0.01, alpha=0.85, weight_decay_mode=AdamConfig.DecayMode.AFTER_WEIGHT_UPDATE)

        # User-specified lr using parameters group
        adam_optim2 = AdamConfig([{'params':['fc1.weight','fc2.weight'], 'lr':0.005}], lr=0.01)
    """

    @unique
    class DecayMode(IntEnum):
        BEFORE_WEIGHT_UPDATE = 0,
        AFTER_WEIGHT_UPDATE = 1

    def __init__(self, params=[], lr=0.001, alpha=0.9, beta=0.999, lambda_coef=0.0, epsilon=1e-8, max_norm_clip=1.0,
                 do_bias_correction=True, weight_decay_mode=DecayMode.BEFORE_WEIGHT_UPDATE):
        assert lr >= 0, "'lr' must be a positive number"
        assert alpha >= 0, "'alpha' must be a positive number"
        assert beta >= 0, "'beta' must be a positive number"
        assert lambda_coef >= 0, "'lambda_coef' must be a positive number"
        assert epsilon >= 0, "'epsilon' must be a positive number"
        assert max_norm_clip != 0, "'max_norm_clip' must not be 0"
        assert isinstance(do_bias_correction, bool), "'do_bias_correction' must be a boolean"
        assert isinstance(weight_decay_mode, AdamConfig.DecayMode), "'weight_decay_mode' must be a AdamConfig.DecayMode"
        for param in params:
            assert 'lr' not in param, "'lr' is not supported inside params"

        defaults = {'lr': lr,
                    'alpha': alpha,
                    'beta': beta,
                    'lambda': lambda_coef,
                    'epsilon': epsilon,
                    'max_norm_clip': max_norm_clip,
                    'do_bias_correction': do_bias_correction,
                    'weight_decay_mode': weight_decay_mode}
        super().__init__(name='AdamOptimizer',
                         params=params,
                         defaults=defaults)
        self.alpha = alpha
        self.beta = beta
        self.lambda_coef = lambda_coef
        self.epsilon = epsilon
        self.max_norm_clip = max_norm_clip
        self.do_bias_correction = do_bias_correction
        self.weight_decay_mode = weight_decay_mode


class LambConfig(_OptimizerConfig):
    r"""Lamb optimizer configuration

    Args:
        params (list of dict, default is []): list of parameter groups.
            Each dict must contain a 'params' key with a list of names of model's parameter that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.defaults`
            for specific model parameters.
            Empty list means all the parameters of the model will use :py:attr:`.defaults` during optimization.
        lr (float, default is 0.001): Learning rate
        alpha (float, default is 0.9): Coefficient of previous gradient in running average of row 1.
        beta (float, default is 0.999):  Coefficient of previous squared gradient in running average.
        lambda (float, default is 0): Regularization coefficient.
        ratio_min (float, default is -inf): Lower bound on confidence ratio.
        ratio_max (float, default is inf): Upper bound on confidence ratio.
        epsilon (float, default is 1e-6): Small scalar to avoid dividing by zero.
        do_bias_correction (bool, default is False): Compute unbiased 1st and 2nd momentums.

    NOTE: To prevent model parameters to be trained, refer to :py:attr:`.ORTTrainerOptions.utils.frozen_weights`.

    Example:

    .. code-block:: python

        # User-specified lr, alpha and weight_decay_mode for all model parameters
        lamb_optim1 = LambConfig(lr=0.01, alpha=0.85)

        # User-specified lr using parameters group
        lamb_optim2 = LambConfig([{'params':['fc1.weight','fc2.weight'], 'lr':0.005}], lr=0.01)
    """

    def __init__(self, params=[], lr=0.001, alpha=0.9, beta=0.999, lambda_coef=0.0,
                 ratio_min=float('-inf'), ratio_max=float('inf'), epsilon=1e-6, max_norm_clip=1.0, do_bias_correction=False):
        assert lr >= 0, "'lr' must be a positive number"
        assert alpha >= 0, "'alpha' must be a positive number"
        assert beta >= 0, "'beta' must be a positive number"
        assert lambda_coef >= 0, "'lambda_coef' must be a positive number"
        assert isinstance(ratio_min, float), "'ratio_min' must be a valid float"
        assert isinstance(ratio_max, float), "'ratio_max' must be a valid float"
        assert epsilon >= 0, "'epsilon' must be a positive number"
        assert max_norm_clip != 0, "'max_norm_clip' must not be 0"
        assert isinstance(do_bias_correction, bool), "'do_bias_correction' must be a boolean"
        for param in params:
            assert 'lr' not in param, "'lr' is not supported inside params"

        defaults = {'lr': lr,
                    'alpha': alpha,
                    'beta': beta,
                    'lambda': lambda_coef,
                    'ratio_min': ratio_min,
                    'ratio_max': ratio_max,
                    'epsilon': epsilon,
                    'max_norm_clip': max_norm_clip,
                    'do_bias_correction': do_bias_correction}
        super().__init__(name='LambOptimizer',
                         params=params,
                         defaults=defaults)
        self.alpha = alpha
        self.beta = beta
        self.lambda_coef = lambda_coef
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.epsilon = epsilon
        self.max_norm_clip = max_norm_clip
        self.do_bias_correction = do_bias_correction
