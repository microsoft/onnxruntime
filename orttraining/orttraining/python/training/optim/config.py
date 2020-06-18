class _OptimizerConfig(object):
    r"""Base class for optimizer configuration

    This private class is not an optimizer, but a means to configure existing ones from ORT backend.
    Once the optimizer is configured, no user intervention is needed to update weights or zero gradients during training.
    The 'parameter group' was inspired by `Pytorch <https://pytorch.org/docs/stable/optim.html#per-parameter-options>`_.

    Args:
        name (str): optimizer names.
            One of 'SGDOptimizer', 'AdamOptimizer' and 'LambOptimizer'
        hyper_parameters (dict): optimizer hyper-parameters applied to all model parameters.
                                 Every optimizer must have a 'lr' entry on this dictionary.
        param_groups (list of dict, default is []): list of parameters groups.
            Each dict must contain a 'params' key with a list of model parameters that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.hyper_parameters` for specific model parameters

    Example:

    .. code-block:: python

        lamb_optim = _OptimizerConfig(name = 'LambOptimizer',
                                    hyper_parameters = {'lr': 0.001, 'alpha' : 0.01, 'beta' : 0.9},
                                    param_groups = [ { 'params' : ['model_param_0', 'model_param1'],
                                                       'epsilon' : 0.03, 'beta' : 0.5},
                                                     { 'params' : ['model_param_2'],
                                                       'alpha' : 0.04},
                                                   ]
                    )
    """

    def __init__(self, name, hyper_parameters, param_groups=[]):
        assert isinstance(name, str), "'name' must be a string"
        assert name in ['AdamOptimizer', 'LambOptimizer', 'SGDOptimizer'], \
            "'name' must be one of 'AdamOptimizer', 'LambOptimizer' or 'SGDOptimizer'"
        assert isinstance(hyper_parameters, dict), "'hyper_parameters' must be a dict"
        assert 'lr' in hyper_parameters, "'hyper_parameters' must contain a {'lr' : positive number} entry"
        assert hyper_parameters['lr'] >= 0, "lr must be a positive number"
        assert isinstance(param_groups, list), "'param_groups' must be a list"
        for group in param_groups:
            assert isinstance(group, dict) and len(group) > 1 and 'params' in group, \
                ("Each dict inside 'param_groups' must contain a {'params' : [model parameter names]} entry"
                 "and additional entries for custom hyper parameter values")
            for k, v in group.items():
                if k != 'params':
                    assert k in hyper_parameters, f"'param_groups' has 'k' hyper parameter not present at 'hyper_parameters'"

        self.name = name
        self.lr = hyper_parameters['lr']
        self.hyper_parameters = hyper_parameters
        self.param_groups = param_groups


class SGD(_OptimizerConfig):
    r"""SGD optimizer configuration

    NOTE: Current implementation does not support :py:attr:`param_groups`, and must be
    passed as an empty list.

    Args:
        param_groups (list of dict, default is []): list of parameters groups.
            Each dict must contain a 'params' key with a list of model parameters that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.hyper_parameters` for specific model parameters
        lr (float, default is 0.001): Learning rate
    """

    def __init__(self, param_groups=[], lr=0.001):
        super().__init__(name='SGDOptimizer', hyper_parameters={'lr':lr}, param_groups=param_groups)
        assert isinstance(param_groups, list) and len(param_groups) == 0, "'param_groups' must be an empty list"



class Adam(_OptimizerConfig):
    r"""Adam optimizer configuration

    Args:
        param_groups (list of dict, default is []): list of parameters groups.
            Each dict must contain a 'params' key with a list of model parameters that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.hyper_parameters` for specific model parameters
        lr (float, default is 0.001): Learning rate
        alpha (float, default is 0.9): Coefficient of previous gradient in running average of row 1.
        beta (float, default is 0.999):  Coefficient of previous squared gradient in running average.
        lambda (float, default is 0): Regularization coefficient.
        epsilon (float, default is 1e-8): Small scalar to avoid dividing by zero.
        do_bias_correction (bool, default is True): Compute unbiased 1st and 2nd momentums.
        weight_decay_mode (bool, default is False): Modes for applying weight decay.
            False means applying decay before weight update,
            True means applying decay after weight update.
    """

    def __init__(self, param_groups=[], lr=0.001, alpha=0.9, beta=0.999, lambda_coef=0.0, epsilon=1e-8, do_bias_correction=True, weight_decay_mode=True):
        assert lr >= 0, "'lr' must be a positive number"
        assert alpha >= 0, "'alpha' must be a positive number"
        assert beta >= 0, "'beta' must be a positive number"
        assert lambda_coef >= 0, "'lambda_coef' must be a positive number"
        assert epsilon >= 0, "'epsilon' must be a positive number"
        assert isinstance(do_bias_correction, bool), "'do_bias_correction' must be a boolean"
        assert isinstance(weight_decay_mode, bool), "'weight_decay_mode' must be a boolean"
        assert isinstance(param_groups, list) and len(param_groups) == 0, "'param_groups' must be an empty list"

        hyper_parameters = {'lr':lr,
                            'alpha' : alpha,
                            'beta' : beta,
                            'lambda_coef' : lambda_coef,
                            'epsilon' : epsilon,
                            'do_bias_correction' : do_bias_correction,
                            'weight_decay_mode' : weight_decay_mode}
        super().__init__(name='AdamOptimizer', hyper_parameters=hyper_parameters, param_groups=param_groups)


class Lamb(_OptimizerConfig):
    r"""Lamb optimizer configuration

    Args:
        param_groups (list of dict, default is []): list of parameters groups.
            Each dict must contain a 'params' key with a list of model parameters that will
            be optimized with the group's custom hyper-parameters values.
            In other words, parameter groups override the default :py:attr:`.hyper_parameters` for specific model parameters
        lr (float, default is 0.001): Learning rate
        alpha (float, default is 0.9): Coefficient of previous gradient in running average of row 1.
        beta (float, default is 0.999):  Coefficient of previous squared gradient in running average.
        lambda (float, default is 0): Regularization coefficient.
        ratio_min (float, default is -inf): Lower bound on confidence ratio.
        ratio_max (float, default is inf): Upper bound on confidence ratio.
        epsilon (float, default is 1e-6): Small scalar to avoid dividing by zero.
        do_bias_correction (bool, default is True): Compute unbiased 1st and 2nd momentums.
    """

    def __init__(self, param_groups=[], lr=0.001, alpha=0.9, beta=0.999, lambda_coef=0.0, ratio_min=float('-inf'), ratio_max=float('inf'), epsilon=1e-6, do_bias_correction=True):
        assert lr >= 0, "'lr' must be a positive number"
        assert alpha >= 0, "'alpha' must be a positive number"
        assert beta >= 0, "'beta' must be a positive number"
        assert lambda_coef >= 0, "'lambda_coef' must be a positive number"
        assert isinstance(ratio_min, float), "'ratio_min' must be a valid float"
        assert isinstance(ratio_max, float), "'ratio_max' must be a valid float"
        assert epsilon >= 0, "'epsilon' must be a positive number"
        assert isinstance(do_bias_correction, bool), "'do_bias_correction' must be a boolean"
        assert isinstance(param_groups, list) and len(param_groups) == 0, "'param_groups' must be an empty list"

        hyper_parameters = {'lr':lr,
                            'alpha' : alpha,
                            'beta' : beta,
                            'lambda_coef' : lambda_coef,
                            'ratio_min' : ratio_min,
                            'ratio_max' : ratio_max,
                            'epsilon' : epsilon,
                            'do_bias_correction' : do_bias_correction}
        super().__init__(name='LambOptimizer', hyper_parameters=hyper_parameters, param_groups=param_groups)
