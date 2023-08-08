from orttraining_test_layer_norm_transform import layer_norm_transform  # noqa: F401
from orttraining_test_model_transform import add_expand_shape, add_name, fix_transpose  # noqa: F401


def postprocess_model(model):
    add_name(model)
