from orttraining_test_layer_norm_transform import layer_norm_transform
from orttraining_test_model_transform import add_expand_shape, add_name, fix_transpose


def postprocess_model(model):
    add_name(model)
