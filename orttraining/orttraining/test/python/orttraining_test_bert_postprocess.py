from orttraining_test_model_transform import add_name, fix_transpose, add_expand_shape
from orttraining_test_layer_norm_transform import layer_norm_transform

def postprocess_model(model):
    add_name(model)
