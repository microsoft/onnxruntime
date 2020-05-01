from orttraining_test_model_transform import add_name, fix_transpose, add_expand_shape
from orttraining_test_layer_norm_transform import layer_norm_transform

def postprocess_model(model):
    add_name(model)

    # remove transpose node if its input is a 2d weight which only feeds to the node
    fix_transpose(model)

    add_expand_shape(model)
    
    layer_norm_transform(model)
