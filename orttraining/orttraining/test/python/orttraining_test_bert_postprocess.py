# TODO: postprocess transforms are to be moved to either pytorch exporter or onnxruntime front end (ort_trainer.py)
from orttraining_test_model_transform import add_name, process_concat, handle_expand_input_is_not_constant_case, fix_expand, fix_dim, fix_transpose, process_dropout, add_expand_shape
from orttraining_test_layer_norm_transform import layer_norm_transform

def postprocess_model(model):

    add_name(model)
    # #replace garther&concat to reshape
    # process_concat(model)

    # # will be longer needed after Range is supported in ORT.
    # handle_expand_input_is_not_constant_case(model)
    
    # # fix the expand with dynamic shape
    # # will be longer needed after Range is supported in ORT.
    # fix_expand(model)

    # #use dynamic batch/sequence
    # fix_dim(model)

    #constant fold transpose
    fix_transpose(model)

    # #replace dropout with trainable dropout
    # process_dropout(model)
    
    #add output shape of expand
    # will be longer needed after Range is supported in ORT.
    add_expand_shape(model)
    #set opset version to 10
    #model.opset_import[0].version = 10


    ################
    # layer_norm
    ################
    layer_norm_transform(model)
