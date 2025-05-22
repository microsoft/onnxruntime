import argparse
import onnx
import os
import sys
import subprocess
import importlib
import inspect
from typing import List, Callable, Dict
import logging

def setup_logging(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    log_filename = f"logs/transform_{model_name}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

from transforms import (
    transform_add_intermediate_tensors_to_outputs,
    all_tensors_are_4d,
    execute_shape_inference
)

def get_available_transforms() -> Dict[str, Callable]:
    """Dynamically load transform functions from transforms.py"""
    try:
        # Import the transforms module
        transforms = importlib.import_module('transforms')
        
        # Find all functions starting with 'transform_'
        transform_functions = {}
        for name, func in inspect.getmembers(transforms, inspect.isfunction):
            if name.startswith('transform_'):
                transform_functions[name] = func
                
        return transform_functions
    except ImportError:
        print("Error: transforms.py module not found. Make sure it exists in the same directory.")
        return {}
    except Exception as e:
        print(f"Error loading transforms: {str(e)}")
        return {}

def apply_transforms(model, transform_sequence: List[str], transform_functions: Dict[str, Callable], options={}) -> onnx.ModelProto:
    """Apply a sequence of transforms to the model"""
    for transform_name in transform_sequence:
        print(f"Applying transform: {transform_name}")
        
        # Check if the transform exists in our function map
        if transform_name not in transform_functions:
            print(f"Warning: Transform '{transform_name}' not found in transforms.py - skipping")
            continue
        
        # Get the function
        transform_func = transform_functions[transform_name]
        
        # Check function signature and apply accordingly
        sig = inspect.signature(transform_func)
        param_names = list(sig.parameters.keys())
        
        if len(param_names) == 1:
            # onnxscript transform functions returns transformed model
            if transform_name in ["transform_reshape_reducesum","transform_reshape_clip_reducesum"]:
                model = transform_func(model)
            else:
                transform_func(model)
        elif len(param_names) > 1:
            # Handle special cases
            if transform_name == "transform_remove_qdq" and "keep_clip_after_inputs" in options:
                # Pass the keep_clip_after_inputs flag
                keep_clip_after_inputs = options.get("keep_clip_after_inputs", False)
                transform_func(model, keep_clip_after_inputs)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform ONNX model for 4D operations')
    parser.add_argument('--original_model', type=str, help='Path to the input ONNX model')
    parser.add_argument('--transformed_model', type=str, help='Path to save the transformed ONNX model')
    parser.add_argument('--replace_qdq_with_clip', action='store_true', 
                        help='Whether to replace QDQ nodes with Clip nodes')
    parser.add_argument('--keep_clip_after_inputs', action='store_true', 
                        help='Whether to replace QDQ nodes after inputs with Clip nodes (this will effectly limits input range). This option is only valid when transform_remove_qdq is in transform_sequence')
    parser.add_argument('--debug_outputs', action='store_true', help='Add intermediate tensors as outputs for debugging')
    parser.add_argument('--transform_sequence', nargs='+', 
                        help='Sequence of transforms to apply, excluding qdq transform, which is controlled by --replace_qdq_with_clip flag')
    parser.add_argument('--save_logging', action='store_true', help='Save log into logs directory')

    args = parser.parse_args()
    original_model = args.original_model
    transformed_model = args.transformed_model
    transform_sequence = args.transform_sequence
    debug_outputs = args.debug_outputs
    if args.save_logging:
        setup_logging(original_model)
    
    options = {
        "keep_clip_after_inputs": args.keep_clip_after_inputs
    }
    
    if args.replace_qdq_with_clip:
        transform_sequence.insert(0, "transform_qdq_to_clip")
        transformed_model = transformed_model.replace('.onnx', '_clipped.onnx')
        # replace transform_reshape_reducesum with transform_reshape_clip_reducesum
        if "transform_reshape_reducesum" in transform_sequence:
            transform_sequence[transform_sequence.index("transform_reshape_reducesum")] = "transform_reshape_clip_reducesum"
    else:
        transform_sequence.insert(0, "transform_remove_qdq")

    original_model_file_name = os.path.basename(original_model)
    transform_functions = get_available_transforms()
    
    # Some models needs shape inference before transforming (unknown output shape error)
    shape_inferenced_original_model = original_model_file_name.replace('.onnx', '_shape_inferenced.onnx')
    # execute_shape_inference(original_model, shape_inferenced_original_model)

    if all_tensors_are_4d(shape_inferenced_original_model):
        print("All tensors are 4D")
        sys.exit(0)
    
    if args.debug_outputs:
        intermediate_tensor_to_add =[]
        debug_original_model = onnx.load(shape_inferenced_original_model)
        debug_tranformed_model_name = original_model.replace('.onnx', '_debug.onnx')
        transform_add_intermediate_tensors_to_outputs(debug_original_model, intermediate_tensor_to_add)
        onnx.save(debug_original_model, debug_tranformed_model_name)

    model = onnx.load(shape_inferenced_original_model)

    model = apply_transforms(model, transform_sequence, transform_functions, options)

    onnx.save(model, transformed_model)

    shape_inferenced_transformed_model = transformed_model.replace('.onnx', '_shape_inferenced.onnx')
    execute_shape_inference(transformed_model, shape_inferenced_transformed_model)

    if debug_outputs:
        assert(args.run_shape_inference_after_transform, "Debug outputs can only be added to shape inferred model")
        model = onnx.load(shape_inferenced_transformed_model)
        debug_tranformed_model = transformed_model.replace('.onnx', '_debug.onnx')
        transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add)
        onnx.save(model, debug_tranformed_model)

    sys.exit(0)