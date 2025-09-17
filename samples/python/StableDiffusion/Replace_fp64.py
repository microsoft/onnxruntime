import onnx
from onnx import numpy_helper
import numpy as np
import argparse
import os
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_fp64_to_fp32(model_path: str, output_path: str):
    """
    Loads an ONNX model, converts all float64 tensors and casts to float32,
    and saves the modified model.
    """
    logging.info(f"Loading model from: {model_path}")
    model = onnx.load(model_path)
    
    # 1. Convert all initializers from float64 to float32
    converted_initializers = 0
    new_initializers = []
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.DOUBLE:
            initializer_np = numpy_helper.to_array(initializer)
            initializer_fp32 = initializer_np.astype(np.float32)
            new_initializer = numpy_helper.from_array(initializer_fp32, name=initializer.name)
            new_initializers.append(new_initializer)
            converted_initializers += 1
        else:
            new_initializers.append(initializer)

    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_initializers)
    
    if converted_initializers > 0:
        logging.info(f"Converted {converted_initializers} initializers from FP64 to FP32.")

    # 2. Convert nodes
    converted_casts = 0
    converted_constants = 0
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value' and attr.t.data_type == onnx.TensorProto.DOUBLE:
                    attr.t.data_type = onnx.TensorProto.FLOAT
                    fp64_array = np.frombuffer(attr.t.raw_data, dtype=np.float64)
                    fp32_array = fp64_array.astype(np.float32)
                    attr.t.raw_data = fp32_array.tobytes()
                    converted_constants += 1
        elif node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == onnx.TensorProto.DOUBLE:
                    attr.i = onnx.TensorProto.FLOAT
                    converted_casts += 1
    
    if converted_casts > 0:
        logging.info(f"Modified {converted_casts} Cast operators from FP64 to FP32.")
    if converted_constants > 0:
        logging.info(f"Modified {converted_constants} Constant operators from FP64 to FP32.")
        
    # 3. Convert all graph inputs, outputs, and value_info from float64 to float32
    converted_tensors = 0
    for tensor in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if tensor.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            converted_tensors += 1
    
    if converted_tensors > 0:
        logging.info(f"Converted {converted_tensors} tensor definitions from FP64 to FP32.")
    
    # 4. Save the modified model
    logging.info(f"Saving modified model to: {output_path}")
    onnx.save(model, output_path, save_as_external_data=True)
    logging.info("Conversion complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert ONNX models in a directory from float64 to float32 precision."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing the input ONNX models.")
    parser.add_argument("output_dir", type=str, help="Directory where the converted models will be saved.")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for root, _, files in os.walk(input_dir):
        # Replicate directory structure in the output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for filename in files:
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_subdir, filename)

            if filename.endswith(".onnx"): 
                logging.info("-" * 50)
                logging.info(f"Processing ONNX file: {input_path}")
                try:
                    convert_fp64_to_fp32(input_path, output_path)
                except Exception as e:
                    logging.error(f"Failed to convert {input_path}: {e}")
                logging.info("-" * 50)
            elif filename.endswith(".onnx_data"):
                # Skip copying .onnx_data files as new ones will be created on save
                continue
            else:
                logging.info(f"Copying file: {input_path} to {output_path}")
                shutil.copy2(input_path, output_path)