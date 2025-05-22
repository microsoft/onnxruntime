import os
import subprocess
import sys
import argparse
import re
import json

def run_command(command):
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    output_lines = stdout.decode('utf-8').strip().split('\n') + stderr.decode('utf-8').strip().split('\n')
    
    sys.stdout.flush()
    return process.returncode, "\n".join(output_lines)

def run_model(model_name, model_config):
    """Run transforms for a single model based on its configuration"""
    orig_model_path = model_config["original_model_path"]
    
    # Validate input file exists
    if not os.path.exists(orig_model_path):
        return (False, f"Error: Original model file not found: {orig_model_path}")
    
    base_transformed_path = model_config["transformed_model_path"]

    keep_clip_flag = "--keep_clip_after_inputs" if model_config.get("keep_clip_after_inputs", False) else ""
    
    transform_cmd_noclip = (
        f"python transform_model.py "
        f"--original_model {orig_model_path} "
        f"--transformed_model {base_transformed_path} "
        f"--transform_sequence {' '.join(model_config['transform_sequence'])} "
        f"{keep_clip_flag}"
    )
    result, msg = run_command(transform_cmd_noclip)
    # print(f"Result: {result}, Msg: {msg}")
    if result != 0:
        return (False, f"Error: Failed to run {transform_cmd_noclip} - {msg}")

    transform_cmd_clip = (
        f"python transform_model.py "
        f"--original_model {orig_model_path} "
        f"--transformed_model {base_transformed_path} "
        f"--transform_sequence {' '.join(model_config['transform_sequence'])} "
        f"--replace_qdq_with_clip"
    )
    result, msg = run_command(transform_cmd_clip)
    # print(f"Result: {result}, Msg: {msg}")
    if result != 0:
        return (False, f"Error: Failed to run {transform_cmd_clip} - {msg}")
    
    '''
    compare_output_cmd = (
        f"python compare_model_outputs.py "
        f"--config_file {args.config_file} "
        f"--model_name {model_name}"
    )
    result, msg = run_command(compare_output_cmd)
    print(f"Result: {result}, Msg: {msg}")
    if result != 0:
        return (False, f"Error: Failed to run {compare_output_cmd} - {msg}")
    '''

    return (True, msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Transformation Pipeline")
    parser.add_argument("--config_file", type=str, required=True,
                      help="Path to the model configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        sys.exit(1)

    results = {}
    for model_name, model_config in config.items():
        print(f"\nProcessing model: {model_name}")
        output = run_model(model_name, model_config)
        results[model_name] = output
    
    # Print summary
    print("\n\n====== TRANSFORM SUMMARY ======")
    for model, (success, msg) in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{model}: {status}")
        print(f"{msg}")

    all_success = all(success for success, _ in results.values())
    if all_success:
        print("\nAll models processed successfully!")
    else:
        print("\nSome models had issues during processing.")
