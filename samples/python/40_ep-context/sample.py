import argparse
import os
import time
import onnxruntime as ort

# Set logger severity to warning level to reduce console output.
ort.set_default_logger_severity(3)

# Default Execution Provider for NVIDIA GPUs as requested.
TRT_RTX_EP = "NvTensorRTRTXExecutionProvider"

def compile(input_path, output_path, provider, embed_mode=False):
    """
    Compiles an ONNX model for a specified execution provider and saves it.
    
    Args:
        input_path (str): Path to the original ONNX model.
        output_path (str): Path to save the compiled model.
        provider (str): The name of the execution provider.
        embed_mode (bool): If True, embeds the compiled binary data into the ONNX file.
    """
    # Remove the output file if it already exists to ensure a clean compilation.
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"> Previous compiled model at {output_path} removed.")

    # Create session options and add the provider.
    session_options = ort.SessionOptions()
    session_options.add_provider(provider, {})

    # Create a ModelCompiler instance using positional arguments.
    model_compiler = ort.ModelCompiler(
        session_options, 
        input_path,
        embed_compiled_data_into_model=embed_mode
    )

    print(f"\n> Compiling model with '{provider}'...")
    start = time.perf_counter()
    # Execute the compilation process.
    model_compiler.compile_to_file(output_path)
    stop = time.perf_counter()

    if os.path.exists(output_path):
        print("> Compiled successfully!")
        print(f"> Compile time: {stop - start:.3f} sec")
        print(f"> Compiled model saved at {output_path}")

def load_session(model_path, provider):
    """
    Loads an ONNX model into an InferenceSession and measures the loading time.
    
    Args:
        model_path (str): Path to the ONNX model file.
        provider (str): The name of the execution provider.
    """
    # Create the list of providers with an empty dictionary for options.
    providers = [(provider, {})]
    
    start = time.perf_counter()
    # Load the model using the specified provider.
    session = ort.InferenceSession(model_path, providers=providers)
    stop = time.perf_counter()
    
    print(f"> Session load time: {stop - start:.3f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile ONNX model with ONNX Runtime")
    parser.add_argument("-i", "--model_path", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the compiled EP context model")
    parser.add_argument("-p", "--provider", default=TRT_RTX_EP, type=str, help="Execution Provider")
    # Using a type=bool for the embed flag.
    parser.add_argument("-e", "--embed", default=False, type=bool, help="Binary data embedded within EP context node")
    args = parser.parse_args()

    # If the output path is not provided, generate a default one.
    if not args.output_path:
        parent_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(parent_dir, "model_ctx.onnx")

    print("-----------------------------------------------")
    print("ONNX Runtime Model Compilation Script")
    print("-----------------------------------------------")
    print(f"> Using Execution Provider: {args.provider}")
    print(f"> Embed Mode: {'Embedded' if args.embed else 'External'}")
    print("-----------------------------------------------")
    print("Available execution provider(s):", ort.get_available_providers())

    # Load and time the original model.
    print("\n> Loading regular onnx...")
    load_session(args.model_path, args.provider)

    # Compile the model.
    compile(args.model_path, args.output_path, args.provider, args.embed)

    # Load and time the compiled model.
    print("\n> Loading EP context model...")
    load_session(args.output_path, args.provider)

    print("\nProgram finished successfully.")
