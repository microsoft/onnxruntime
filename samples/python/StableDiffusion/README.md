# Stable Diffusion 3 Medium ONNX Export Guide

This guide provides the steps to convert the `stabilityai/stable-diffusion-3-medium` model to the ONNX format for use with the CUDA execution provider. It also includes a step to address an issue with mixed-precision nodes that may occur during the conversion process.

## 1. Prerequisites and Installation

Install the required Python packages using the following `requirements.txt` content:

```
numpy
torch --index-url https://download.pytorch.org/whl/cu121
optimum[onnxruntime]
onnxruntime-gpu
diffusers
sentencepiece
transformers
```

You can save this to a `requirements.txt` file and install it with:
```bash
pip install -r requirements.txt
```
This will install `onnxruntime-gpu` with the CUDA execution provider, which is necessary for model conversion.

## 2. Model Conversion

Run the following command to export the model to ONNX format. This command uses `optimum-cli` to convert the model to half-precision (`fp16`) on a CUDA device.

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-3-medium --dtype fp16 --device cuda fp16_optimum
```

This will download the model and convert it into multiple ONNX files in the `fp16_optimum` directory.

## 3. Correcting FP64 Nodes

The PyTorch model may contain some `fp64` nodes, which are exported as-is during the conversion. If you encounter issues with these nodes, you can use the provided `Replace_fp64.py` script to replace them with `fp32` nodes. This script will process all `.onnx` files in the input directory and save the corrected files to the output directory.

```bash
python Replace_fp64.py fp16_optimum corrected_model
```
This will create a `corrected_model` directory with the FP64 nodes converted to FP32.

## 4. Using a Custom ONNX Runtime

If you have a locally built ONNX Runtime wheel with specific optimizations (e.g., for NvTensorRTRTXExecutionProvider), ensure that you install it in your environment before running inference. Additionally, be sure to uninstall the default `onnxruntime` package installed via `requirements.txt` to avoid any conflicts.

## 5. Running Inference

To run inference with the converted ONNX model, use the provided `RunSd.py` script. This script loads the ONNX model and generates an image based on a prompt.

Here is an example command to run the script:
```bash
python RunSd.py --model_path corrected_model --prompt "A beautiful landscape painting of a waterfall in a lush forest" --output_dir generated_images
```

### Command-line Arguments

The `RunSd.py` script accepts several arguments to customize the image generation process:

*   `--model_path`: Path to the directory containing the ONNX models (e.g., `corrected_model`). (Required)
*   `--prompt`: The text prompt to generate the image from.
*   `--negative_prompt`: The prompt not to guide the image generation.
*   `--height`: The height of the generated image (default: 512).
*   `--width`: The width of the generated image (default: 512).
*   `--steps`: The number of inference steps (default: 50).
*   `--guidance_scale`: Guidance scale for the prompt (default: 7.5).
*   `--seed`: A seed for reproducibility.
*   `--output_dir`: The directory to save the generated images (default: `generated_images`).
*   `--execution_provider`: The ONNX Runtime execution provider to use (default: `NvTensorRTRTXExecutionProvider`).

For a full list of arguments, you can run:
```bash
python RunSd.py --help
```

The generated image will be saved in the specified output directory. 
