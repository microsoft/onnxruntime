"""
This script runs the Stable Diffusion 3 ONNX pipeline for image generation.

For detailed setup instructions, test procedures, and verification steps, 
please refer to the QA_Test_Plan.md file included in the project.

Command-line arguments:
  --model_path: Path to the ONNX model directory. (Required)
  --prompt: The prompt for image generation.
  --height: The height of the generated image.
  --width: The width of the generated image.
  --steps: Number of inference steps.
  --num_iterations: Number of times to run the inference loop.
  --output_dir: Directory to save generated images.
  --negative_prompt: The prompt not to guide the image generation.
  --guidance_scale: Higher guidance scale encourages to generate images that are closely linked to the text prompt.
  --seed: The seed for reproducibility.
  --execution_provider: The execution provider to use for ONNX Runtime.
"""
from pathlib import Path
import onnxruntime as ort
import torch
import numpy as np
import argparse
from optimum.onnxruntime import ORTStableDiffusion3Pipeline



class OrtWrapper(ort.InferenceSession):
    def __init__(self, onnx_path,  session_options, provider, provider_options={}):

        session_options.add_session_config_entry("session.use_env_allocators", "1")
        super().__init__(onnx_path,
                         sess_options=session_options,
                         providers=[provider],
                         provider_options=[provider_options])
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.get_outputs())}


def get_transformer(model_root, batch_size, height, width, provider):
    config_path = Path(model_root) / "transformer"
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", 2*batch_size)
    session_options.add_free_dimension_override_by_name("height", height//8)
    session_options.add_free_dimension_override_by_name("width", width//8)

    
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)

def get_text_encoder(model_root, batch_size, provider):
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", batch_size)
    session_options.add_free_dimension_override_by_name("sequence_length", 77)
    config_path = Path(model_root) / "text_encoder"
    
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)

def get_vae_encoder(model_root, batch_size, height, width, provider):
    config_path = Path(model_root) / "vae_encoder"
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", batch_size)
    session_options.add_free_dimension_override_by_name("sample_height", height)
    session_options.add_free_dimension_override_by_name("sample_width", width)
    
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)

def get_vae_decoder(model_root, batch_size, height, width, provider):
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", batch_size)
    session_options.add_free_dimension_override_by_name("latent_height", height//8)
    session_options.add_free_dimension_override_by_name("latent_width", width//8)
    config_path = Path(model_root) / "vae_decoder"
    
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)

def get_text_encoder_2(model_root, batch_size, provider):
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", batch_size)
    session_options.add_free_dimension_override_by_name("sequence_length", 77)
    config_path = Path(model_root) / "text_encoder_2"
    
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)

def get_text_encoder_3(model_root, batch_size, provider):
    session_options=ort.SessionOptions()
    session_options.add_free_dimension_override_by_name("batch_size", batch_size)
    session_options.add_free_dimension_override_by_name("sequence_length", 77)
    config_path = Path(model_root) / "text_encoder_3"
   
    return OrtWrapper(
        onnx_path=str(config_path / "model.onnx"),
        session_options=session_options,
        provider=provider)



def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion 3 ONNX pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model directory.")
    parser.add_argument("--prompt", type=str, default="close up view of colorful chameleon", help="The prompt for image generation.")
    parser.add_argument("--height", type=int, default=512, help="The height of the generated image.")
    parser.add_argument("--width", type=int, default=512, help="The width of the generated image.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of times to run the inference loop.")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save generated images.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="The prompt not to guide the image generation.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.")
    parser.add_argument("--seed", type=int, default=None, help="The seed for reproducibility.")
    parser.add_argument("--execution_provider", type=str, default="NvTensorRTRTXExecutionProvider", help="The execution provider to use for ONNX Runtime.")
    
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Loading models...")
    vae_decoder = get_vae_decoder(args.model_path, 1, args.height, args.width, args.execution_provider)
    transformer = get_transformer(args.model_path, 1, args.height, args.width, args.execution_provider)
    vae_encoder = get_vae_encoder(args.model_path, 1, args.height, args.width, args.execution_provider)
    text_encoder = get_text_encoder(args.model_path, 1, args.execution_provider)
    text_encoder_2 = get_text_encoder_2(args.model_path, 1, args.execution_provider)
    text_encoder_3 = get_text_encoder_3(args.model_path, 1, args.execution_provider)
    print("Models loaded.")


    print("Creating pipeline...")
    pipeline = ORTStableDiffusion3Pipeline.from_pretrained(
                    args.model_path,
                    use_io_binding=True,  # Not supported by Optimum version 1.17.1 at the time of verification.
                    transformer_session=transformer,
                    text_encoder_session=text_encoder,
                    text_encoder_2_session=text_encoder_2,
                    text_encoder_3_session=text_encoder_3,
                    vae_encoder_session=vae_encoder,
                    vae_decoder_session=vae_decoder,
                )
    print("Pipeline created.")


    print("Warmup iteration...")
    images = pipeline(
                prompt=[args.prompt]*1,
                height=args.height,
                width=args.width,
                num_inference_steps=10,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                max_sequence_length=77
            ).images
    print("Warmup finished.")



    inference_times = []
    all_images = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(f"Running inference for {args.num_iterations} iterations...")
    for i in range(args.num_iterations):
        print(f"Iteration {i+1}/{args.num_iterations}")
        start_event.record(stream=torch.cuda.default_stream())
        result = pipeline(
            prompt=[args.prompt]*1,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            max_sequence_length=77
        )
        end_event.record(stream=torch.cuda.default_stream())
        end_event.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        inference_times.append(start_event.elapsed_time(end_event))
        all_images.extend(result.images)
    print("Inference finished.")

    # Save generated images to disk
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Saving images to {output_dir}...")

    for idx, img in enumerate(all_images):
        img_path = output_dir / f"generated_image_{idx+1}.png"
        img.save(img_path)
        print(f"Saved image {idx+1} to {img_path}")
    print("Images saved.")

    total_time = sum(inference_times)
    print("\n--- Performance ---")
    print(f"Total pipeline execution for {args.num_iterations} inferences took {total_time:.2f} ms")
    if args.num_iterations > 0:
        print(f"Average time per inference: {total_time / args.num_iterations:.2f} ms")
        print(f"Median time per inference: {np.median(inference_times):.2f} ms")


if __name__ == "__main__":
    main()