<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	const title = 'Generative AI + ONNX Runtime';
	const description =
		'Integrate the power of generative AI in your apps and services with ONNX Runtime. Broad platform support and deep optimizations empower usage of state-of-the-art models for image synthesis, text generation, and more.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	import stablediffusion1 from '../../images/StableDiffusion1.webp';
	import stablediffusion2 from '../../images/StableDiffusion2.webp';
</script>

<svelte:head>
	<meta
		name="description"
		content="Integrate the power of generative AI in your apps and services with ONNX Runtime."
	/>
</svelte:head>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl">Stable Diffusion + ONNX Runtime</h1>
	<p class="pb-4">Use ONNX Runtime to accelerate this popular image generation model.</p>
	<h2 class="text-3xl pb-4">Benefits</h2>
	<div class="grid gap-10 grid-cols-1 lg:grid-cols-2 pb-4">
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="card-title pb-4">Run Stable Diffusion outside of a Python environment</h2>
				<div class="card-actions">
					
					<a
						href="https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html"
						class="btn-primary btn ">Inference Stable Diffusion →</a
					>
				</div>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="card-title pb-4">
					Speed up inference of Stable Diffusion on NVIDIA and AMD GPUs
				</h2>
				<div class="card-actions">
					
					<a
						href="https://medium.com/microsoftazure/accelerating-stable-diffusion-inference-with-onnx-runtime-203bd7728540"
						class="btn btn-primary ">Accelerate Stable Diffusion →</a
					>
				</div>
			</div>
		</div>
	</div>
	<h2 class="text-3xl">Performance</h2>
	<p class="pb-4">The average latency in seconds on Stable Diffusion v1.5 and v2.1 models:</p>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 pb-10">
		<div class="">
			<img src={stablediffusion1} alt="Stable Diffusion v1.5 latency graphs" />
		</div>
		<div class="">
			<img src={stablediffusion2} alt="Stable Diffusion v2.1 latency graphs" />
		</div>
	</div>
	<h2 class="text-3xl">Large Language Models + ONNX Runtime</h2>
	<p class="pb-4">
		ONNX Runtime supports many popular large language model (LLM) families in the Hugging Face Model
		Hub. These, along with thousands of other models, are easily convertible to ONNX using the
		Optimum API.
	</p>
	<div class="">
		<a href="https://huggingface.co/models?other=llama" class="btn btn-primary">LLaMA →</a>
		<a href="https://huggingface.co/models?other=gpt_neo" class="btn btn-primary">GPT Neo →</a>
		<a href="https://huggingface.co/models?other=bloom" class="btn btn-primary">BLOOM →</a>
		<a href="https://huggingface.co/models?other=opt" class="btn btn-primary">OPT →</a>
		<a href="https://huggingface.co/models?other=gpt-j" class="btn btn-primary">GPT-J →</a>
		<a href="https://huggingface.co/models?other=flan-t5" class="btn btn-primary">FLAN-T5 →</a>
	</div>
</div>
