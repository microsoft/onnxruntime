<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImagesHf1 from '../../images/undraw/image_HF1.svelte';
	import ImageHf2 from '../../images/undraw/image_HF2.svelte';
	import ImageHf3 from '../../images/undraw/image_HF3.svelte';
	import OnnxLight from '../../images/ONNX-Light.svelte';
	const title = 'Hugging Face + ONNX Runtime';
	const description =
		'ONNX Runtime can be used to accelerate well over 130,000 of the models available on Hugging Face.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
</script>

<svelte:head>
	<meta
		name="description"
		content="Learn about how ONNX Runtime works with hundreds of thousands of models on Hugging Face."
	/>
</svelte:head>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">{title}</h1>
			<br /><br />
			<p class="text-xl">
				{description}
			</p>
			<br />
			<br />
			<a
				href="https://cloudblogs.microsoft.com/opensource/2023/10/04/accelerating-over-130000-hugging-face-models-with-onnx-runtime/"
				class="btn btn-primary mr-0 mb-4 md:mr-4">Recent blog with Hugging Face →</a
			>
		</div>
		<div class="m-auto">
			<OnnxLight height={250} width={250} />
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-3xl pb-4">Export Hugging Face Models to ONNX</h1>
			<p class="pb-4">
				Hugging Face provides many options for exporting models to ONNX, including an ONNX Export
				Space for PyTorch models from the Hugging Face Model Hub.
			</p>
			<a href="https://huggingface.co/spaces/onnx/export" class="btn btn-primary"
				>Export PyTorch models to ONNX →</a
			>
			<a href="https://huggingface.co/docs/transformers/serialization" class="btn btn-primary"
				>Other ONNX export options →</a
			>
		</div>
		<div class="mx-auto md:pt-10">
			<ImagesHf1 />
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div>
		<div>
			<h1 class="text-3xl pb-4">Supported Models</h1>
			<p class="pb-4">
				The top 30 most popular model architectures on Hugging Face are all supported by ONNX
				Runtime, and over 80 Hugging Face model architectures in total boast ORT support. This list
				includes <a href="https://huggingface.co/models?other=bert" class="text-blue-500">BERT</a>,
				<a href="https://huggingface.co/models?other=gpt2" class="text-blue-500">GPT2</a>,
				<a href="https://huggingface.co/models?other=t5" class="text-blue-500">T5</a>,
				<a href="https://huggingface.co/models?other=stable-diffusion" class="text-blue-500"
					>Stable Diffusion</a
				>,
				<a href="https://huggingface.co/models?other=whisper" class="text-blue-500">Whisper</a>, and
				many more.
			</p>
			<p class="pb-4">
				ONNX models can be found directly from the Hugging Face Model Hub in its <a
					href="https://huggingface.co/models?library=onnx"
					class="text-blue-500">ONNX model library</a
				>.
			</p>
			<p class="pb-4">
				Hugging Face also provides ONNX support for a variety of other models not listed in the ONNX
				model library. With <a
					href="https://huggingface.co/docs/optimum/exporters/onnx/overview"
					class="text-blue-500">Hugging Face Optimum</a
				>, you can easily convert pretrained models to ONNX, and
				<a href="https://huggingface.co/docs/transformers.js/index" class="text-blue-500"
					>Transformers.js</a
				> lets you run Hugging Face Transformers directly from your browser!
			</p>
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-3xl pb-4">Large Language Models</h1>
			<p class="pb-4">
				ONNX Runtime also supports many increasingly popular large language model (LLM)
				architectures, including <a
					href="https://huggingface.co/models?other=llama"
					class="text-blue-500">LLaMA</a
				>,
				<a href="https://huggingface.co/models?other=gpt_neo" class="text-blue-500">GPT Neo</a>,
				<a href="https://huggingface.co/models?other=bloom" class="text-blue-500">BLOOM</a>, and
				many more.
			</p>
			<p>
				Hugging Face also provides an <a
					href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
					class="text-blue-500">Open LLM Leaderboard</a
				> with more detailed tracking and evaluation of recently releases LLMs from the community.
			</p>
		</div>
		<div class="hidden md:grid mx-auto">
			<ImageHf2 widthscale={2} heightscale={2} />
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-3xl pb-4">Cloud Models</h1>
			<p class="pb-4">
				Models accelerated by ONNX Runtime can be easily deployed to the cloud through Azure Machine
				Learning, which improves time to value, streamlines MLOps, and provides built-in security.
			</p>
			<p>
				<a href="https://ml.azure.com/" class="text-blue-500">Azure Machine Learning</a> also publishes
				a curated model list that is updated regularly and includes some of the most popular models at
				the moment. Of the models on this list that are available on Hugging Face, there is currently
				Optimum ONNX support for over 85%.
			</p>
		</div>
		<div class="hidden md:grid mx-auto">
			<ImageHf3 widthscale={2} heightscale={2} />
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div>
		<div>
			<h1 class="text-3xl pb-4">Transformers.js + ONNX Runtime Web</h1>
			<p class="pb-4">
				<a href="https://huggingface.co/docs/transformers.js/index" class="text-blue-500"
					>Transformers.js</a
				>
				is amazing tool to run transformers on the web, designed to be functionally equivalent to Hugging
				Face’s
				<a href="https://github.com/huggingface/transformers" class="text-blue-500">transformers</a>
				python library.
			</p>
			<p class="pb-4">
				Powered by ONNX Runtime Web, it enables you to execute cutting-edge Machine Learning tasks
				in areas such as Natural Language Processing, Computer Vision, Audio, and Multimodal
				directly within your web browser, eliminating the need for a server.
			</p>
		</div>
	</div>
</div>
