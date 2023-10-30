<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImagesHf1 from '../../images/undraw/image_HF1.svelte';
	import ImageHf2 from '../../images/undraw/image_HF2.svelte';
	import ImageHf3 from '../../images/undraw/image_HF3.svelte';
	import OnnxLight from '../../images/ONNX-Light.svelte';
	const title = 'Hugging Face + ONNX Runtime';
	const description =
		'ONNX Runtime can be used to accelerate well over 100,000 of the models available on Hugging Face.';
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
			<br>
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
	<div >
		<div>
			<h1 class="text-4xl pb-4">Supported Models</h1>
			<p>
				The top 30 most popular model families on Hugging Face are all supported by ONNX Runtime,
				and over 80 Hugging Face model families in total boast ORT support. <span
					class="hidden md:inline">The table outlines the models:</span
				>
			</p>
			<br />
			<p class="pb-4">
				ONNX models can be found directly from the Hugging Face Model Hub in its ONNX model library.
			</p>
			<a href="https://huggingface.co/models?library=onnx" class="btn btn-primary"
				>Check out Hugging Face's ONNX model library →</a
			>
			<br />
			<br />
			<p class="pb-4">
				Hugging Face also provides ONNX support for a variety of other models not listed in the ONNX
				model library. With Hugging Face Optimum, you can easily convert pretrained models to ONNX,
				and Transformers.js lets you run Hugging Face Transformers directly from your browser!
			</p>
			<a
				href="https://huggingface.co/docs/optimum/exporters/onnx/overview"
				class="btn btn-primary mr-0 mb-4 md:mr-4">Learn more about Hugging Face Optimum →</a
			>
			<a href="https://huggingface.co/docs/transformers.js/index" class="btn btn-primary"
				>Learn more about Transformers.js →</a
			>
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="flex">
		<h1 class="text-4xl pb-4">Export Hugging Face Models to ONNX</h1>
		<div class="ml-5 hidden md:flex">
			<ImagesHf1 />
		</div>
	</div>
	<h2 class="text-2xl">PyTorch Models</h2>
	<p class="pb-4">
		Hugging Face's ONNX Export Space allows users to easily export PyTorch models from the Hugging
		Face Model Hub to ONNX.
	</p>
	<a href="https://huggingface.co/spaces/onnx/export" class="btn btn-primary"
		>Try out the Hugging Face ONNX Export Space for yourself →</a
	>
	<br />
	<br />
	<h2 class="text-2xl">Other Models</h2>
	<p class="pb-4">Hugging Face provides many alternatives for exporting models to ONNX.</p>
	<a href="https://huggingface.co/docs/transformers/serialization" class="btn btn-primary"
		>Other ONNX export options →</a
	>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div >
		<div class="col-span-1">
			<div class="flex">
				<h1 class="text-4xl pb-4">Large Language Models</h1>
				<div class="ml-5 hidden md:flex">
					<ImageHf2 />
				</div>
			</div>
			<p>
				ONNX Runtime also supports many increasingly popular large language model (LLM) families,
				including most of those available in the HF Model Hub. <span class="hidden md:inline"
					>These model families are showcased in the table.</span
				>
			</p>
			<br />
			<p>
				Hugging Face also provides a leaderboard with more detailed tracking and evaluation of
				recently releases LLMs from the community.
			</p>
			<br />
			<a
				class="btn btn-primary"
				href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
				>Hugging Face Open LLM Leaderboard →</a
			>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="flex">
		<h1 class="text-4xl pb-4">Cloud Models</h1>
		<div class="ml-5 hidden md:flex">
			<ImageHf3 />
		</div>
	</div>
	<p>
		Models accelerated by ONNX Runtime can be easily deployed to the cloud through Azure ML, which
		improves time to value, streamlines MLOps, and provides built-in security.
	</p>
	<br />
	<p>
		Azure ML also publishes a curated model list that is updated regularly and includes some of the
		most popular models at the moment. Of the models on this list that are available on Hugging
		Face, there is currently Optimum ONNX support for over 85%.
	</p>
</div>
