scenario<script lang="ts">
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
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'onnxruntime, onnx runtime hugging face, onnx runtime hugging face models, onnx runtime hugging face deployment, onnx runtime hugging face performance, onnx runtime hugging face time to market, onnx runtime hugging face deploy anywhere, onnx runtime hugging face boost performance, onnx runtime hugging face improve time to market, onnx runtime hugging face production ready, onnx runtime hugging face lower latency, onnx runtime hugging face higher throughput, onnx runtime hugging face get innovations into production faster, onnx runtime hugging face testimonials, onnx runtime hugging face performance enhancements, onnx runtime hugging face production ready, onnx runtime hugging face lower latency, onnx runtime hugging face higher throughput, onnx runtime hugging face get innovations into production faster, onnx runtime hugging face testimonials, onnx runtime hugging face performance enhancements'
</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={image} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	<meta property="og:image" content={image} />
	
	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={image} />
	<meta property="twitter:card" content={imageSquare} />
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
				class="btn btn-primary rounded-sm mr-0 mb-4 md:mr-4">Recent blog with Hugging Face →</a
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
			<a href="https://huggingface.co/spaces/onnx/export" class="btn btn-primary rounded-sm"
				>Export PyTorch models to ONNX →</a
			>
			<a
				href="https://huggingface.co/docs/transformers/serialization"
				class="btn btn-primary rounded-sm">Other ONNX export options →</a
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
				includes <a href="https://huggingface.co/models?other=bert" class="dark:text-blue-300 text-blue-800 underline">BERT</a>,
				<a href="https://huggingface.co/models?other=gpt2" class="dark:text-blue-300 text-blue-800 underline">GPT2</a>,
				<a href="https://huggingface.co/models?other=t5" class="dark:text-blue-300 text-blue-800 underline">T5</a>,
				<a href="https://huggingface.co/models?other=stable-diffusion" class="dark:text-blue-300 text-blue-800 underline"
					>Stable Diffusion</a
				>,
				<a href="https://huggingface.co/models?other=whisper" class="dark:text-blue-300 text-blue-800 underline">Whisper</a>, and
				many more.
			</p>
			<p class="pb-4">
				ONNX models can be found directly from the Hugging Face Model Hub in its <a
					href="https://huggingface.co/models?library=onnx"
					class="dark:text-blue-300 text-blue-800 underline">ONNX model library</a
				>.
			</p>
			<p class="pb-4">
				Hugging Face also provides ONNX support for a variety of other models not listed in the ONNX
				model library. With <a
					href="https://huggingface.co/docs/optimum/exporters/onnx/overview"
					class="dark:text-blue-300 text-blue-800 underline">Hugging Face Optimum</a
				>, you can easily convert pretrained models to ONNX, and
				<a href="https://huggingface.co/docs/transformers.js/index" class="dark:text-blue-300 text-blue-800 underline"
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
					class="dark:text-blue-300 text-blue-800 underline">LLaMA</a
				>,
				<a href="https://huggingface.co/models?other=gpt_neo" class="dark:text-blue-300 text-blue-800 underline">GPT Neo</a>,
				<a href="https://huggingface.co/models?other=bloom" class="dark:text-blue-300 text-blue-800 underline">BLOOM</a>, and
				many more.
			</p>
			<p>
				Hugging Face also provides an <a
					href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
					class="dark:text-blue-300 text-blue-800 underline">Open LLM Leaderboard</a
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
				Learning, which improves time-to-value, streamlines MLOps, provides built-in AI governance,
				and designs responsible AI solutions.
			</p>
			<p>
				<a href="https://ml.azure.com/" class="dark:text-blue-300 text-blue-800 underline">Azure Machine Learning</a> publishes a
				curated model list that is updated regularly and includes the most popular models. You can run
				the vast majority of the models on the curated list with ONNX Runtime, using HuggingFace Optimum.
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
				<a href="https://huggingface.co/docs/transformers.js/index" class="dark:text-blue-300 text-blue-800 underline"
					>Transformers.js</a
				>
				is an amazing tool to run transformers on the web, designed to be functionally equivalent to
				Hugging Face’s
				<a href="https://github.com/huggingface/transformers" class="dark:text-blue-300 text-blue-800 underline">transformers</a>
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
