<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import { fade } from 'svelte/transition';
	import coffee from './coffee.png';
	import whisper from './whisper.png';
	import vision_ui from './vision_UI.png';
	import mobile from './mobile.png';
	import desktop from './desktop.png';
	import cloud from './browser.png';
	import aibrain from './aibrain.webp';
	const title = 'Generative AI';
	const description =
		'Learn about integrating the power of generative AI in your apps and services. Use state-of-the-art models for text generation, audio synthesis, and more to create innovative experiences.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png';
	let authors = ['']; // Ensure to populate with valid author names if needed
	const keywords =
		'onnxruntime, onnx runtime generative ai, onnx runtime generative ai models, onnx runtime generative ai deployment, onnx runtime generative ai performance, onnx runtime generative ai time to market, onnx runtime generative ai production ready, onnx runtime generative ai lower latency, onnx runtime generative ai higher throughput, onnx runtime generative ai testimonials, onnx runtime generative ai performance enhancements';
	const cycleWords = ['Desktop', 'Mobile', 'Cloud'];
	let cycleIndex = 0;

	// Set interval for cycling platform display
	const interval = setInterval(() => {
		cycleIndex = (cycleIndex + 1) % cycleWords.length;
	}, 3000);

	// Cleanup interval on component destruction
	import { onDestroy } from 'svelte';
	onDestroy(() => {
		clearInterval(interval);
	});
</script>

<svelte:head>
	<!-- Meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={aibrain} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />

	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description} />
	<meta property="og:image" content={aibrain} />

	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={aibrain} />
	<meta property="twitter:card" content="summary_large_image" />
</svelte:head>

<LandingHero {title} {description} {imgsrc} {imgalt} />

<div class="container mx-auto px-10 my-10">
	<div class=" grid grid-cols-1 md:grid-cols-4">
		<div class="m-2 lg:m-8">
			<img class="rounded-xl" src={aibrain} alt="Representing generative AI" />
		</div>
		<div class="col-span-3 my-auto">
			<h2 class="text-3xl pb-4">What is Generative AI?</h2>
			<p>
				Generative AI refers to artificial intelligence that creates new content—such as text,
				images, audio, or code—based on patterns learned from existing data. Generative AI leverages
				transformer models for text and diffusion models for images. These innovations are
				transforming industries, enabling personalized experiences, automating creative processes,
				and opening new possibilities for content generation!
			</p>
		</div>
	</div>

	<div class="divider" />

	<div>
		<h2 class="text-3xl">Generative AI Models</h2>
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 my-8 gap-4">
			<!-- Model cards -->
			{#each [{ title: 'Text Generation Models', description: 'Generate human-like text for chatbots, content creation, summarization, and more.', demos: [{ name: 'Llama', url: 'https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python/README.md' }, { name: 'Mistral', url: 'https://github.com/microsoft/Olive/tree/main/examples/mistral' }, { name: 'Phi', url: 'https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-tutorial.md' }] }, { title: 'Image Generation Models', description: 'Create artwork or realistic images from descriptions using AI models like Stable Diffusion.', demos: [{ name: 'Stable Diffusion', url: 'https://github.com/microsoft/Olive/tree/main/examples/stable_diffusion' }] }, { title: 'Audio Models', description: 'Generate audio, music, or speech from data inputs with AI models like Whisper.', demos: [{ name: 'Whisper', url: 'https://huggingface.co/spaces/Xenova/whisper-web' }] }, { title: 'Other Models', description: 'Generate diverse outputs like code, video, or 3D designs.', demos: [{ name: 'Request a Model', url: 'https://github.com/microsoft/onnxruntime-genai/discussions/categories/model-support' }] }] as model}
				<div class="mx-auto flex flex-col gap-4 h-full">
					<div class="flex-1">
						<h3 class="text-2xl">{model.title}</h3>
						<p>{model.description}</p>
					</div>
					<div class="grid gap-4 grid-cols-{model.demos.length} mt-auto">
						{#each model.demos as demo}
							<a href={demo.url} class="btn btn-primary btn-block">{demo.name}</a>
						{/each}
					</div>
				</div>
			{/each}
		</div>
	</div>

	<div class="divider" />

	<div>
		<h2 class="text-3xl">ONNX Runtime ❤️ Generative AI</h2>
		<p>
			Use ONNX Runtime for high performance, scalability, and flexibility when deploying generative
			AI models. With support for diverse frameworks and hardware acceleration, ONNX Runtime ensures
			efficient, cost-effective model inference across platforms.
		</p>

		<div class="grid grid-cols-1 lg:grid-cols-2 my-8">
			<div>
				<h2 class="text-2xl text-center">Run ONNX Runtime on:</h2>
				<h2 class="text-center text-4xl">
					{#key cycleIndex}
						<strong in:fade={{ duration: 1000 }}>{cycleWords[cycleIndex]}</strong>
						<img
							class="mx-auto mt-4"
							height="512px"
							width="512px"
							in:fade={{ duration: 1000 }}
							src={cycleIndex === 0 ? desktop : cycleIndex === 1 ? mobile : cloud}
							alt="Platform for desktop, mobile, and cloud usage"
						/>
					{/key}
				</h2>
			</div>

			<!-- Feature Cards -->
			<div class="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-2 my-8">
				{#each [{ title: 'Multiplatform', description: 'Run ONNX Runtime on Desktop 🖥️, Mobile 📱, Browser 💻, or Cloud ☁️.' }, { title: 'On Device', description: 'Inference privately 🔐 and save costs ⚙️ with on-device models.' }, { title: 'Multimodal Compatibility', description: 'Use ONNX Runtime with vision or omni models. We work to quickly enable all new model scenarios 🚀.' }, { title: 'Easy to Use', description: 'Get started quickly ⏩ with our examples and tutorials.' }] as feature}
					<div class="card bg-primary dark:text-black text-neutral-content">
						<div class="card-body items-center text-center">
							<h2 class="card-title">{feature.title}</h2>
							<p class="my-auto pt-10">{feature.description}</p>
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>

	<div>
		<h3 class="text-2xl">Tutorials & Demos</h3>
		<p>Get started with any of these tutorials and demos:</p>
		<div class="grid grid-cols-1 gap-4 lg:grid-cols-3 my-8">
			<!-- Tutorial cards -->
			{#each [{ title: 'Phi-3 Vision', img: coffee, description: 'A Desktop app demo to interact with text and images simultaneously.', url: 'https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html' }, { title: 'Olive Examples', img: vision_ui, description: 'Use Olive, a hardware-aware optimizer, to quickly generate the ideal ONNX model for your needs.', url: 'https://github.com/microsoft/Olive/tree/main/examples' }, { title: 'Whisper in Browser', img: whisper, description: 'Run whisper to transcribe user audio in your browser!', url: 'https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/ort-whisper' }] as tutorial}
				<div class="card bg-base-100 image-full sm:w-80 mx-auto">
					<figure><img class="brightness-50" src={tutorial.img} alt={tutorial.title} /></figure>
					<div class="card-body items-center text-center">
						<h2 class="card-title text-white">{tutorial.title}</h2>
						<p class="text-white">{tutorial.description}</p>
						<a href={tutorial.url} class="btn btn-primary">Try it out!</a>
					</div>
				</div>
			{/each}
			<!-- More tutorials -->
		</div>
		<a
			href="https://github.com/microsoft/onnxruntime-inference-examples"
			class="btn btn-secondary btn-block mt-auto">See more demos!</a
		>
	</div>
</div>
