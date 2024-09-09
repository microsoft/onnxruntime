<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import { fade } from 'svelte/transition';
	import coffee from './coffee.png';
	import genny from './genny.png';
	import vision_ui from './vision_UI.png';
	import mobile from './mobile.png';
	import desktop from './desktop.png';
	import cloud from './browser.png';
	import aibrain from './aibrain.webp'
	const title = 'Generative AI';
	const description =
		'Integrate the power of generative AI in your apps and services with ONNX Runtime. Broad platform support and deep optimizations empower usage of state-of-the-art models for text generation, audio synthesis, and more.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png';
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png';
	let authors = [''];
	let keywords =
		'onnxruntime, onnx runtime generative ai, onnx runtime generative ai models, onnx runtime generative ai deployment, onnx runtime generative ai performance, onnx runtime generative ai time to market, onnx runtime generative ai deploy anywhere, onnx runtime generative ai boost performance, onnx runtime generative ai improve time to market, onnx runtime generative ai production ready, onnx runtime generative ai lower latency, onnx runtime generative ai higher throughput, onnx runtime generative ai get innovations into production faster, onnx runtime generative ai testimonials, onnx runtime generative ai performance enhancements, onnx runtime generative ai production ready, onnx runtime generative ai lower latency, onnx runtime generative ai higher throughput, onnx runtime generative ai get innovations into production faster, onnx runtime generative ai testimonials, onnx runtime generative ai performance enhancements';
	let cycleWords = ['Desktop', 'Mobile', 'Cloud'];
	let cycleIndex = 0;
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
	<!-- Dynamic meta tags -->
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
	<meta property="twitter:card" content={aibrain} />
</svelte:head>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="my-8 grid grid-cols-1 md:grid-cols-4 gap-4">
		<img class="m-auto rounded-xl" src={aibrain} alt="Representing generative AI">
		<div class="col-span-3 my-auto">

			<h2 class="text-3xl">What is Generative AI?</h2>
			<p>
Generative AI refers to a type of artificial intelligence that creates new content—such as text, images, audio, or code— in response to a user prompt or instruction. 
				text, images, audio, or code—based on patterns learned from existing data. Unlike
				traditional AI models, which primarily classify or predict based on given inputs, generative
				AI models produce entirely new outputs.
				<br /><br />
				They accomplish this through advanced techniques like deep learning, often using models such
				as Transformers and Generative Adversarial Networks (GANs). Examples include AI that generates
				human-like text responses, creates realistic images from descriptions, or composes music. Generative
				AI is driving innovation across industries by enabling personalized experiences, automating creative
				processes, and opening new possibilities for content generation!
			</p>
		</div>
		
	</div>
	<div class="divider" />
	<div class="">
		<div class="">
			<h2 class="text-3xl">Generative AI models</h2>
			
		</div>
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 my-8 gap-4">
			<div class="flex flex-col h-full gap-4">
				<div class="flex-1">
					<h3 class="text-2xl">Text Generation Models</h3>
					<p>
						Text generation models are AI systems designed to generate human-like text based on
						prompts. They're used in chatbots, content creation, summarization, and creative
						writing. Check out our phi-3 and mistral demos below:
					</p>
				</div>
			
				<div class="grid gap-4 grid-cols-1 md:grid-cols-2 mt-auto">
					<a href="https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-tutorial.md" class="btn btn-primary btn-block">Phi-3</a>
					<a href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/models/mistral" class="btn btn-primary btn-block">Mistral</a>
				</div>
			</div>
			
			<div class="mx-auto flex flex-col gap-4 h-full">
				<div class="flex-1">
					<h3 class="text-2xl">Image Generation Models</h3>
					<p>
						Image generation models use AI to create visuals from text descriptions or other inputs.
						These models generate artwork, realistic images, and visual content for design and media
						applications. Check out Stable Diffusion for a great example!
					</p>
				</div>
				<div class="grid gap-4 grid-cols-1 mt-auto">
					<a href="https://github.com/microsoft/onnxruntime-inference-examples/tree/77989cff19f102300e3c4f99b957b55f74daecb4/js/sd-turbo" class="btn btn-primary btn-block">Stable Diffusion</a>
				</div>
			</div>
			
			<div class="mx-auto flex flex-col gap-4 h-full">
				<div class="flex-1">
					<h3 class="text-2xl">Audio Models</h3>
					<p>
						Many audio models transcribe text from audio, but audio generation models exist too, using AI to create sound, music, or speech based on input data.
						Check out Whisper and its audio recognition capabilities:
					</p>
				</div>
				<div class="grid gap-4 grid-cols-1 mt-auto">
					<a href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/ort-whisper" class="btn btn-primary btn-block">Whisper</a>
				</div>
			</div>
			
			<div class="mx-auto flex flex-col gap-4 h-full">
				<div class="flex-1">
					<h3 class="text-2xl">Other Models</h3>
					<p>
						Other generative models create diverse outputs like code, video, or 3D designs. These
						models expand creative possibilities, enabling automation and innovation in fields
						ranging from software development to digital art. 
				</div>
				<div class="grid gap-4 grid-cols-1 mt-auto">
					<a href="https://github.com/microsoft/onnxruntime-genai" class="btn btn-primary btn-block">Request a model on GitHub</a>
				</div>
			</div>
			
		</div>
	</div>
	<div class="divider" />
	<div class="">
		<h2 class="text-3xl">ONNX Runtime ❤️ Generative AI</h2>
		<p>
			Use ONNX Runtime for its high performance, scalability, and flexibility in deploying
			generative AI models. With support for diverse frameworks and hardware acceleration, it
			ensures efficient, cost-effective model inference across various environments.
		</p>
		<div class="grid grid-cols-1 lg:grid-cols-2 my-8">
			<div class="">
				<h2 class="text-2xl text-center">
					Run ONNX Runtime on:
				</h2>
				<h2 class="text-center text-4xl">
					{#key cycleIndex}
						<strong in:fade={{ duration: 1000 }}>{cycleWords[cycleIndex]}</strong>
						<img
							height="512px"
							width="512px"
							class="mx-auto mt-4"
							in:fade={{ duration: 1000 }}
							src={cycleIndex == 0 ? desktop : cycleIndex == 1 ? mobile : cloud}
							alt="Desktop, mobile, and cloud usage of ONNX Runtime Gen AI"
						/>
					{/key}
				</h2>
			</div>
			<!-- <div class="my-auto">
				<a href="" class="btn btn-primary btn-block m-4">Run on Desktop</a>
				<a href="" class="btn btn-primary btn-block m-4">Run on Mobile</a>
				<a href="" class="btn btn-primary btn-block m-4">Run in Browser</a>
			</div> -->
			<div class="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-2 my-8">
				<div class="card bg-base-300 text-neutral-content">
					<div class="card-body items-center text-center">
						<h2 class="card-title">Multiplatform</h2>
						<p>Whether it be Desktop, Mobile, Browser, or the Cloud, run ONNX Runtime on the platform of your choosing!</p>
					</div>
				</div>
				<div class="card bg-base-300 text-neutral-content">
					<div class="card-body items-center text-center">
						<h2 class="card-title">On Device</h2>
						<p>Run ONNX Runtime on device and inference privately and with significant cost savings. When you consider hybrid scenarios, the possibilities are endless!</p>
					</div>
				</div>
				<div class="card bg-base-300 text-neutral-content">
					<div class="card-body items-center text-center">
						<h2 class="card-title">Multimodal compatibility</h2>
						<p>You aren't limited to just LLMs with ONNX Runtime - you can use your favourite vision or omni models too.</p>
					</div>
				</div>
				<div class="card bg-base-300 text-neutral-content">
					<div class="card-body items-center text-center">
						<h2 class="card-title">Easy to use!</h2>
						<p>Getting ramped up is super easy. Get started using any of the various examples we have on this page!</p>
					</div>
				</div>
			</div>
		</div>
		
	</div>
	<div class="">
		<h3 class="text-2xl">Tutorials & Demos</h3>
		<p>
			Raring to go? Bring your models to all platforms and get started with any of the following
			tutorials and demos:
		</p>
		<div class="">
			<div class="grid grid-cols-1 gap-4 lg:grid-cols-3 my-8 ">
				<div class="card bg-base-100 image-full sm:w-80 mx-auto">
					<figure>
						<img class="brightness-50" src={coffee} alt="A cappuccino used to inference with Phi3 vision" />
					</figure>
					<div class="card-body items-center text-center">
						<h2 class="card-title text-white">Phi-3 Vision</h2>
						<p class="text-white">
							A Desktop app demo to interact with text and images simultaneously.
						</p>
						<div class="card-actions justify-end">
							<a
								href="https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html"
								class="btn btn-primary">Try it out!</a
							>
						</div>
					</div>
				</div>
				<div class="card bg-base-100 image-full sm:w-80 mx-auto">
					<figure>
						<img class="brightness-50" src={vision_ui} alt="UI of the vision LLM chat app" />
					</figure>
					<div class="card-body items-center text-center">
						<h2 class="card-title text-white">LLM Chat App</h2>
						<p class="text-white">
							An LLM chat app with UI. Pick your favourite model and get chatting!
						</p>
						<div class="card-actions justify-end">
							<a
								href="https://github.com/microsoft/onnxruntime-genai/tree/main/examples/chat_app"
								class="btn btn-primary">Try it out!</a
							>
						</div>
					</div>
				</div>
				<div class="card bg-base-100 image-full sm:w-80 mx-auto">
					<figure>
						<img class="brightness-50" src={genny} alt="Genny's model testing UI" />
					</figure>
					<div class="card-body items-center text-center">
						<h2 class="card-title text-white">Genny (model tester)</h2>
						<p class="text-white">A demo app to debug and test models with ORT GenAI.</p>
						<div class="card-actions justify-end">
							<a
								href="https://github.com/microsoft/onnxruntime-genai/tree/main/examples/csharp/Genny"
								class="btn btn-primary">Try it out!</a
							>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
