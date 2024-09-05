<script lang="ts">
	import Table from './table.svelte';
	import githublogo from '../../images/logos/GitHub-Logo.svg';
	import ImageTutorials from '../../images/undraw/image_tutorials.svelte';
	let description = 'Quickly ramp up with ONNX Runtime, using a variety of platforms to deploy on hardware of your choice.'
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'getting started, getting started onnx runtime, onnxruntime getting started, onnxruntime tutorial, onnxruntime tutorials, onnxruntime quick start, onnxruntime quickstart, onnxruntime quick start guide, onnxruntime quickstart guide, onnxruntime quick start tutorials, onnxruntime quickstart tutorials, onnxruntime quick start tutorial, onnxruntime quickstart tutorial, onnxruntime quick start documentation, onnxruntime quickstart documentation, onnxruntime quick start instructions, onnxruntime quickstart instructions, onnxruntime quick start guide, onnxruntime quickstart guide, onnxruntime quick start tutorials, onnxruntime quickstart tutorials, onnxruntime quick start tutorial, onnxruntime quickstart tutorial, onnxruntime quick start documentation, onnxruntime quickstart documentation, onnxruntime quick start instructions, onnxruntime quickstart instructions'
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
	<h1 class="text-4xl pb-4">Get Started</h1>
	<h1 class="text-2xl pb-4">Installation Instructions</h1>
	<div class="lg:hidden">The installation table is visible on desktop only.</div>
	<div class="hidden lg:grid">
		<Table />
	</div>
	<p class="pt-4">
		For more in-depth installation instructions, check out the <a
			href="https://onnxruntime.ai/docs/tutorials/"
			class="dark:text-blue-300 text-blue-800 underline">ONNX Runtime documentation</a
		>.
	</p>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-2 lg:grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl pb-4">GitHub</h1>
			If you are interested in joining the ONNX Runtime open source community, you might want to join
			us on GitHub where you can interact with other users and developers, participate in<a
				href="https://github.com/microsoft/onnxruntime/discussions"
				class="dark:text-blue-300 text-blue-800 underline">discussions</a
			>, and get help with any
			<a href="https://github.com/microsoft/onnxruntime/issues" class="dark:text-blue-300 text-blue-800 underline">issues</a> you
			encounter. You can also contribute to the project by reporting bugs, suggesting features, or
			submitting pull requests.
			<div class="py-4">
				<a href="https://github.com/microsoft/onnxruntime" class="btn btn-primary rounded-sm mr-4"
					>ONNX Runtime GitHub →</a
				>
			</div>
		</div>
		<img class="hidden lg:grid h-48 mx-auto" src={githublogo} alt="GitHub Logo" />
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-2 md:grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl">Tutorials</h1>
			<h1 class="text-2xl pb-2">
				Learn more about ONNX Runtime with tutorials from our documentation.
			</h1>

			<a href="./docs/tutorials/" rel="external" class="btn btn-primary rounded-sm mr-4"
				>ONNX Runtime Tutorials →</a
			>
		</div>
		<div class="col-span-1 mx-auto">
			<div class="m-10">
				<ImageTutorials widthscale={0.5} heightscale={0.5} />
			</div>
		</div>
	</div>
</div>
