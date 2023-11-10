<script lang="ts">
	import Table from './table.svelte';
	import githublogo from '../../images/logos/GitHub-Logo.svg';
	import ImageTutorials from '../../images/undraw/image_tutorials.svelte';
</script>

<svelte:head>
	<meta
		name="description"
		content="Quickly ramp up with ONNX Runtime, using a variety of platforms to deploy on hardware of your choice."
	/>
</svelte:head>

<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Get Started</h1>
	<h1 class="text-2xl pb-4">Installation Instructions</h1>
	<div class="lg:hidden">
		The installation table is visible on desktop only.
	</div>
	<div class="hidden lg:grid">
		<Table />
	</div>
	<p class="pt-4">
		For more in-depth installation instructions, check out the <a
			href="https://onnxruntime.ai/docs/tutorials/"
			class="text-blue-500">ONNX Runtime documentation</a
		>.
	</p>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-2 lg:grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl pb-4">GitHub</h1>
			If you are interested in joining the ONNX Runtime open source community, you might want to join us on GitHub where you can
			interact with other users and developers, participate in <a href="https://github.com/microsoft/onnxruntime/discussions" class="text-blue-500">discussions</a>, and get help with any
			<a href="https://github.com/microsoft/onnxruntime/issues" class="text-blue-500">issues</a> you encounter. You can also contribute to the project by reporting bugs, suggesting features,
			or submitting pull requests.
			<div class="py-4">
				<a href="https://github.com/microsoft/onnxruntime" class="btn btn-primary mr-4"
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

			<a href="./docs/tutorials/" rel="external" class="btn btn-primary mr-4"
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
