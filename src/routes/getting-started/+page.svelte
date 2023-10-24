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
	<h1 class="text-4xl pb-4">Installation Instructions</h1>
	<Table />
	<p class="pt-4">
		For more in-depth installation instructions, check out the <a
			href="https://onnxruntime.ai/docs/tutorials/"
			class="text-blue-500">ONNX Runtime documentation</a
		>.
	</p>
	<p class="pt-4">
		For installation instructions, check out the <a
			href="https://onnxruntime.ai/docs/tutorials/"
			class="text-blue-500">ONNX Runtime documentation</a
		>.
	</p>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl pb-4">GitHub</h1>
			<div class="pb-4">
				<h1 class="text-2xl pb-2">Want to join our community?</h1>
				<a href="https://github.com/microsoft/onnxruntime" class="btn btn-primary mr-4"
					>ONNX Runtime GitHub →</a
				>
			</div>
			<div class="pb-4">
				<h1 class="text-2xl pb-2">Need help with an issue or feature request?</h1>
				<a href="https://github.com/microsoft/onnxruntime/issues" class="btn btn-primary mr-4"
					>ONNX Runtime GitHub Issues →</a
				>
			</div>
			<div class="pb-4">
				<h1 class="text-2xl pb-4">Have a question or idea?</h1>
				<a href="https://github.com/microsoft/onnxruntime/discussions" class="btn btn-primary mr-4"
					>ONNX Runtime GitHub Discussions →</a
				>
			</div>
		</div>
		<img src={githublogo} alt="GitHub Logo" />
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl">Videos</h1>
	<h1 class="text-2xl">Check out some of our videos to help you get started!</h1>
	<div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mx-auto my-10">
		<div>
			<iframe
				class="w-full h-64 pb-2"
				src="https://www.youtube.com/embed/F22zYGJdpB0?si=G00YdjyDSUA_akE8"
				title="YouTube video player"
				frameborder="0"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
				allowfullscreen
			/>
			<p class="text-center text-xl">
				Finetuning T5 Model in AzureML using Azure Container for PyTorch Curated Environment
			</p>
		</div>
		<div>
			<iframe
				class="w-full h-64 pb-2"
				src="https://www.youtube.com/embed/2dbXFQxsIDY?si=XFWUx8wbqgs6JAZ-"
				title="YouTube video player"
				frameborder="0"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
				allowfullscreen
			/>
			<p class="text-center text-xl">
				ONNX Runtime Azure EP for Hybrid Inferencing on Edge and Cloud
			</p>
		</div>
		<div>
			<iframe
				class="w-full h-64 pb-2"
				src="https://www.youtube.com/embed/waIeC3OIn70?si=jFs5SztQx4v9KBNj"
				title="YouTube video player"
				frameborder="0"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
				allowfullscreen
			/>
			<p class="text-center text-xl">v1.14 ONNX Runtime - Release Review</p>
		</div>
	</div>
	<a href="https://www.youtube.com/@ONNXRuntime" class="btn btn-primary mr-4"
		>ONNX Runtime YouTube channel →</a
	>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl">Tutorials</h1>
			<h1 class="text-2xl pb-2">
				Learn more about ONNX Runtime with tutorials from our documentation.
			</h1>

			<a href="/docs/tutorials/" rel="external" class="btn btn-primary mr-4"
				>ONNX Runtime Tutorials →</a
			>
		</div>
		<div class="col-span-1">
			<div class="m-auto">
				<ImageTutorials widthscale={0.5} heightscale={0.5} />
			</div>
		</div>
	</div>
</div>
