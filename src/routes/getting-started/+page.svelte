<script lang="ts">
	import Table from './table.svelte';
	import githublogo from '../../images/logos/GitHub-Logo.svg';
	import ImageTutorials from '../../images/undraw/image_tutorials.svelte';
</script>

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
	<div class="grid grid-cols-3">
		<div class="col-span-2">
			<h1 class="text-4xl">Tutorials</h1>
			<h1 class="text-2xl">Learn more about ONNX Runtime with tutorials from our documentation.</h1>
		
			<a href="/docs/tutorials/" rel="external" class="btn btn-primary mr-4"
							>ONNX Runtime Tutorials →</a
						>
		</div>
		<div class="col-span-1">
			<div class="m-auto">
				<ImageTutorials widthscale={0.5} heightscale={0.5}/>
			</div>
		</div>
	</div> mn
</div>