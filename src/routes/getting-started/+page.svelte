<script lang="ts">
	import Table from './table.svelte';
	import Temp from './temp.svelte'
	import githublogo from '../../images/logos/GitHub-Logo.svg';
	import ImageTutorials from '../../images/undraw/image_tutorials.svelte';
</script>

<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Installation Instructions</h1>
	<!-- <p class="pb-4">
		ONNX Runtime supports a variety of hardware and architectures to fit any need. Select the
		configuration you want to use and run the corresponding installation script.
	</p> -->
	<!-- <Table /> -->
	<Temp/>
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
	<h1 class="text-4xl">Tutorials</h1>
	<div class="grid grid-cols-3">
		<div class="col-span-3 lg:col-span-2">
			<div class="my-4">
				<h2 class="text-2xl">API Basics</h2>
				<p class="pb-4">
					These tutorials from the docs demonstrate basic ONNX Runtime inferencing with each
					language API.
				</p>
				<div>
					<a href="" class="btn btn-primary mr-4">Python →</a>
					<a href="" class="btn btn-primary mr-4">C++ →</a>
					<a href="" class="btn btn-primary mr-4">C# →</a>
					<a href="" class="btn btn-primary mr-4">Java →</a>
					<a href="" class="btn btn-primary mr-4">JavaScript →</a>
				</div>
			</div>
			<div class="my-4">
				<h2 class="text-2xl">Accelerated Machine Learning Frameworks</h2>
				<p class="pb-4">
					These tutorials from the docs demonstrate how to accelerate machine learning frameworks
					using ONNX Runtime.
				</p>
				<div>
					<a href="" class="btn btn-primary mr-4">PyTorch →</a>
					<a href="" class="btn btn-primary mr-4">TensorFlow →</a>
					<a href="" class="btn btn-primary mr-4">HuggingFace →</a>
				</div>
			</div>
			<div class="my-4">
				<h2 class="text-2xl">Deployment</h2>
				<p class="pb-4">
					These tutorials from the docs demonstrate cross-platform ONNX Runtime deployment.
				</p>
				<div>
					<a href="" class="btn btn-primary mr-4">AzureML →</a>
					<a href="" class="btn btn-primary mr-4">Mobile →</a>
					<a href="" class="btn btn-primary mr-4">Web →</a>
					<a href="" class="btn btn-primary mr-4">IoT & Edge →</a>
					<a href="" class="btn btn-primary mr-4">Traditional ML →</a>
				</div>
			</div>
		</div>
		<div class="m-auto hidden lg:grid">
			<ImageTutorials />
		</div>
	</div>
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
