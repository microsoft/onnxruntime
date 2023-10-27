<script>
	import { base } from '$app/paths';
	import Ortgenerativeai from '../../images/undraw/image_ortgenerativeai.svelte';
</script>

<div class="container mx-auto px-10 mb-10">
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10">
		<div class="">
			<h1 class="text-4xl">Generative AI</h1>
			<br /><br />
			<p class="text-xl">
				Integrate the power of Generative AI and Large language Models (LLMs) in your apps and services with ONNX Runtime. No matter what language you develop in or what platform you need to run on, you can make use of state-of-the-art models
				for image synthesis, text generation, and more.
			</p>
			<br />
			<a href="./generative-ai" class="btn btn-primary">Learn more about ORT & Generative AI →</a>
		</div>
		<div class="m-auto overflow:hidden">
			<Ortgenerativeai />
		</div>
	</div>
</div>
