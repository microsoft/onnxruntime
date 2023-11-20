<script>
	import { base } from '$app/paths';
	import OnnxLight from '../images/ONNX-Light.svelte';
</script>

<div class="container mx-auto text-center">
	<div class="flex items-center justify-center">
		<h1 class="text-9xl">4</h1>
		<div>
			<OnnxLight width={100} height={100} />
		</div>
		<h1 class="text-9xl">4</h1>
	</div>
	<p>We're not sure how you got here, but you're probably lost!</p>
	<p>If you'd like, here's a way to get <a class="underline" href="./">home</a>.</p>
	<br />
	If you feel like something should be here, please
	<a class="underline" href="https://github.com/microsoft/onnxruntime/issues">open an issue</a> on GitHub.
</div>
