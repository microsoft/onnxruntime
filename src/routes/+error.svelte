<script>
    import { base } from '$app/paths';
	import OnnxIcon from '../images/onnx-icon.svelte';
</script>

<div class="container mx-auto text-center">
	<!-- TODO: Fix 404 for mobile -->
    <div class="flex items-center justify-center">
        <h1 class="text-huge">4</h1>
        <div>
            <OnnxIcon width={165} height={165} />
        </div>
        <h1 class="text-huge">4</h1>
    </div>
	<p>We're not sure how you got here, but you're probably lost!</p>
	<p>If you'd like, here's a way to get <a class="underline" href="{base}/">home</a>.</p>
    <br/>
    If you feel like something should be here, please <a class="underline" href="https://github.com/microsoft/onnxruntime/issues">open an issue</a> on GitHub.
</div>

<style>
	.text-huge {
		font-size: 15em;
	}
</style>
