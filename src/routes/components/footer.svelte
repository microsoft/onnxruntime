<script>
	import { base } from '$app/paths';
	import FaGithub from 'svelte-icons/fa/FaGithub.svelte';
	import FaYoutube from 'svelte-icons/fa/FaYoutube.svelte';
	import FaLinkedinIn from 'svelte-icons/fa/FaLinkedinIn.svelte';
	export let pathvar = '.';
</script>

<footer class="footer p-10 mt-10 bg-base-200 text-base-content z-40">
	<div>
		<p>ONNX Runtime<br />Copyright © Microsoft. All rights reserved.</p>
		<span class="footer-title">Follow us at:</span>
		<div class="grid grid-flow-col gap-4">
			<a aria-label="youtube" href="https://www.youtube.com/onnxruntime" target="_blank"
				><div class="w-8 h-8 pt-0.5 hover:text-primary"><FaYoutube /></div></a
			>
			<a aria-label="linkedin" href="https://www.linkedin.com/company/onnxruntime" target="_blank"
				><div class="w-8 h-8 hover:text-primary"><FaLinkedinIn /></div></a
			>
			<a aria-label="github" href="https://github.com/microsoft/onnxruntime" target="_blank"
				><div class="w-8 h-8 hover:text-primary"><FaGithub /></div></a
			>
		</div>
	</div>
	<div />
	<div>
		<span class="footer-title">Get Started</span>
		<a href={pathvar+"/getting-started"} class="link link-hover">Install</a>
		<a href={pathvar+"/pytorch"} class="link link-hover">PyTorch</a>
	</div>
	<div>
		<span class="footer-title">Resources</span>
		<a href={pathvar+"/blogs"} class="link link-hover">Blogs</a>
		<a rel="external" href={pathvar+"/docs/tutorials"} class="link link-hover">Tutorials</a>
		<a rel="external" href={pathvar+"/docs/api/"} class="link link-hover">APIs</a>
	</div>
</footer>
