<script>
	import FaGithub from 'svelte-icons/fa/FaGithub.svelte';
	import FaYoutube from 'svelte-icons/fa/FaYoutube.svelte';
	import FaLinkedinIn from 'svelte-icons/fa/FaLinkedinIn.svelte';
</script>

<footer class="footer p-10 mt-10 bg-base-200 text-base-content">
	<div>
		<p>ONNX Runtime<br />Copyright © Microsoft. All rights reserved.</p>
		<span class="footer-title">Follow us at:</span>
		<div class="grid grid-flow-col gap-4">
			<a aria-label="youtube" href="https://www.youtube.com/onnxruntime" target="_blank"
				><div class="w-8 h-8 pt-0.5"><FaYoutube /></div></a
			>
			<a aria-label="linkedin" href="https://www.linkedin.com/company/onnxruntime" target="_blank"
				><div class="w-8 h-8"><FaLinkedinIn /></div></a
			>
			<a aria-label="tiktok" href="https://www.tiktok.com/@onnxruntimeai" target="_blank"
				><svg xmlns="http://www.w3.org/2000/svg" height="2em" viewBox="0 0 448 512" fill="currentColor"
					><!--! Font Awesome Free 6.4.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path
						d="M448,209.91a210.06,210.06,0,0,1-122.77-39.25V349.38A162.55,162.55,0,1,1,185,188.31V278.2a74.62,74.62,0,1,0,52.23,71.18V0l88,0a121.18,121.18,0,0,0,1.86,22.17h0A122.18,122.18,0,0,0,381,102.39a121.43,121.43,0,0,0,67,20.14Z"
					/></svg
				>
			</a>
			<a aria-label="github" href="https://github.com/microsoft/onnxruntime" target="_blank"
				><div class="w-8 h-8"><FaGithub /></div></a
			>
		</div>
	</div>
	<div />
	<div>
		<span class="footer-title">Get Started</span>
		<a class="link link-hover">Install</a>
		<a class="link link-hover">PyTorch</a>
	</div>
	<div>
		<span class="footer-title">Resources</span>
		<a class="link link-hover">Blogs</a>
		<a class="link link-hover">Tutorials</a>
		<a class="link link-hover">APIs</a>
	</div>
</footer>

<style>
</style>
