<script>
	import { onMount } from 'svelte';
	import { themeChange } from 'theme-change';
	import onnximage from '../../images/ONNX-Icon.svg';
	import FaGithub from 'svelte-icons/fa/FaGithub.svelte';
	import FaYoutube from 'svelte-icons/fa/FaYoutube.svelte';
	import FaLinkedinIn from 'svelte-icons/fa/FaLinkedinIn.svelte';
	onMount(() => {
		themeChange(false);
	});
</script>
<!-- TODO: ONNX Icon color -->
<div class="navbar bg-base-100">
	<div class="navbar-start">
		<div class="dropdown">
			<button tabindex="0" class="btn btn-ghost lg:hidden">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
					><path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M4 6h16M4 12h8m-8 6h16"
					/></svg
				>
			</button>
			<button
				tabindex="0"
				class="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52"
			>
				<li><a href="/blogs">Blogs</a></li>
				<li><a href="/">Docs</a></li>
				<li>
					<p>Community</p>
					<ul class="p-2">
						<li><a href="/about">About ORT</a></li>
						<li><a href="/testimonials">Testimonials</a></li>
						<li>
							<a href="https://www.youtube.com/onnxruntime" target="_blank"
								><div class="w-4 h-4"><FaYoutube /></div>
								Youtube</a
							>
						</li>
						<li>
							<a href="https://www.linkedin.com/company/onnxruntime/" target="_blank"
								><div class="w-4 h-4"><FaLinkedinIn /></div>
								LinkedIn</a
							>
						</li>
						<li>
							<a href="https://www.tiktok.com/@onnxruntimeai" target="_blank"
								><svg
									xmlns="http://www.w3.org/2000/svg"
									height="1em"
									viewBox="0 0 448 512"
									fill="currentColor"
									><!--! Font Awesome Free 6.4.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path
										d="M448,209.91a210.06,210.06,0,0,1-122.77-39.25V349.38A162.55,162.55,0,1,1,185,188.31V278.2a74.62,74.62,0,1,0,52.23,71.18V0l88,0a121.18,121.18,0,0,0,1.86,22.17h0A122.18,122.18,0,0,0,381,102.39a121.43,121.43,0,0,0,67,20.14Z"
									/></svg
								>TikTok</a
							>
						</li>
					</ul>
				</li>
				<li>
					<a href="https://github.com/microsoft/onnxruntime" target="_blank"
						><div class="w-4 h-4"><FaGithub /></div>
						GitHub</a
					>
				</li>
			</button>
		</div>
		<a href="/" class="btn btn-ghost normal-case text-xl">
			<img src={onnximage} class="w-8 h-8 hidden lg:block" alt="ONNX Runtime logo" />ONNXRuntime
			<img src={onnximage} class="w-8 h-8 lg:hidden" alt="ONNX Runtime logo" />
		</a>
	</div>
	<div class="navbar-center hidden lg:flex">
		<ul class="menu menu-horizontal px-1">
			<li><a href="/blogs">Blogs</a></li>
			<li><a href="/">Docs</a></li>
			<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
			<li tabindex="0">
				<details class="z-[1]">
					<summary>Community</summary>
					<ul class="p-2">
						<li><a href="/about">About ORT</a></li>
						<li><a href="/testimonials">Testimonials</a></li>
						<li>
							<a href="https://www.youtube.com/onnxruntime" target="_blank"
								><div class="w-4 h-4"><FaYoutube /></div>
								Youtube</a
							>
						</li>
						<li>
							<a href="https://www.linkedin.com/company/onnxruntime/" target="_blank"
								><div class="w-4 h-4"><FaLinkedinIn /></div>
								LinkedIn</a
							>
						</li>
						<li>
							<a href="https://www.tiktok.com/@onnxruntimeai" target="_blank"
								><svg
									xmlns="http://www.w3.org/2000/svg"
									height="1em"
									viewBox="0 0 448 512"
									fill="currentColor"
									><!--! Font Awesome Free 6.4.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path
										d="M448,209.91a210.06,210.06,0,0,1-122.77-39.25V349.38A162.55,162.55,0,1,1,185,188.31V278.2a74.62,74.62,0,1,0,52.23,71.18V0l88,0a121.18,121.18,0,0,0,1.86,22.17h0A122.18,122.18,0,0,0,381,102.39a121.43,121.43,0,0,0,67,20.14Z"
									/></svg
								>TikTok</a
							>
						</li>
					</ul>
				</details>
			</li>
			<li>
				<a href="https://github.com/microsoft/onnxruntime" target="_blank"
					><div class="w-4 h-4"><FaGithub /></div>
					GitHub</a
				>
			</li>
		</ul>
	</div>
	<div class="navbar-end">
		<input
			aria-label="toggle color theme"
			data-toggle-theme="business, corporate"
			type="checkbox"
			class="toggle toggle-dark"
			checked
		/>
	</div>
</div>

<style>
</style>
