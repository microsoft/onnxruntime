<script>
	import { base } from '$app/paths';
	import { onMount } from 'svelte';
	import { themeChange } from 'theme-change';
	import FaGithub from 'svelte-icons/fa/FaGithub.svelte';
	import FaYoutube from 'svelte-icons/fa/FaYoutube.svelte';
	import FaLinkedinIn from 'svelte-icons/fa/FaLinkedinIn.svelte';
	import OnnxLight from '../../images/ONNX-Light.svelte';
	export let pathvar = '.';
	onMount(() => {
		themeChange(false);
	});
</script>

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
				<li><a href={pathvar + '/blogs'}>Blogs</a></li>
				<li><a rel="external" href={pathvar + '/docs'}>Docs</a></li>
				<li>
					<p>Community</p>
					<ul class="p-2">
						<li><a href={pathvar + '/testimonials'}>Testimonials</a></li>
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
		<a href={pathvar + '/'} class="btn btn-ghost normal-case text-xl">
			<div class="hidden lg:inline">
				<OnnxLight width={32} height={32} />
			</div>
			ONNX Runtime
			<div class="lg:hidden">
				<OnnxLight width={32} height={32} />
			</div>
		</a>
	</div>
	<div class="navbar-center hidden lg:flex">
		<ul class="menu menu-horizontal px-1">
			<li><a href={pathvar + '/blogs'}>Blogs</a></li>
			<li><a rel="external" href={pathvar + '/docs'}>Docs</a></li>
			<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
			<li tabindex="0">
				<details class="z-[1]">
					<summary>Community</summary>
					<ul class="p-2">
						<li><a href={pathvar + '/testimonials'}>Testimonials</a></li>
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
			data-toggle-theme="lightmode, darkmode"
			type="checkbox"
			class="toggle"
			checked
			id="colortoggle"
		/>
	</div>
</div>
