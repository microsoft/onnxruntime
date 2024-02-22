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

<div class="navbar bg-base-100 sticky top-0 z-50">
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
				<li><a href={pathvar + '/getting-started'}>Get Started</a></li>
				<li><a href={pathvar + '/blogs'}>Blogs</a></li>
				<li><a rel="external" href={pathvar + '/docs'}>Docs</a></li>
				<li>
					<p>Community</p>
					<ul class="p-2">
						<li><a href={pathvar + '/events'}>Events</a></li>
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
			<li><a href={pathvar + '/getting-started'}>Get Started</a></li>
			<li><a href={pathvar + '/blogs'}>Blogs</a></li>
			<li><a rel="external" href={pathvar + '/docs'}>Docs</a></li>
			<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
			<li tabindex="0">
				<details class="z-[1]">
					<summary>Community</summary>
					<ul class="p-2">
						<li><a href={pathvar + '/events'}>Events</a></li>
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
		<label class="cursor-pointer grid place-items-center">
			<input
				aria-label="toggle color theme"
				data-toggle-theme="lightmode, darkmode"
				type="checkbox"
				class="toggle row-start-1 col-start-1 col-span-2 bg-base-content rounded-md"
				checked
				id="colortoggle"
			/><svg
				class="col-start-1 row-start-1 stroke-base-100 fill-base-100"
				xmlns="http://www.w3.org/2000/svg"
				width="14"
				height="14"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
				><circle cx="12" cy="12" r="5" /><path
					d="M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"
				/></svg
			>
			<svg
				class="col-start-2 row-start-1 stroke-base-100 fill-base-100"
				xmlns="http://www.w3.org/2000/svg"
				width="14"
				height="14"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg
			>
		</label>
		<!-- <label class="swap swap-rotate border rounded-lg bg-base-content p-1">
			<input type="checkbox" class="theme-controller" value="darkmode" />
			<svg class="swap-on fill-black w-8 h-8" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.64,17l-.71.71a1,1,0,0,0,0,1.41,1,1,0,0,0,1.41,0l.71-.71A1,1,0,0,0,5.64,17ZM5,12a1,1,0,0,0-1-1H3a1,1,0,0,0,0,2H4A1,1,0,0,0,5,12Zm7-7a1,1,0,0,0,1-1V3a1,1,0,0,0-2,0V4A1,1,0,0,0,12,5ZM5.64,7.05a1,1,0,0,0,.7.29,1,1,0,0,0,.71-.29,1,1,0,0,0,0-1.41l-.71-.71A1,1,0,0,0,4.93,6.34Zm12,.29a1,1,0,0,0,.7-.29l.71-.71a1,1,0,1,0-1.41-1.41L17,5.64a1,1,0,0,0,0,1.41A1,1,0,0,0,17.66,7.34ZM21,11H20a1,1,0,0,0,0,2h1a1,1,0,0,0,0-2Zm-9,8a1,1,0,0,0-1,1v1a1,1,0,0,0,2,0V20A1,1,0,0,0,12,19ZM18.36,17A1,1,0,0,0,17,18.36l.71.71a1,1,0,0,0,1.41,0,1,1,0,0,0,0-1.41ZM12,6.5A5.5,5.5,0,1,0,17.5,12,5.51,5.51,0,0,0,12,6.5Zm0,9A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"/></svg>

			<svg class="swap-off fill-white w-8 h-8" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M21.64,13a1,1,0,0,0-1.05-.14,8.05,8.05,0,0,1-3.37.73A8.15,8.15,0,0,1,9.08,5.49a8.59,8.59,0,0,1,.25-2A1,1,0,0,0,8,2.36,10.14,10.14,0,1,0,22,14.05,1,1,0,0,0,21.64,13Zm-9.5,6.69A8.14,8.14,0,0,1,7.08,5.22v.27A10.15,10.15,0,0,0,17.22,15.63a9.79,9.79,0,0,0,2.1-.22A8.11,8.11,0,0,1,12.14,19.73Z"/></svg>
			
		  </label>  -->
	</div>
</div>
