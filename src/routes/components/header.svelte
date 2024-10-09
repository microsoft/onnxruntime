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

	let innerHeight = 0; 
	let innerWidth= 0; 
	$: smallWindowHeight = innerHeight <= 360;
	$: smallWindowWidth = innerWidth <= 360;
</script>
<svelte:window bind:innerHeight bind:innerWidth />
<div role="navigation" class="navbar bg-base-100 top-0 z-50 {smallWindowHeight ? "" : "sticky"}">
	<!-- Navbar for mobile -->
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
				<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/getting-started'}>Get Started</a></li>
				<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/blogs'}>Blogs</a></li>
				<li><a class="hover:bg-primary focus:bg-primary" rel="external" href={pathvar + '/docs'}>Docs</a></li>
				<li>
					<p class="hover:bg-primary focus:bg-primary">Community</p>
					<ul class="p-2">
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/models'}>Models</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/roadmap'}>Roadmap</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/events'}>Events</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/testimonials'}>Testimonials</a></li>
						<li>
							<a class="hover:bg-primary focus:bg-primary" href="https://www.youtube.com/onnxruntime" target="_blank"
								><div class="w-4 h-4"><FaYoutube /></div>
								Youtube</a
							>
						</li>
						<li>
							<a class="hover:bg-primary focus:bg-primary" href="https://www.linkedin.com/company/onnxruntime/" target="_blank"
								><div class="w-4 h-4"><FaLinkedinIn /></div>
								LinkedIn</a
							>
						</li>
					</ul>
				</li>
				<li>
					<a class="hover:bg-primary focus:bg-primary" href="https://github.com/microsoft/onnxruntime" target="_blank"
						><div class="w-4 h-4"><FaGithub /></div>
						GitHub</a
					>
				</li>
			</button>
		</div>
		<a class="hover:bg-primary focus:bg-primary btn btn-ghost normal-case text-xl" href={pathvar + '/'} >
			<div class="hidden lg:inline">
				<OnnxLight width={32} height={32} />
			</div>
			{smallWindowWidth ? "ORT" : "ONNX Runtime" }
			<div class="lg:hidden">
				<OnnxLight width={32} height={32} />
			</div>
		</a>
		
		<a class="hover:bg-primary focus:bg-primary menu-item py-2 sr-only focus:not-sr-only" href="#main-content">Skip to main content</a>
	</div>
	<!-- Navbar for deskop -->
	<div class="navbar-center hidden lg:flex">
		<ul class="menu menu-horizontal px-1">		
			<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/getting-started'}>Get Started</a></li>
			<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/blogs'}>Blogs</a></li>
			<li><a class="hover:bg-primary focus:bg-primary" rel="external" href={pathvar + '/docs'}>Docs</a></li>
			<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
			<li tabindex="0">
				<details class="z-[1]">
					<summary class="hover:bg-primary focus:bg-primary">Community</summary>
					<ul class="p-2">
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/models'}>Models</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/roadmap'}>Roadmap</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/events'}>Events</a></li>
						<li><a class="hover:bg-primary focus:bg-primary" href={pathvar + '/testimonials'}>Testimonials</a></li>
						<li>
							<a class="hover:bg-primary focus:bg-primary" href="https://www.youtube.com/onnxruntime" target="_blank"
								><div class="w-4 h-4"><FaYoutube /></div>
								Youtube</a
							>
						</li>
						<li>
							<a class="hover:bg-primary focus:bg-primary" href="https://www.linkedin.com/company/onnxruntime/" target="_blank"
								><div class="w-4 h-4"><FaLinkedinIn /></div>
								LinkedIn</a
							>
						</li>
					</ul>
				</details>
			</li>
			<li>
				<a class="hover:bg-primary focus:bg-primary" href="https://github.com/microsoft/onnxruntime" target="_blank"
					><div class="w-4 h-4"><FaGithub /></div>
					GitHub</a
				>
			</li>
		</ul>
	</div>
	<div class="navbar-end">
		<label class="cursor-pointer grid place-items-center mr-4">
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
	</div>
</div>
