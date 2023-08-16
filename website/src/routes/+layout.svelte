<script lang="ts">
	import '../app.css';
	import Header from './components/header.svelte';
	import Footer from './components/footer.svelte';
	import { onMount } from 'svelte';
	import IoIosClose from 'svelte-icons/io/IoIosClose.svelte';
	import anime from 'animejs';
	import oneLight from 'svelte-highlight/styles/one-light';

	let removetoast = (e: any) => {
		e.target.parentNode.parentNode.remove();
	};
	onMount(() => {
		anime({
			targets: '.toast',
			opacity: '1',
			duration: 1000,
			delay: 3000,
			easing: 'easeInOutQuad',
			begin: function () {
				document.querySelector<HTMLElement>('.toast')!.style.display = 'block';
			}
		});
		anime({
			targets: '.toast',
			delay: 3500,
			easing: 'easeOutElastic(1, .25)',
			translateY: -10
		});
	});
</script>

<svelte:head>
	{@html oneLight}
	<title>ONNX Runtime | Home</title>
	<meta property="og:title" content="ONNX Runtime | Home" />
	<meta
		name="description"
		content="Cross-platform accelerated machine learning. Built-in optimizations speed up training and inferencing with your existing technology stack."
	/>
	<meta http-equiv="X-UA-Compatible" content="ie=edge" />
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<!-- <meta name="theme-color" content="#00D1B2" /> -->
	<!-- <meta name="msapplication-TileColor" content="#da532c" /> -->
	<!-- <meta name="theme-color" content="#ffffff" /> -->
	<!-- OpenGraph meta tags -->
	<meta
		property="og:description"
		content="Cross-platform accelerated machine learning. Built-in optimizations speed up training and inferencing with your existing technology stack."
	/>
	<meta property="og:image" content="https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png" />
	<meta property="og:url" content="https://onnxruntime.ai" />
	<meta property="og:type" content="website" />
</svelte:head>
<div class="toast z-[2] opacity-0 hidden">
	<div class="alert alert-info">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			fill="none"
			viewBox="0 0 24 24"
			class="stroke-current shrink-0 w-6 h-6"
			><path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
			/></svg
		>
		<span
			>Please help us improve ONNX Runtime <br class="md:hidden" />by participating in our
			<a class="underline" href="https://ncv.microsoft.com/UySXuzobM9">customer survey</a>.</span
		> <button aria-label="close"  on:click={removetoast} class="w-8 h-8 pt-1"><IoIosClose /></button>
	</div>
</div>
<div class="selection:bg-success">
	<Header />
	<slot />
	<Footer />
</div>
