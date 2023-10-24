<script lang="ts">
	import '../app.css';
	import Header from './components/header.svelte';
	import Footer from './components/footer.svelte';
	import oneLight from 'svelte-highlight/styles/one-light';
	import { fade } from 'svelte/transition';
	import Analytics from './components/analytics.svelte';
	import { page } from '$app/stores';
	export let data;
</script>

<svelte:head>
	{@html oneLight}
	<title
		>ONNX Runtime | {data.pathname == '/'
			? 'Home'
			: data.pathname.substring(1).charAt(0).toUpperCase() + data.pathname.substring(2)}</title
	>
	<meta
		property="og:title"
		content="ONNX Runtime | {data.pathname == '/'
			? 'Home'
			: data.pathname.substring(1).charAt(0).toUpperCase() + data.pathname.substring(2)}"
	/>
	<meta
		name="description"
		content="Cross-platform accelerated machine learning. Built-in optimizations speed up training and inferencing with your existing technology stack."
	/>
	<meta http-equiv="X-UA-Compatible" content="ie=edge" />
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<meta name="theme-color" content="#B2B2B2" />
	<meta name="msapplication-TileColor" content="#B2B2B2" />
	<meta name="theme-color" content="#B2B2B2" />
	<!-- OpenGraph meta tags -->
	<meta
		property="og:description"
		content="Cross-platform accelerated machine learning. Built-in optimizations speed up training and inferencing with your existing technology stack."
	/>
	<meta property="og:image" content="https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png" />
	<meta property="og:url" content="https://onnxruntime.ai" />
	<meta property="og:type" content="website" />
</svelte:head>
<Analytics />
<div class="selection:bg-primary">
	{#if !$page.url.pathname.startsWith('/blogs/')}
		<Header />
	{/if}
	{#key data.pathname}
		<div in:fade={{ duration: 300, delay: 400 }} out:fade={{ duration: 300 }}>
			<slot />
		</div>
	{/key}
	{#if !$page.url.pathname.startsWith('/blogs/')}
		<Footer />
	{/if}
</div>
