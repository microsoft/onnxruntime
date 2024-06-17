<script>
	import Header from '../components/header.svelte';
	import Footer from '../components/footer.svelte';
	import './github-markdown-light.css';
	import { onMount } from 'svelte';
	/**
	 * @type {any}
	 */
	export let title;
	/**
	 * @type {any}
	 */
	export let description;
	/**
	 * @type {any}
	 */
	export let keywords;
	/**
	 * @type {any[]}
	 */
	export let authors;
	/**
	 * @type {string[]}
	 */
	export let authorsLink;
	/**
	 * @type {string}
	 */
	export let date;
	/**
	 * @type {undefined}
	 */
	export let updated;
	/**
	 * @type {any}
	 */
	export let url;
	/**
	 * @type {any}
	 */
	export let image;
	/**
	 * @type {string}
	 */
	 export let imageSquare;
	/**
	 * @type {string}
	 */
</script>

<svelte:head>
	<!-- Primary Meta Tags -->
<title>ONNX Runtime | {title}</title>
<meta name="title" content={title} />
<meta name="description" content={description} />
<meta name="author" content={authors.join(', ')} />
<meta name="keywords" content={keywords} />
<meta name="image" content={image} />

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website" />
<meta property="og:url" content={url} />
<meta property="og:title" content={title} />
<meta property="og:description" content={description}/>
<meta property="og:image" content={imageSquare ? imageSquare : image} />

<!-- Twitter -->
<meta property="twitter:card" content={imageSquare ? imageSquare : image} />
<meta property="twitter:url" content={url} />
<meta property="twitter:title" content={title} />
<meta property="twitter:description" content={description} />
<meta property="twitter:image" content={image} />
</svelte:head>
<Header pathvar="" />
<div class="container mx-auto px-4 md:px-8 lg:px-48 pt-8">
	<article class="">
		<h1 class="text-5xl pb-2">{title}</h1>
		<p class="text-neutral">
			{#if authors.length === 0}
				<br/>
			{:else}
				<p class="inline">By:</p>
			{/if}
			{#each authors as author, i}
				<a href={authorsLink[i]} class="text-blue-700">{author}</a>{i + 1 === authors.length
					? ''
					: ', '}
			{/each}
		</p>
		<p class="text-neutral">
			{date.toLocaleUpperCase()}
			{#if updated != undefined}
				<span class="italic text-stone-500">(Updated {updated})</span>
			{/if}
		</p>
		<div class="py-4 markdown-body">
			<slot />
		</div>
	</article>
</div>
<Footer pathvar="" />
