<script>
	import Header from '../components/header.svelte';
	import Footer from '../components/footer.svelte';
	import './github-markdown-light.css'
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
	 export let image;
	/**
	 * @type {any}
	 */
	 export let url;
	/**
	 * @type {any}
	 */
	 export let robots;
	 /**
	 * @type {any}
	 */

</script>

<svelte:head>
	<meta name="title" content={title} />
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="date" content={date} />
	<meta name="og:title" content={title} />
	<meta name="og:description" content={description} />
	<meta name="og:type" content="article" />
	<meta name="og:url" content={url} />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={description} />
	<!-- <meta name="image" content={image} />
	<meta name="og:image" content={image}  />
	<meta name="twitter:card" content={image}  />
	<meta name="twitter:image" content={image}  /> -->

	<meta name="robots" content={robots} />
</svelte:head>
<Header pathvar="" />
<div class="container mx-auto px-4 md:px-8 lg:px-48 pt-8">
	<article class="">
		<h1 class="text-5xl pb-2">{title}</h1>
		<p class="text-neutral">
            By: 
			{#each authors as author, i}
            <a href={authorsLink[i]} class="text-blue-500"
            >{author}</a
        >{i + 1 === authors.length ? '' : ', '}
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
