<script lang="ts">
	import { tick } from 'svelte';

	export let src: string;
	export let title: string;
	export let className = '';

	const supportedHosts = new Set([
		'youtube.com',
		'www.youtube.com',
		'youtube-nocookie.com',
		'www.youtube-nocookie.com'
	]);

	function parseEmbedUrl(value: string) {
		const url = new URL(value);
		const path = url.pathname.split('/').filter(Boolean);

		if (
			url.protocol !== 'https:' ||
			!supportedHosts.has(url.hostname) ||
			path[0] !== 'embed' ||
			!path[1]
		) {
			throw new Error(`Unsupported YouTube embed URL: ${value}`);
		}

		return { url, videoId: path[1] };
	}

	function autoplayUrl(url: URL) {
		const autoplay = new URL(url);
		autoplay.searchParams.set('autoplay', '1');
		return autoplay.toString();
	}

	let loaded = false;
	let iframe: HTMLIFrameElement;

	$: video = parseEmbedUrl(src);
	$: embedUrl = autoplayUrl(video.url);
	$: thumbnailUrl = `https://i.ytimg.com/vi/${encodeURIComponent(video.videoId)}/hqdefault.jpg`;

	async function loadVideo() {
		loaded = true;
		await tick();
		iframe.focus();
	}
</script>

<div class="relative overflow-hidden bg-black {className}">
	{#if loaded}
		<iframe
			bind:this={iframe}
			class="h-full w-full"
			src={embedUrl}
			title="{title} - YouTube video"
			frameborder="0"
			allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
			allowfullscreen
		/>
	{:else}
		<button
			type="button"
			class="group relative h-full w-full cursor-pointer focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-inset focus-visible:ring-white"
			aria-label="Play {title}"
			on:click={loadVideo}
		>
			<img class="h-full w-full object-cover" src={thumbnailUrl} alt="" loading="lazy" />
			<span
				class="absolute left-1/2 top-1/2 flex h-16 w-20 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-2xl bg-red-600 text-white shadow-lg transition-transform group-hover:scale-110 group-focus-visible:scale-110"
				aria-hidden="true"
			>
				<svg class="h-8 w-8" viewBox="0 0 24 24" fill="currentColor">
					<path d="M8 5v14l11-7z" />
				</svg>
			</span>
		</button>
	{/if}
</div>
