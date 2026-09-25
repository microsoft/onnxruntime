<script lang="ts">
	import { cn } from '$lib/utils/cn';
	import { onMount } from 'svelte';

	export let items: {
		href: string;
		src: any;
		alt: string;
	}[];
	export let direction: 'left' | 'right' | undefined = 'left';
	export let speed: 'fast' | 'normal' | 'slow' | undefined = 'fast';
	export let pauseOnHover: boolean | undefined = true;
	export let className: string | undefined = undefined;

	let containerRef: HTMLDivElement;
	let scrollerRef: HTMLUListElement;
	let userPaused = false;

	onMount(() => {
		getDirection();
		getSpeed();
		start = true;
	});

	let start = false;

	const getDirection = () => {
		if (containerRef) {
			if (direction === 'left') {
				containerRef.style.setProperty('--animation-direction', 'forwards');
			} else {
				containerRef.style.setProperty('--animation-direction', 'reverse');
			}
		}
	};
	const getSpeed = () => {
		if (containerRef) {
			if (speed === 'fast') {
				containerRef.style.setProperty('--animation-duration', '20s');
			} else if (speed === 'normal') {
				containerRef.style.setProperty('--animation-duration', '40s');
			} else {
				containerRef.style.setProperty('--animation-duration', '80s');
			}
		}
	};

	const toggleScroll = () => {
		if (scrollerRef) {
			userPaused = !userPaused;
			scrollerRef.style.animationPlayState = userPaused ? 'paused' : '';
		}
	};

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			toggleScroll();
		}
	};

	const handleLinkFocus = (event: FocusEvent) => {
		if (event.target instanceof HTMLElement && scrollerRef) {
			scrollerRef.style.animation = 'none';
			event.target.scrollIntoView({ block: 'nearest', inline: 'nearest' });
		}
	};

	const handleLinkBlur = (event: FocusEvent) => {
		if (
			scrollerRef &&
			containerRef &&
			!(event.relatedTarget instanceof Node && scrollerRef.contains(event.relatedTarget))
		) {
			containerRef.scrollLeft = 0;
			scrollerRef.style.removeProperty('animation');
			scrollerRef.style.animationPlayState = userPaused ? 'paused' : '';
		}
	};
</script>

<div
	bind:this={containerRef}
	class={cn('scroller relative z-2 overflow-hidden ', className)}
	data-customer-carousel
>
	<button
		class="hover:bg-primary focus:bg-primary menu-item py-2 sr-only focus:not-sr-only"
		on:keydown={handleKeyDown}
		on:click={toggleScroll}>Toggle scrolling</button
	>
	<ul
		bind:this={scrollerRef}
		class={cn(
			'moving-cards flex w-max min-w-full shrink-0 flex-nowrap gap-4 py-4',
			start && 'animate-scroll',
			pauseOnHover && 'hover:[animation-play-state:paused]'
		)}
		aria-label="Customer testimonials"
	>
		{#each items as item (item.alt)}
			<li
				class="bg-slate-300 m-auto relative h-28 w-[200px] max-w-full flex-shrink-0 hover:scale-105 transition duration-200 rounded-md border border-2 border-secondary md:w-[200px]"
				data-carousel-original
			>
				<a
					href={item.href}
					class="block h-full w-full rounded-md focus:outline-none focus-visible:ring-4 focus-visible:ring-inset focus-visible:ring-primary focus-visible:ring-offset-2"
					on:focus={handleLinkFocus}
					on:blur={handleLinkBlur}
				>
					<img class="h-28 p-2 m-auto" src={item.src} alt={item.alt} />
				</a>
			</li>
		{/each}
		{#each items as item (item.alt)}
			<li
				class="bg-slate-300 m-auto relative h-28 w-[200px] max-w-full flex-shrink-0 hover:scale-105 transition duration-200 rounded-md border border-2 border-secondary md:w-[200px]"
				aria-hidden="true"
				data-carousel-copy
			>
				<a href={item.href} tabindex="-1">
					<img class="h-28 p-2 m-auto" src={item.src} alt="" />
				</a>
			</li>
		{/each}
	</ul>
</div>

<style>
	.moving-cards:focus-within {
		animation: none !important;
	}

	@media (prefers-reduced-motion: reduce) {
		.moving-cards {
			animation: none !important;
		}
	}
</style>
