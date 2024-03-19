<script lang="ts">
	import { cn } from '$lib/utils/cn';
	import { onMount } from 'svelte';

	export let items: {
		href: string,
    	src: any,
    	alt: string,
	}[];
	export let direction: 'left' | 'right' | undefined = 'left';
	export let speed: 'fast' | 'normal' | 'slow' | undefined = 'fast';
	export let pauseOnHover: boolean | undefined = true;
	export let className: string | undefined = undefined;

	let containerRef: HTMLDivElement;
	let scrollerRef: HTMLUListElement;

	onMount(() => {
		addAnimation();
	});

	let start = false;

	function addAnimation() {
		if (containerRef && scrollerRef) {
			const scrollerContent = Array.from(scrollerRef.children);

			scrollerContent.forEach((item) => {
				const duplicatedItem = item.cloneNode(true);
				if (scrollerRef) {
					scrollerRef.appendChild(duplicatedItem);
				}
			});

			getDirection();
			getSpeed();
			start = true;
		}
	}
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
</script>

<div bind:this={containerRef} class={cn('scroller relative z-20 overflow-hidden ', className)}>
	<ul
		bind:this={scrollerRef}
		class={cn(
			' flex w-max min-w-full shrink-0 flex-nowrap gap-4 py-4',
			start && 'animate-scroll ',
			pauseOnHover && 'hover:[animation-play-state:paused]'
		)}
	>
		{#each items as item, idx (item.alt)}
			<a
			href={item.href}
				class="bg-slate-300 m-auto relative h-28 w-[200px] max-w-full flex-shrink-0 hover:scale-105 transition duration-200 rounded-md border border-2 border-secondary md:w-[200px]"
			>
				<img class="h-28 p-2 m-auto" src={item.src} alt={item.alt}>
			</a>
		{/each}
	</ul>
</div>
