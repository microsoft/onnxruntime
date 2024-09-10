<script lang="ts">
	import anime from 'animejs';
	export let title: string;
	export let description: string;
	export let date: string;
	export let link: string;
	export let image: string;
	export let imagealt: string;
	export let linkarr: { name: string; link: string }[];
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.02,
			duration: 1500
		});
	};
	let handleLeave = (e: any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500
		});
	};
</script>

<article
	on:mouseenter={handleEnter}
	on:mouseleave={handleLeave}
	class="mb-4 bg-base-300 border rounded-md border-primary"
>
	<a href={link}>
		<div class="card card-side grid grid-cols-3">
			<div class="card-body col-span-3 md:col-span-2">
				<h2 class="card-title">{title}</h2>
				<p>{description}</p>
				<p class="text-blue-800 text-right">
					{date}
				</p>
				<div class="card-actions">
					{#each linkarr as item}
						<a class="btn btn-secondary rounded-sm" href={item.link} target="_blank">{item.name}</a>
					{/each}
				</div>
			</div>
			<div class="card-image col-span-1 m-auto hidden md:flex">
				<img src={image} alt={imagealt} />
			</div>
		</div>
	</a>
</article>
