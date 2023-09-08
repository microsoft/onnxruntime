<script lang="ts">
	import anime from 'animejs';
	export let title: string;
	export let description: string;
	export let date: string;
	export let link: string;
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
		<div class="card card-side">
			<div class="card-body">
				<h2 class="card-title">{title}</h2>
				<p>{description}</p>
				<p class="text-blue-500 text-right">
					{date}
				</p>
			</div>
		</div>
	</a>
</article>
