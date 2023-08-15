<script lang="ts">
	export let title: string;
	export let description: string;
	export let imgsrc: string;
	export let imgalt: string;
    export let author: string;
	import anime from "animejs";
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.04,
			duration: 1500,
		})
	}
	let handleLeave = (e: any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<article on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="card bg-base-200">
	<figure><img src={imgsrc} alt={imgalt} /></figure>
	<div class="card-body">
		<h2 class="card-title">{title}</h2>
		<p>
			{description}
		</p>
        <p class="text-primary text-right">{author}</p>
	</div>
</article>
