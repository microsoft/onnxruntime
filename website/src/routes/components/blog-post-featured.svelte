<script>
	import anime from "animejs";

	let badges = [
		['ONNX', 'primary'],
		['ML', 'primary']
	];
	let handleEnter = (e) => {
		anime({
			targets: e.target,
			scale: 1.05,
			duration: 1500,
		})
	}
	let handleLeave = (e) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<div on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="m-10">
	<a href="http://blog.com">
		<div class="card bg-base-100 border border-primary" data-svelte-h="svelte-1oufy6g">
			<figure>
				<img src="https://source.unsplash.com/random/400x300" alt="random" />
			</figure>
			<div class="card-body">
				<h2 class="card-title">Title</h2>
				<p>Description</p>
				<div class="card-actions">
					<div>
						{#each badges as badge}
							<div class="badge badge-{badge[1]} mr-1">{badge[0]}</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	</a>
</div>
