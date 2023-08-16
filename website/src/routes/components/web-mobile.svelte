<script lang="ts">
    import anime from "animejs"
    let handleEnter = (e:any) => {
		anime({
			targets: e.target,
			scale: 1.05,
			duration: 1500,
		})
	}
	let handleLeave = (e:any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<div class="container mx-auto lg:-mt-20">
	<div class="lg:grid pb-10 gap-10 grid-cols-2">
		<article on:mouseenter={handleEnter} on:mouseleave={handleLeave}  class="card bg-base-300 lg:mb-48 mb-10 grid md:grid-cols-3">
            <!-- Order-first and order-last classes makes for cleaner small device viewing -->
			<div class="card-body col-span-2 order-last md:order-first">
				<h2 class="card-title">ORT Training</h2>
				<p>
					ONNX Runtime can be used to accelerate both large model training and on-device training.
				</p>
				<div class="card-actions">
					<button class="btn btn-primary">Learn More →</button>
				</div>
			</div>
			<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
				<img src="https://placehold.co/250x225" alt="">
			</figure>
		</article>
		<article on:mouseenter={handleEnter} on:mouseleave={handleLeave}  class="card bg-base-300 lg:mt-48 grid md:grid-cols-3">
			<div class="card-body col-span-2 order-last md:order-first">
				<h2 class="card-title">ORT Inferencing</h2>
				<p>
					ONNX Runtime Inference powers machine learning models in key Microsoft products and
					services across Office, Azure, Bing, as well as dozens of community projects.
				</p>
				<div class="card-actions">
					<button class="btn btn-primary">Learn More →</button>
				</div>
			</div>
			<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
				<img src="https://placehold.co/250x225" alt="">
			</figure>
		</article>
	</div>
</div>
