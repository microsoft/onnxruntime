<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	const title = 'PyTorch + ONNX Runtime';
	const description =
		'PyTorch leads the deep learning landscape with its readily digestible and flexible API; the large number of ready-made models available, particularly in the natural language (NLP) domain; as well as its domain specific libraries.';
	const imgsrc = 'https://placehold.co/600x300';
	const imgalt = '';
</script>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<img
					src=""
					alt=""
					class="rounded-xl"
				/>
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Deploy anywhere</h2>
				<p>Run PyTorch models on cloud, desktop, mobile, IoT, and even in the browser</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<img
					src=""
					alt=""
					class="rounded-xl"
				/>
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Boost performance</h2>
				<p>Accelerate PyTorch models to improve user experience and reduce costs</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<img
					src=""
					alt=""
					class="rounded-xl"
				/>
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improve time to market</h2>
				<p>Used by Microsoft and many others for their production PyTorch workloads</p>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Why PyTorch + ONNX Runtime?</h1>
	<div class="container mx-auto">
		<div class="lg:grid pb-10 gap-10 grid-cols-2">
			<article class="card bg-base-300 lg:mb-48 mb-10 grid md:grid-cols-3">
				<!-- Order-first and order-last classes makes for cleaner small device viewing -->
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Native support in PyTorch</h2>
					<p>
						PyTorch includes support for ONNX through the torch.onnx APIs to simplify exporting your
						PyTorch model to the portable ONNX format. The ONNX Runtime team maintains these
						exporter APIs to ensure a high level of compatibility with PyTorch models.
					</p>
					<div class="card-actions">
						<a class="btn btn-primary">ORT Web Docs →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<img src="https://placehold.co/250x225" alt="" />
				</figure>
			</article>
			<article class="card bg-base-300 lg:mt-48 grid md:grid-cols-3">
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Python not required</h2>
					<p>
						Training PyTorch models requires Python but that can be a significant obstacle to
						deploying PyTorch models to many production environments, especially Android and iOS
						mobile devices. ONNX Runtime is designed for production and provides APIs in C/C++, C#,
						Java, and Objective-C, helping create a bridge from your PyTorch training environment to
						a successful PyTorch production deployment.
					</p>
					<div class="card-actions">
						<a class="btn btn-primary">ORT Mobile Docs →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<img src="https://placehold.co/250x225" alt="" />
				</figure>
			</article>
		</div>
	</div>
	<div class="container mx-auto">
		<div class="lg:grid pb-10 gap-10 grid-cols-2">
			<article class="card bg-base-300 lg:mb-48 mb-10 grid md:grid-cols-3">
				<!-- Order-first and order-last classes makes for cleaner small device viewing -->
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Lower latency, higher throughput</h2>
					<p>
						Better performance can help improve your user experience and lower your operating costs.
						A wide range of models from computer vision (ResNet, MobileNet, Inception, YOLO, super
						resolution, etc) to speech and NLP (BERT, RoBERTa, GPT-2, T5, etc) can benefit from ONNX
						Runtime's optimized performance. The ONNX Runtime team regularly benchmarks and
						optimizes top models for performance. ONNX Runtime also integrates with top hardware
						accelerator libraries like TensorRT and OpenVINO so you can get the best performance on
						the hardware available while using the same common APIs across all your target
						platforms.
					</p>
					<div class="card-actions">
						<a class="btn btn-primary">ORT Web Docs →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<img src="https://placehold.co/250x225" alt="" />
				</figure>
			</article>
			<article class="card bg-base-300 lg:mt-48 grid md:grid-cols-3">
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Get innovations into production faster</h2>
					<p> 
						Development agility is a key factor in overall costs. ONNX Runtime was built on the
						experience of taking PyTorch models to production in high scale services like Microsoft
						Office, Bing, and Azure. It used to take weeks and months to take a model from R&D to
						production. With ONNX Runtime, models can be ready to be deployed at scale in hours or
						days.
					</p>
					<div class="card-actions">
						<a class="btn btn-primary">ORT Mobile Docs →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<img src="https://placehold.co/250x225" alt="" />
				</figure>
			</article>
		</div>
	</div>
</div>
