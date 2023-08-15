<script>
	import TestimonialCard from '../components/testimonial-card.svelte';
	let imgsrc = 'https://placehold.co/375x225';
	let imgalt = 'test';
	let title = 'Adobe';
	let description =
		'“With ONNX Runtime, Adobe Target got flexibility and standardization in one package: flexibility for our customers to train ML models in the frameworks of their choice, and standardization to robustly deploy those models at scale for fast inference, to deliver true, real-time personalized experiences.”';
	let author = '–Georgiana Copil, Senior Computer Scientist, Adobe';
</script>

<div class="container mx-auto">
	<h1 class="text-4xl">Our Partners</h1>
	<h2 class="text-2xl mb-4">Hear from some of the products and companies using ONNX Runtime:</h2>
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
		<TestimonialCard {imgsrc} {imgalt} {title} {description} {author} />
	</div>
</div>
