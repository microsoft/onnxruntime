<script>
	import Sectionone from "./sectionone.svelte";
    import Sectiontwo from "./sectiontwo.svelte";
    import Sectionthree from "./sectionthree.svelte";
    import Sectionfour from "./sectionfour.svelte";
    import Sectionfive from "./sectionfive.svelte";
    const title = 'Generative AI + ONNX Runtime';
	const description =
		'Integrate the power of generative AI in your apps and services with ONNX Runtime. Broad platform support and deep optimizations empower usage of state-of-the-art models for image synthesis, text generation, and more.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	import stablediffusion1 from '../../images/StableDiffusion1.webp';
	import stablediffusion2 from '../../images/StableDiffusion2.webp';
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'onnxruntime, onnx runtime generative ai, onnx runtime generative ai models, onnx runtime generative ai deployment, onnx runtime generative ai performance, onnx runtime generative ai time to market, onnx runtime generative ai deploy anywhere, onnx runtime generative ai boost performance, onnx runtime generative ai improve time to market, onnx runtime generative ai production ready, onnx runtime generative ai lower latency, onnx runtime generative ai higher throughput, onnx runtime generative ai get innovations into production faster, onnx runtime generative ai testimonials, onnx runtime generative ai performance enhancements, onnx runtime generative ai production ready, onnx runtime generative ai lower latency, onnx runtime generative ai higher throughput, onnx runtime generative ai get innovations into production faster, onnx runtime generative ai testimonials, onnx runtime generative ai performance enhancements'

</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={image} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	<meta property="og:image" content={image} />
	
	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={image} />
	<meta property="twitter:card" content={imageSquare} />
</svelte:head>

<div class="container mx-auto">
    <Sectionone />
    <Sectiontwo />
    <Sectionthree />
    <Sectionfour />
    <Sectionfive />
</div>