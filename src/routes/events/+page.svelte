<script>
	// @ts-nocheck
	import anime from 'animejs';
	import { onMount } from 'svelte';
	import ImageEvents from '../../images/undraw/image_events.svelte';
	import converttoort from '../../images/events/converttoort.jpg';
	import Event from './event-post.svelte';
	import Videogallery from '../components/videogallery.svelte';
	let events = [
		{
			title: 'ONNX Runtime Community Meetup',
			date: 'February 9th, 2024',
			blurb:
				'ONNX Runtime community meet up where attendees learned more about how they can integrate the power of Generative AI and Large language Models (LLMs) in their apps and services, combined with ONNX Runtime speedup.',
			linkarr: [
				{ name: 'Event Page', link: 'https://developer.microsoft.com/en-us/reactor/events/21648/' },
				{
					name: 'Event Videos',
					link: 'https://www.youtube.com/playlist?list=PLqPV7GJuaUV8qw9Zj5zXvzUZdSxx_sn7U'
				}
			],
			image: converttoort,
			imagealt: 'Slide detailing how to convert from various frameworks to ONNX, then deploy anywhere using ORT'
		}
	];

	let videos = [
		{
			title: 'Workshop: LLMs Fine Tuning and Inferencing Using ONNXRuntime',
			link: 'https://www.youtube-nocookie.com/embed/lOp8WK0fa94?si=rHn8fxAii3Ksr8Ey'
		}
	];
	let description = 'ONNX Runtime Events - information on past and future ONNX Runtime Events.';
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'onnxruntime, onnx runtime events, onnx runtime community events, onnx runtime community meetup, onnx runtime community meetups'
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
	<div class="flex">
		<h1 class="text-5xl my-auto mx-4">Events</h1>
		<div class="ml-5 hidden md:flex">
			<ImageEvents />
		</div>
	</div>
	<div class="pt-5 mx-4 md:mx-10">
		<h3 class="text-3xl pb-8">Upcoming Events</h3>
		<div class="grid gap-4 grid-cols-1 lg:grid-cols-3" />
	</div>
	<div class="mx-4 md:mx-10">
		<h3 class="text-3xl pb-8">Past Events</h3>
		<div class="grid gap-4 grid-cols-1">
			{#each events as event, i}
				<Event
					title={event.title}
					description={event.blurb}
					date={event.date}
					linkarr={event.linkarr}
					image={event.image}
					imagealt={event.imagealt}
				/>
			{/each}
		</div>
	</div>
	<div class="mx-4 md:mx-10">
		<h3 class="text-3xl py-4">Past Event Videos</h3>
		<Videogallery {videos} title="false" px="0" my="0" />
	</div>
</div>
