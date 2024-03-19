<script lang="ts">
	import InfiniteMovingCards from '$lib/components/ui/InfiniteMovingCards/InfiniteMovingCards.svelte';
	import pytorchlogo from '../../images/logos/PyTorch_logo_black.svg';
	import windowslogo from '../../images/logos/Windows_logo_and_wordmark.svg';
	import huggingfacelogoTitle from '../../images/logos/hf-logo-with-title.svg';
	import adobeLogo from '../../images/logos/adobe-logo.png';
	import amdLogo from '../../images/logos/amd-logo.png';
	import antgroupLogo from '../../images/logos/antgroup-logo.png';
	import ATLASLogo from '../../images/logos/ATLAS-logo.png';
	import bazaarvoiceLogo from '../../images/logos/bazaarvoice-logo.png';
	import clearbladeLogo from '../../images/logos/clearblade-logo.png';
	import deezerLogo from '../../images/logos/deezer-logo.png';
	import hypefactorsLogo from '../../images/logos/hypefactors-logo.png';
	import infarmLogo from '../../images/logos/infarm-logo.png';
	import intelLogo from '../../images/logos/intel-logo.png';
	import intelligenzaEticaLogo from '../../images/logos/intelligenza-etica-logo.png';
	import navitaireAmadeusLogo from '../../images/logos/navitaire-amadeus-logo.png';
	import PeakSpeedLogo from '../../images/logos/PeakSpeed_logo.png';
	import piecesLogo from '../../images/logos/pieces-logo.png';
	import redisLogo from '../../images/logos/redis-logo.png';
	import RockchipLogo from '../../images/logos/Rockchip-logo.png';
	import samtecLogo from '../../images/logos/samtec-logo.png';
	import sasLogo from '../../images/logos/sas-logo.png';
	import teradataLogo from '../../images/logos/teradata-logo.png';
	import topazlabsLogo from '../../images/logos/topazlabs-logo.png';
	import ueLogo from '../../images/logos/ue-logo.png';
	import usdaLogo from '../../images/logos/usda-logo.png';
	import vespaLogo from '../../images/logos/vespa-logo.png';
	import writerLogo from '../../images/logos/writer-logo.png';
	import xilinxLogo from '../../images/logos/xilinx-logo.png';
	import huggingfaceLogo from '../../images/logos/huggingface-logo.png';
	import nvidiaLogo from '../../images/logos/nvidia.png';
	import oracleLogo from '../../images/logos/oracle-logo.png';

	const testimonials = [
		{
			href: './testimonials#Adobe',
			src: adobeLogo,
			alt: 'Adobe'
		},
		{
			href: './testimonials#AMD',
			src: amdLogo,
			alt: 'AMD'
		},
		{
			href: './testimonials#Ant%20Group',
			src: antgroupLogo,
			alt: 'Ant Group'
		},
		{
			href: './testimonials#ATLAS',
			src: ATLASLogo,
			alt: 'ATLAS'
		},
		{
			href: './testimonials#Bazaarvoice',
			src: bazaarvoiceLogo,
			alt: 'Bazaarvoice'
		},
		{
			href: './testimonials#ClearBlade',
			src: clearbladeLogo,
			alt: 'ClearBlade'
		},
		{
			href: './testimonials#Deezer',
			src: deezerLogo,
			alt: 'Deezer'
		},
		{
			href: './testimonials#Hugging%20Face',
			src: huggingfaceLogo,
			alt: 'Hugging Face'
		},
		{
			href: './testimonials#Hypefactors',
			src: hypefactorsLogo,
			alt: 'Hypefactors'
		},
		{
			href: './testimonials#Infarm',
			src: infarmLogo,
			alt: 'InFarm'
		},
		{
			href: './testimonials#Intel',
			src: intelLogo,
			alt: 'Intel'
		},
		{
			href: './testimonials#Intelligenza%20Etica',
			src: intelligenzaEticaLogo,
			alt: 'Intelligenza Etica'
		},
		{
			href: './testimonials#Navitaire%20Amadeus',
			src: navitaireAmadeusLogo,
			alt: 'Navitaire'
		},
		{
			href: './testimonials#NVIDIA',
			src: nvidiaLogo,
			alt: 'NVIDIA'
		},
		{
			href: './testimonials#Oracle',
			src: oracleLogo,
			alt: 'Oracle'
		},
		{
			href: './testimonials#PeakSpeed',
			src: PeakSpeedLogo,
			alt: 'Peakspeed'
		},
		{
			href: './testimonials#Pieces.app',
			src: piecesLogo,
			alt: 'Pieces'
		},
		{
			href: './testimonials#Redis',
			src: redisLogo,
			alt: 'Redis'
		},
		{
			href: './testimonials#Rockchip',
			src: RockchipLogo,
			alt: 'Rockchip'
		},
		{
			href: './testimonials#Samtec',
			src: samtecLogo,
			alt: 'Samtec'
		},
		{
			href: './testimonials#SAS',
			src: sasLogo,
			alt: 'SAS'
		},
		{
			href: './testimonials#Teradata',
			src: teradataLogo,
			alt: 'Teradata'
		},
		{
			href: './testimonials#Topaz%20Labs',
			src: topazlabsLogo,
			alt: 'Topaz Labs'
		},
		{
			href: './testimonials#Unreal%20Engine',
			src: ueLogo,
			alt: 'Unreal Engine'
		},
		{
			href: './testimonials#USDA',
			src: usdaLogo,
			alt: 'USDA'
		},
		{
			href: './testimonials#Vespa',
			src: vespaLogo,
			alt: 'Vespa'
		},
		{
			href: './testimonials#Writer',
			src: writerLogo,
			alt: 'Writer'
		},
		{
			href: './testimonials#Xilinx',
			src: xilinxLogo,
			alt: 'Xilinx'
		}
	];
</script>

<div class="lg:my-5">
	<h1 class="text-3xl my-5 text-center">Trusted By</h1>
	<div class="divider" />
	<InfiniteMovingCards
		items={testimonials.sort(() => Math.random() - 0.5)}
		direction="left"
		speed="slow"
	/>
	<div class="container mx-auto md:px-16 px-8 lg:my-10">
		<h1 class="text-2xl pt-10 pb-4">Learn more about how to use ONNX Runtime with</h1>
		<div class="grid md:grid-cols-3 grid-cols-1 gap-4 mx-auto pb-10">
			<a
				href="./pytorch"
				class="pt-2 hover:scale-105 transition duration-200 bg-slate-300 border-solid border-2 border-secondary rounded h-full"
				><img class="px-10 py-5" src={pytorchlogo} alt="PyTorch Logo" /></a
			>
			<a
				href="./windows"
				class="pt-2 hover:scale-105 transition duration-200 bg-slate-300 border-solid border-2 border-secondary rounded h-full"
				><img class="px-10 py-5" src={windowslogo} alt="Windows Logo" /></a
			>
			<a
				href="./huggingface"
				class=" hover:scale-105 transition duration-200 bg-slate-300 border-solid border-2 border-secondary rounded h-full"
				><img class="px-10 py-5" src={huggingfacelogoTitle} alt="HuggingFace Logo" /></a
			>
		</div>
	</div>
</div>