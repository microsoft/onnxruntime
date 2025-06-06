<script>
	import TestimonialCard from './testimonial-card.svelte';
	import ImageTestimonials from '../../images/undraw/image_testimonials.svelte';
	import adobeLogo from '../../images/logos/adobe-logo.png';
	import amdLogo from '../../images/logos/amd-logo.png';
	import antgroupLogo from '../../images/logos/antgroup-logo.png';
	import algoriddimLogo from '../../images/logos/algoriddim-logo.png';
	import atlasLogo from '../../images/logos/ATLAS-logo.png';
	import autodeskLogo from '../../images/logos/autodesk-logo.png';
	import bazaarvoiceLogo from '../../images/logos/bazaarvoice-logo.png';
	import camoLogo from '../../images/logos/camo-logo.png';
	import cephableLogo from '../../images/logos/cephable-logo.png';
	import clearbladeLogo from '../../images/logos/clearblade-logo.png';
	import deezerLogo from '../../images/logos/deezer-logo.png';
	import goodnotesLogo from '../../images/logos/goodnotes-logo.png';
	import graiphicLogo from '../../images/logos/graiphic-logo.png'
	import huggingfaceLogo from '../../images/logos/huggingface-logo.png';
	import hypefactorsLogo from '../../images/logos/hypefactors-logo.png';
	import infarmLogo from '../../images/logos/infarm-logo.png';
	import intelLogo from '../../images/logos/intel-logo.png';
	import intelligenzaeticaLogo from '../../images/logos/intelligenza-etica-logo.png';
	import navitaireLogo from '../../images/logos/navitaire-amadeus-logo.png';
	import nvidiaLogo from '../../images/logos/nvidia.png';
	import opennlpLogo from '../../images/logos/opennlp-logo.png';
	import oracleLogo from '../../images/logos/oracle-logo.png';
	import peakspeedLogo from '../../images/logos/PeakSpeed_logo.png';
	import piecesLogo from '../../images/logos/pieces-logo.png';
	import ptwLogo from '../../images/logos/ptw-logo.png';
	import redisLogo from '../../images/logos/redis-logo.png';
	import rockchipLogo from '../../images/logos/Rockchip-logo.png';
	import samtecLogo from '../../images/logos/samtec-logo.png';
	import sasLogo from '../../images/logos/sas-logo.png';
	import teradataLogo from '../../images/logos/teradata-logo.png';
	import topazlabsLogo from '../../images/logos/topazlabs-logo.png';
	import unrealengineLogo from '../../images/logos/ue-logo.png';
	import usdaLogo from '../../images/logos/usda-logo.png';
	import vespaLogo from '../../images/logos/vespa-logo.png';
	import writerLogo from '../../images/logos/writer-logo.png';
	import xilinxLogo from '../../images/logos/xilinx-logo.png';

	const quotes = [
		{
			title: 'Adobe',
			quote:
				'With ONNX Runtime, Adobe Target got flexibility and standardization in one package: flexibility for our customers to train ML models in the frameworks of their choice, and standardization to robustly deploy those models at scale for fast inference, to deliver true, real-time personalized experiences.',
			author: 'Georgiana Copil, Senior Computer Scientist, Adobe',
			imgsrc: adobeLogo,
			imgalt: 'Adobe logo'
		},
		{
			title: 'AMD',
			quote:
				"The ONNX Runtime integration with AMD's ROCm open software ecosystem helps our customers leverage the power of AMD Instinct GPUs to accelerate and scale their large machine learning models with flexibility across multiple frameworks.",
			author:
				'Andrew Dieckmann, Corporate Vice President and General Manager, AMD Data Center GPU & Accelerated Processing',
			imgsrc: amdLogo,
			imgalt: 'AMD logo'
		},
		{
			title: 'Ant Group',
			quote:
				'Using ONNX Runtime, we have improved the inference performance of many computer vision (CV) and natural language processing (NLP) models trained by multiple deep learning frameworks. These are part of the Alipay production system. We plan to use ONNX Runtime as the high-performance inference backend for more deep learning models in broad applications, such as click-through rate prediction and cross-modal prediction.',
			author: 'Xiaoming Zhang, Head of Inference Team, Ant Group',
			imgsrc: antgroupLogo,
			imgalt: 'Ant Group logo'
		},
		{
			title: 'Algoriddim',
			quote:
				'At Algoriddim we are using ONNX Runtime on Windows devices to power our Neural Mix™ feature that allows users to isolate vocal and instruments of any song in real-time, as well as our Automix feature that allows for seamless automatic DJ mixes. ONNX Runtime strikes the perfect balance between abstraction and flexibility, and using the QNN execution provider allows us to leverage the NPU on Copilot+ PCs for unparalleled inference performance while keeping the CPU free for other tasks.',
			author: 'Frederik Seiffert, CTO, Algoriddim',
			imgsrc: algoriddimLogo,
			imgalt: 'Algoriddim logo'
		},
		{
			title: 'Atlas Experiment',
			quote:
				'At CERN in the ATLAS experiment, we have integrated the C++ API of ONNX Runtime into our software framework: Athena. We are currently performing inferences using ONNX models especially in the reconstruction of electrons and muons. We are benefiting from its C++ compatibility, platform*-to-ONNX converters (* Keras, TensorFlow, PyTorch, etc) and its thread safety.',
			author: 'ATLAS Experiment team, CERN (European Organization for Nuclear Research)',
			imgsrc: atlasLogo,
			imgalt: 'Atlas Experiment logo'
		},
		{
			title: 'Autodesk',
			quote:
				"Autodesk Flame's use of ONNX Runtime offers major advantages with cross-platform compatibility and performance, providing artists the flexibility and interactivity they expect. This allows them to make use of machine learning models directly in Flame's creative toolset, augmenting the quality of their work and increasing the software's expandability. Microsoft's ONNX Runtime team has provided expert guidance and support throughout the development process, enabling us to put AI-powered creative tools in the hands of artists seeking high-quality VFX and finishing solutions.",
			author: 'Louis Martin, Sr. Manager of Software Development for Autodesk Flame',
			imgsrc: autodeskLogo,
			imgalt: 'Autodesk logo'
		},
		{
			title: 'Bazaarvoice',
			quote:
				'Building and deploying AI solutions to the cloud at scale is complex. With massive datasets and performance considerations, finding a harmonious balance is crucial. ONNX Runtime provided us with the flexibility to package a scikit-learn model built with Python, deploy it serverlessly to a Node.js environment, and run it in the cloud with impressive performance.',
			author: 'Matthew Leyburn, Software Engineer, Bazaarvoice',
			imgsrc: bazaarvoiceLogo,
			imgalt: 'Bazaarvoice logo'
		},
		{
			title: 'Camo',
			quote:
				'ONNX Runtime enables Camo Studio to deliver features like background segmentation and feature detection with speed and accuracy. It seamlessly integrated with our existing models and lets us target any processor, including the latest NPUs, saving us valuable development time and allowing us to bring innovative features to all our users. We recommend ONNX Runtime to any developer looking to streamline model deployment and unlock the full potential of their applications.',
			author: 'Aidan Fitzpatrick, Founder & CEO, Reincubate',
			imgsrc: camoLogo,
			imgalt: 'Camo logo'
		},
		{
			title: 'Cephable',
			quote:
				'The ONNX Runtime allows us to simultaneously target CPU, GPU and NPU enabled devices. Converting a model to NPU, using ONNX Runtime and AI Hub reduced our engineering effort from 30 days to 7 days. Given the current state of the art, that would likely be only 3 days today. This allows us to deliver cutting edge performance to our users, minimizing impact of AI/ML workloads when running other applications, and leaves more time to focus on feature work.',
			author: 'Jon Campbell, Director of Engineering, Cephable',
			imgsrc: cephableLogo,
			imgalt: 'Cephable logo'
		},
		{
			title: 'ClearBlade',
			quote:
				"ClearBlade's integration of ONNX Runtime with our Enterprise loT and Edge Platforms enables customers and partners to build Al models using any industry Al tool they want to use. Using this solution, our customers can use the ONNX Runtime Go language APIs to seamlessly deploy any model to run on equipment in remote locations or on the factory floor!",
			author: 'Aaron Allsbrook, CTO & Founder, ClearBlade',
			imgsrc: clearbladeLogo,
			imgalt: 'Clearblade logo'
		},
		{
			title: 'Deezer',
			quote:
				"At Deezer, we use ONNX Runtime for machine learning powered features for music recommendations in our streaming service. ONNX Runtime's C API is easy to integrate with our software stack and enables us to run and deploy transformer models with great performance for real-time use cases.",
			author: 'Mathieu Morlon, Software Engineer, Deezer',
			imgsrc: deezerLogo,
			imgalt: 'Deezer logo'
		},
		{
			title: 'Goodnotes',
			quote:
				"Thanks to ONNX Runtime Web, Goodnotes has seamlessly implemented Scribble to Erase. The first Goodnotes AI feature for Android, Windows, and Web, delivering lightning-fast performance and an incredibly smooth user experience. It's a game-changer!",
			author: 'Pedro Gómez, Senior Software Engineer, Goodnotes',
			imgsrc: goodnotesLogo,
			imgalt: 'Goodnotes logo'
		},
		{
			title: 'Graiphic',
			quote:
				"With SOTA, we have developed the first complete ecosystem fully based on ONNX and ONNX Runtime. More than just supporting AI workloads, SOTA orchestrates graph-based computation at its core, enabling modular, scalable, and transparent execution across AI and non-AI domains alike. We believe ONNX is not just a format, it is the foundation for the future of graph-native computation.",
			author: 'Youssef Menjour, CTO and Co-founder, Graiphic',
			imgsrc: graiphicLogo,
			imgalt: 'Graiphic logo'
		},
		{
			title: 'Hugging Face',
			quote:
				'We use ONNX Runtime to easily deploy thousands of open-source state-of-the-art models in the Hugging Face model hub and accelerate private models for customers of the Accelerated Inference API on CPU and GPU.',
			author: 'Morgan Funtowicz, Machine Learning Engineer, Hugging Face',
			imgsrc: huggingfaceLogo,
			imgalt: 'Hugging Face logo'
		},
		{
			title: 'Hypefactors',
			quote:
				'ONNX Runtime powers many of our Natural Language Processing (NLP) and Computer Vision (CV) models that crunch the global media landscape in real-time. It is our go-to framework for scaling our production workload, providing important features ranging from built-in quantization tools to easy GPU and VNNI acceleration.',
			author: 'Viet Yen Nguyen, CTO, Hypefactors',
			imgsrc: hypefactorsLogo,
			imgalt: 'Hypefactors logo'
		},
		{
			title: 'InFarm',
			quote:
				'InFarm delivers machine-learning powered solutions for intelligent farming, running computer vision models on a variety of hardware, including on-premise GPU clusters, edge computing devices like NVIDIA Jetsons, and cloud-based CPU and GPU clusters. ONNX Runtime enables InFarm to standardise the model formats and outputs of models generated across multiple teams to simplify deployment while also providing the best performance on all hardware targets.',
			author: 'Ashley Walker, Chief Information and Technology Officer, InFarm',
			imgsrc: infarmLogo,
			imgalt: 'InFarm logo'
		},
		{
			title: 'Intel',
			quote:
				'We are excited to support ONNX Runtime on the Intel® Distribution of OpenVINO™. This accelerates machine learning inference across Intel hardware and gives developers the flexibility to choose the combination of Intel hardware that best meets their needs from CPU to VPU or FPGA.',
			author: 'Jonathan Ballon, Vice President and General Manager, Intel Internet of Things Group',
			imgsrc: intelLogo,
			imgalt: 'Intel logo'
		},
		{
			title: 'Intelligenza Etica',
			quote:
				"We integrate AI models in various markets and regulated industries using many stacks and frameworks, merging R&D and Ethics. With ONNX Runtime, we provide maximum performance and flexibility to use the customers' preferred technology, from cloud to embedded systems.",
			author: 'Mauro Bennici, AI Architect and AI Ethicist, Intelligenza Etica',
			imgsrc: intelligenzaeticaLogo,
			imgalt: 'Intelligenza Etica logo'
		},
		{
			title: 'Navitaire',
			quote:
				"With customers around the globe, we're seeing increased interest in deploying more effective models to power pricing solutions via ONNX Runtime. ONNX Runtime's performance has given us the confidence to use this solution with our customers with more extreme transaction volume requirements.",
			author: 'Jason Coverston, Product Director, Navitaire',
			imgsrc: navitaireLogo,
			imgalt: 'Navitaire Amadeus logo'
		},
		{
			title: 'NVIDIA',
			quote:
				"ONNX Runtime enables our customers to easily apply NVIDIA TensorRT's powerful optimizations to machine learning models, irrespective of the training framework, and deploy across NVIDIA GPUs and edge devices.",
			author:
				'Kari Ann Briski, Sr. Director, Accelerated Computing Software and AI Product, NVIDIA',
			imgsrc: nvidiaLogo,
			imgalt: 'NVIDIA logo'
		},
		{
			title: 'Apache OpenNLP',
			quote:
				'The integration of ONNX Runtime into Apache OpenNLP 2.0 enables easy use of state-of-the-art Natural Language Processing (NLP) models in the Java ecosystem. For libraries and applications already using OpenNLP, such as Apache Lucene and Apache Solr, using ONNX Runtime via OpenNLP provides exciting new possibilities.',
			author:
				'Jeff Zemerick, Search Relevance Engineer at OpenSource Connections and Chair of the Apache OpenNLP project',
			imgsrc: opennlpLogo,
			imgalt: 'Apache OpenNLP logo'
		},
		{
			title: 'Oracle',
			quote:
				'The ONNX Runtime API for Java enables Java developers and Oracle customers to seamlessly consume and execute ONNX machine-learning models, while taking advantage of the expressive power, high performance, and scalability of Java.',
			author: 'Stephen Green, Director of Machine Learning Research Group, Oracle',
			imgsrc: oracleLogo,
			imgalt: 'Oracle logo'
		},
		{
			title: 'Peakspeed',
			quote:
				'Using a common model and code base, the ONNX Runtime allows Peakspeed to easily flip between platforms to help our customers choose the most cost-effective solution based on their infrastructure and requirements.',
			author: 'Oscar Kramer, Chief Geospatial Scientist, Peakspeed',
			imgsrc: peakspeedLogo,
			imgalt: 'Peakspeed logo'
		},
		{
			title: 'Pieces.app',
			quote:
				'ONNX Runtime provides us with a lightweight runtime that focuses on performance, yet allows our ML engineers to choose the best frameworks and models for the task at hand.',
			author: 'Brian Lambert, Machine Learning Engineer, Pieces.app',
			imgsrc: piecesLogo,
			imgalt: 'Pieces logo'
		},
		{
			title: 'PTW Dosimetry',
			quote:
				'The mission of PTW is to guarantee radiation therapy safely. Bringing an AI model from research into the clinic can be a challenge, however. These are very different software and hardware environments. ONNX Runtime bridges the gap and allows us to choose the best possible tools for research and be sure deployment into any environment will just work.',
			author: 'Jan Weidner, Research Software Engineer, PTW Dosimetry',
			imgsrc: ptwLogo,
			imgalt: 'PTW logo'
		},
		{
			title: 'Redis',
			quote:
				"ONNX Runtime underpins RedisAI's distinctive capability to run machine-learning and deep-learning model inference seamlessly inside of Redis. This integration allows data scientists to train models in their preferred ML framework (PyTorch, TensorFlow, etc), and serve those models from Redis for low-latency inference.",
			author: 'Sam Partee, Principal Engineer, Applied AI, Redis',
			imgsrc: redisLogo,
			imgalt: 'Redis logo'
		},
		{
			title: 'Rockchip',
			quote:
				'With support for ONNX Runtime, our customers and developers can cross the boundaries of the model training framework, easily deploy ML models in Rockchip NPU powered devices.',
			author: 'Feng Chen, Senior Vice President, Rockchip',
			imgsrc: rockchipLogo,
			imgalt: 'Rockchip logo'
		},
		{
			title: 'Samtec',
			quote:
				"We needed a runtime engine to handle the transition from data science land to a high-performance production runtime system. ONNX Runtime (ORT) simply ‘just worked'. Having no previous experience with ORT, I was able to easily convert my models, and had prototypes running inference in multiple languages within just a few hours. ORT will be my go-to runtime engine for the foreseeable future.",
			author: 'Bill McCrary, Application Architect, Samtec',
			imgsrc: samtecLogo,
			imgalt: 'Samtec logo'
		},
		{
			title: 'SAS',
			quote:
				'The unique combination of ONNX Runtime and SAS Event Stream Processing changes the game for developers and systems integrators by supporting flexible pipelines and enabling them to target multiple hardware platforms for the same AI models without bundling and packaging changes. This is crucial considering the additional build and test effort saved on an ongoing basis.',
			author: 'Saurabh Mishra, Senior Manager, Product Management, Internet of Things, SAS',
			imgsrc: sasLogo,
			imgalt: 'SAS logo'
		},
		{
			title: 'Teradata',
			quote:
				'Teradata provides a highly extensible framework that enables importation and inference of previously trained Machine Learning (ML) and Deep Learning (DL) models. ONNX Runtime enables us to expand the capabilities of Vantage Bring Your Own Model (BYOM) and gives data scientists more options for ML and DL models integration, inference and production deployment within Teradata Vantage ecosystem.',
			author: 'Michael Riordan, Director, Vantage Data Science and Analytics Products, Teradata',
			imgsrc: teradataLogo,
			imgalt: 'Teradata logo'
		},
		{
			title: 'Topaz Labs',
			quote:
				"ONNX Runtime's simple C API with DirectML provider enabled Topaz Labs to add support for AMD GPUs and NVIDIA Tensor Cores in just a couple of days. Furthermore, our models load many times faster on GPU than any other frameworks. Even our larger models with about 100 million parameters load within seconds.",
			author: 'Suraj Raghuraman, Head of AI Engine, Topaz Labs',
			imgsrc: topazlabsLogo,
			imgalt: 'Topaz Labs logo'
		},
		{
			title: 'Unreal Engine',
			quote:
				"We selected ONNX Runtime as the backend of Unreal Engine's Neural Network Interface (NNI) plugin inference system because of its extensibility to support the platforms that Unreal Engine runs on, while enabling ML practitioners to develop ML models in the frameworks of their choice. NNI evaluates neural networks in real time in Unreal Engine and acts as the foundation for game developers to use and deploy ML models to solve many development challenges, including animation, ML-based AI, camera tracking, and more.",
			author: 'Francisco Vicente Carrasco, Research Engineering Lead, Epic Games',
			imgsrc: unrealengineLogo,
			imgalt: 'Unreal Engine logo'
		},
		{
			title: 'United States Department of Agriculture, Agricultural Research Service',
			quote:
				'At the USDA we use ONNX Runtime in GuideMaker, a program we developed to design pools of guide RNAs needed for large-scale gene editing experiments with CRISPR-Cas. ONNX allowed us to make an existing model more interoperable and ONNX Runtime speeds up predictions of guide RNA binding.',
			author:
				'Adam Rivers, Computational Biologist, United States Department of Agriculture, Agricultural Research Service',
			imgsrc: usdaLogo,
			imgalt: 'USDA logo'
		},
		{
			title: 'Vespa.ai',
			quote:
				"ONNX Runtime has vastly increased Vespa.ai's capacity for evaluating large models, both in performance and model types we support.",
			author: 'Lester Solbakken, Principal Engineer, Vespa.ai',
			imgsrc: vespaLogo,
			imgalt: 'Vespa logo'
		},
		{
			title: 'Writer',
			quote:
				'ONNX Runtime has been very helpful to us at Writer in optimizing models for production. It lets us deploy more powerful models and still deliver results to our customers with the latency they expect.',
			author: 'Dave Buchanan, Director of AI and NLP, Writer',
			imgsrc: writerLogo,
			imgalt: 'Writer logo'
		},
		{
			title: 'Xilinx',
			quote:
				'Xilinx is excited that Microsoft has announced Vitis™ AI interoperability and runtime support for ONNX Runtime, enabling developers to deploy machine learning models for inference to FPGA IaaS such as Azure NP series VMs and Xilinx edge devices.',
			author: 'Sudip Nag, Corporate Vice President, Software & AI Products, Xilinx',
			imgsrc: xilinxLogo,
			imgalt: 'Xilinx logo'
		}
	];
	let description =
		"ONNX Runtime's many consumer testimonials. Hear from some of the products and companies using ONNX Runtime.";
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png';
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png';
	let authors = [''];
	let keywords = 'onnx runtime testimonials';
</script>

<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={image} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description} />
	<meta property="og:image" content={image} />

	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={image} />
	<meta property="twitter:card" content={imageSquare} />
</svelte:head>

<div class="container mx-auto">
	<div class="flex mb-5">
		<div class="mr-5 my-auto">
			<h1 class="text-5xl mx-4">Our Customers</h1>
			<h2 class="text-xl mb-4 mx-4">
				Hear from some of the products and companies using ONNX Runtime
			</h2>
		</div>
		<div class="hidden md:flex">
			<ImageTestimonials />
		</div>
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mx-4 md:mx-0">
		{#each quotes as quote}
			<TestimonialCard
				title={quote.title}
				description={quote.quote}
				author={quote.author}
				imgsrc={quote.imgsrc}
				imgalt={quote.imgalt}
			/>
		{/each}
	</div>
</div>
