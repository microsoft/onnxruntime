<script>
	// @ts-nocheck

	import Blog from './blog-post.svelte';
	import FeaturedBlog from './blog-post-featured.svelte';
	import anime from 'animejs';
	import { onMount } from 'svelte';
	import ImageBlogs from '../../images/undraw/image_blogs.svelte';
	import HFImage from '../../images/blogs/hugging-face-blog-img.png';
	import LlamaImage from '../../images/blogs/accelerating-llama-2/Figure1-LLaMA-2-7B-E2E-Throughput.png';
	onMount(() => {
		anime({
			targets: '.border-primary',
			translateY: -20,
			direction: 'alternate',
			loop: false,
			delay: function (el, i, l) {
				return 500 + i * 50;
			},
			endDelay: function (el, i, l) {
				return (l - i) * 50;
			}
		});
	});
	let featuredblog = [
		{
			title: 'Accelerating LLaMA-2 Inference with ONNX Runtime',
			date: 'November 14th, 2023',
			blurb: 'Learn how ONNX Runtime can speed up LLaMA-2 inference by up to 4.5X',
			link: 'blogs/accelerating-llama-2',
			image: LlamaImage,
			imgalt: 'LLaMA-2 e2e throughput'
		},
		{
			title: 'Run PyTorch models on the edge',
			date: 'October 12th, 2023',
			blurb:
				'Everything you need to know about running PyTorch models on the edge with ONNX Runtime.',
			link: 'blogs/pytorch-on-the-edge',
			image:
				'https://onnxruntime.ai/_app/immutable/assets/pytorch-on-the-edge-with-ort.cdaa9c84.png',
				imgalt: 'Run PyTorch models on the edge'
		},
		{
			title: 'Accelerating over 130,000 Hugging Face models with ONNX Runtime',
			date: 'October 4th, 2023',
			blurb:
				'Learn more on how ONNX Runtime helps users accelerate open source machine learning models from Hugging Face.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/10/04/accelerating-over-130000-hugging-face-models-with-onnx-runtime/',
			image: HFImage,
			imgalt: 'Hugging Face models with ONNX Runtime'
		}
	];
	let blogs = [
		{
			title: 'On-Device Training with ONNX Runtime: A deep dive',
			date: 'July 5th, 2023',
			blurb:
				'This blog presents technical details of On-Device training with ONNX Runtime. It explains how On-Device Training works and what are the different steps and artifacts involved in the training process. This information will help you train your models on edge devices.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/07/05/on-device-training-with-onnx-runtime-a-deep-dive/',
			image:
				'https://cloudblogs.microsoft.com/opensource/wp-content/uploads/sites/37/2023/06/Open-Source-1.webp'
		},
		{
			title:
				'Build and deploy fast and portable speech recognition applications with ONNX Runtime and Whisper',
			date: 'June 7th, 2023',
			blurb:
				'Learn how ONNX Runtime accelerates Whisper and makes it easy to deploy on desktop, mobile, in the cloud, and even in the browser.',
			link: 'https://medium.com/microsoftazure/build-and-deploy-fast-and-portable-speech-recognition-applications-with-onnx-runtime-and-whisper-5bf0969dd56b',
			image: 'https://miro.medium.com/v2/resize:fit:1100/format:webp/1*DJH8_6GS06-N32tkVhdTOw.png'
		},
		{
			title: 'On-Device Training: Efficient training on the edge with ONNX Runtime',
			date: 'May 31st, 2023',
			blurb:
				'This blog introduces On-Device Training to enable training models on edge devices with the data available on-edge. It extends ORT Inference on edge to include federated learning and personalization scenarios.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/05/31/on-device-training-efficient-training-on-the-edge-with-onnx-runtime/'
		},
		{
			title:
				'Unlocking the end-to-end Windows AI developer experience using ONNX runtime and Olive',
			date: 'May 23th, 2023',
			blurb:
				'This blog reviews the new capabilities of ONNX Runtime and the Olive toolchain to support hybrid inferencing, NPU EPs, and hardware aware model optimizations on Windows and other platforms',
			link: 'https://blogs.windows.com/windowsdeveloper/2023/05/23/unlocking-the-end-to-end-windows-ai-developer-experience-using-onnx-runtime-and-olive'
		},
		{
			title:
				'Bringing the power of AI to Windows 11 - unlocking a new era of productivity for customers and developers with Windows Copilot and Dev Home',
			date: 'May 23th, 2023',
			blurb:
				'This blog reviews AI in Windows 11, including ONNX Runtime as the gateway to Windows AI and new ONNX Runtime capabilities on Windows',
			link: 'https://blogs.windows.com/windowsdeveloper/2023/05/23/bringing-the-power-of-ai-to-windows-11-unlocking-a-new-era-of-productivity-for-customers-and-developers-with-windows-copilot-and-dev-home'
		},
		{
			title: 'Optimize DirectML performance with Olive',
			date: 'May 23th, 2023',
			blurb: 'This blog shows how to use Olive to optimize models for DML EP in ONNX Runtime',
			link: 'https://devblogs.microsoft.com/windowsai/optimize-directml-performance-with-olive'
		},
		{
			title: 'DirectML ❤ Stable Diffusion',
			date: 'May 23th, 2023',
			blurb:
				'This blog shows how to use the Stable Diffusion model on DML EP using Olive to optimize the Stable Diffusion model',
			link: 'https://devblogs.microsoft.com/windowsai/dml-stable-diffusion/'
		},
		{
			title: 'Accelerating Stable Diffusion Inference with ONNX Runtime',
			date: 'May 10th, 2023',
			blurb:
				'This blog shows how to accelerate the Stable Diffusion models from Hugging Face on NVIDIA and AMD GPUs with ONNX Runtime. It includes benchmark results obtained on A100 and RTX3060 and MI250X.',
			link: 'https://medium.com/microsoftazure/accelerating-stable-diffusion-inference-with-onnx-runtime-203bd7728540'
		},
		{
			title: 'Azure Container for PyTorch is now Generally Available in Azure Machine Learning!',
			date: 'March 22nd, 2023',
			blurb:
				'ACPT provides a ready-to-use distributed training environment for users to run on the latest multi-node GPU infrastructure offered in Azure. With Nebula, a new fast checkpointing capability in ACPT, you can save your checkpoints 1000 times faster with a simple API that works asynchronously with your training process.',
			link: 'https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/azure-container-for-pytorch-is-now-generally-available-in-azure/ba-p/3774616'
		},
		{
			title: 'High-performance deep learning in Oracle Cloud with ONNX Runtime',
			date: 'March 15th, 2023',
			blurb:
				'Enabling scenarios through the usage of Deep Neural Network (DNN) models is critical to our AI strategy at Oracle, and our Cloud AI Services team has built a solution to serve DNN models for customers in the healthcare sector. In this blog post, we’ll share challenges our team faced, and how ONNX Runtime solves these as the backbone of success for high-performance inferencing.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/03/15/high-performance-deep-learning-in-oracle-cloud-with-onnx-runtime/'
		},
		{
			title: 'Inference Stable Diffusion with C# and ONNX Runtime',
			date: 'March 9th, 2023',
			blurb:
				'In this tutorial we will learn how to do inferencing for the popular Stable Diffusion deep learning model in C#. Stable Diffusion models take a text prompt and create an image that represents the text. ',
			link: 'https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html'
		},
		{
			title: 'Video super resolution in Microsoft Edge',
			date: 'March 8th, 2023',
			blurb:
				'VSR in Microsoft Edge builds on top of ONNX Runtime and DirectML making our solution portable across GPU vendors and allowing VSR to be available to more users. Additional graphics cards which support these technologies and have sufficient computing power will receive support in the future. The ONNX Runtime and DirectML teams have fine-tuned their technology over many years, resulting in VSR making the most of the performance and capabilities of your graphics card’s processing power.',
			link: 'https://blogs.windows.com/msedgedev/2023/03/08/video-super-resolution-in-microsoft-edge/'
		},
		{
			title:
				'OctoML drives down production AI inference costs at Microsoft through new integration with ONNX Runtime ecosystem',
			date: 'March 2nd, 2023',
			blurb:
				'Over the past year, OctoML engineers worked closely with Watch For to design and implement the TVM Execution Provider (EP) for ONNX Runtime - bringing the model optimization potential of Apache TVM to all ONNX Runtime users. This builds upon the collaboration we began in 2021, to bring the benefits of TVM’s code generation and flexible quantization support to production scale at Microsoft.',
			link: 'https://octoml.ai/blog/octoml-drives-down-costs-at-microsoft-through-new-integration-with-onnx-runtime/'
		},
		{
			title: 'Performant on-device inferencing with ONNX Runtime',
			date: 'February 8th, 2023',
			blurb:
				'On-device machine learning model serving is a difficult task, especially given the limited bandwidth of early-stage startups. This guest post from the team at Pieces shares the problems and solutions evaluated for their on-device model serving stack and how ONNX Runtime serves as their backbone of success.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/02/08/performant-on-device-inferencing-with-onnx-runtime/'
		},
		{
			title:
				'Improve BERT inference speed by combining the power of Optimum, OpenVINO™, ONNX Runtime, and Azure',
			date: 'January 25th, 2023',
			blurb:
				'In this blog, we will discuss one of the ways to make huge models like BERT smaller and faster with OpenVINO™ Neural Networks Compression Framework (NNCF) and ONNX Runtime with OpenVINO™ Execution Provider through Azure Machine Learning.',
			link: 'https://cloudblogs.microsoft.com/opensource/2023/01/25/improve-bert-inference-speed-by-combining-the-power-of-optimum-openvino-onnx-runtime-and-azure/'
		},
		{
			title: 'Optimum + ONNX Runtime: Easier, Faster training for your Hugging Face models',
			date: 'January 24th, 2023',
			blurb:
				'Hugging Face’s Optimum library, through its integration with ONNX Runtime for training, provides an open solution to improve training times by 35% or more for many popular Hugging Face models. We present details of both Hugging Face Optimum and the ONNX Runtime Training ecosystem, with performance numbers highlighting the benefits of using the Optimum library.',
			link: 'https://huggingface.co/blog/optimum-onnxruntime-training/'
		},
		{
			title: 'Live demos of machine learning models with ONNX and Hugging Face Spaces',
			date: 'June 6, 2022',
			blurb:
				'Choosing which machine learning model to use, sharing a model with a colleague, and quickly trying out a model are all reasons why you may find yourself wanting to quickly run inference on a model. You can configure your environment and download Jupyter notebooks, but it would be nicer if there was a way to run a model with even less effort...',
			link: 'https://cloudblogs.microsoft.com/opensource/2022/06/06/live-demos-of-machine-learning-models-with-onnx-and-hugging-face-spaces/'
		},
		{
			title:
				'Optimizing and deploying transformer INT8 inference with ONNX Runtime-TensorRT on NVIDIA GPUs',
			date: 'May 2, 2022',
			blurb:
				'Transformer-based models have revolutionized the natural language processing (NLP) domain. Ever since its inception, transformer architecture has been integrated into models like Bidirectional Encoder Representations from Transformers (BERT) and Generative Pre-trained Transformer (GPT) for performing tasks such as text generation or summarization and question and answering to name a few...',
			link: 'https://cloudblogs.microsoft.com/opensource/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus/'
		},
		{
			title:
				'Scaling-up PyTorch inference: Serving billions of daily NLP inferences with ONNX Runtime',
			date: 'April 19, 2022',
			blurb:
				'Scale, performance, and efficient deployment of state-of-the-art Deep Learning models are ubiquitous challenges as applied machine learning grows across the industry. We’re happy to see that the ONNX Runtime Machine Learning model inferencing solution we’ve built and use in high-volume Microsoft products and services also resonates with our open source community, enabling new capabilities that drive content relevance and productivity...',
			link: 'https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/'
		},
		{
			title: 'Add AI to mobile applications with Xamarin and ONNX Runtime',
			date: 'December 14, 2021',
			blurb:
				'ONNX Runtime now supports building mobile applications in C# with Xamarin. Support for Android and iOS is included in the ONNX Runtime release 1.10 NuGet package. This enables C# developers to build AI applications for Android and iOS to execute ONNX models on mobile devices with ONNX Runtime...',
			link: 'https://cloudblogs.microsoft.com/opensource/2021/12/14/add-ai-to-mobile-applications-with-xamarin-and-onnx-runtime/'
		},
		{
			title: 'ONNX Runtime Web—running your machine learning model in browser',
			date: 'September 2, 2021',
			blurb:
				'We are introducing ONNX Runtime Web (ORT Web), a new feature in ONNX Runtime to enable JavaScript developers to run and deploy machine learning models in browsers. It also helps enable new classes of on-device computation. ORT Web will be replacing the soon to be deprecated onnx.js...',
			link: 'https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/'
		},
		{
			title: 'Accelerate PyTorch transformer model training with ONNX Runtime – a deep dive',
			date: 'July 13, 2021',
			blurb:
				'ONNX Runtime (ORT) for PyTorch accelerates training large scale models across multiple GPUs with up to 37% increase in training throughput over PyTorch and up to 86% speed up when combined with DeepSpeed...',
			link: 'https://techcommunity.microsoft.com/t5/azure-ai/accelerate-pytorch-transformer-model-training-with-onnx-runtime/ba-p/2540471'
		},
		{
			title: 'Accelerate PyTorch training with torch-ort',
			date: 'July 13, 2021',
			blurb:
				'With a simple change to your PyTorch training script, you can now speed up training large language models with torch_ort.ORTModule, running on the target hardware of your choice. Training deep learning models requires ever-increasing compute and memory resources. Today we release torch_ort.ORTModule, to accelerate distributed training of PyTorch models, reducing the time and resources needed for training...',
			link: 'https://cloudblogs.microsoft.com/opensource/2021/07/13/accelerate-pytorch-training-with-torch-ort/'
		},
		{
			title:
				'ONNX Runtime release 1.8.1 previews support for accelerated training on AMD GPUs with the AMD ROCm™ Open Software Platform',
			date: 'July 13, 2021',
			blurb:
				'ONNX Runtime is an open-source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms. Today, we are excited to announce a preview version of ONNX Runtime in release 1.8.1 featuring support for AMD Instinct™ GPUs facilitated by the AMD ROCm™ open software platform...',
			link: 'https://cloudblogs.microsoft.com/opensource/2021/07/13/onnx-runtime-release-1-8-1-previews-support-for-accelerated-training-on-amd-gpus-with-the-amd-rocm-open-software-platform/'
		},
		{
			title: 'Journey to optimize large scale transformer model inference with ONNX Runtime',
			date: 'June 30, 2021',
			blurb:
				'Large-scale transformer models, such as GPT-2 and GPT-3, are among the most useful self-supervised transformer language models for natural language processing tasks such as language translation, question answering, passage summarization, text generation, and so on...',
			link: 'https://cloudblogs.microsoft.com/opensource/2021/06/30/journey-to-optimize-large-scale-transformer-model-inference-with-onnx-runtime/'
		}
	];
</script>

<svelte:head>
	<meta
		name="description"
		content="ONNX Runtime Blogs - your source for staying updated on the latest ONNX Runtime updated and information."
	/>
</svelte:head>
<div class="container mx-auto">
	<div class="flex">
		<h1 class="text-5xl my-auto mx-4">Blogs & Announcements</h1>
		<div class="ml-5 hidden md:flex">
			<ImageBlogs />
		</div>
	</div>
	<div class="pt-5 mx-4 md:mx-10">
		<h3 class="text-3xl pb-8">Featured posts</h3>
		<div class="grid gap-4 grid-cols-1 lg:grid-cols-3">
			{#each featuredblog as blog}
				<FeaturedBlog
					title={blog.title}
					description={blog.blurb}
					date={blog.date}
					link={blog.link}
					image={blog.image}
				/>
			{/each}
		</div>
	</div>
	<div class="mx-4 md:mx-10">
		<h3 class="text-3xl pb-8">Recent posts</h3>
		<div class="grid gap-4 grid-cols-1 md:grid-cols-2">
			{#each blogs as blog, i}
				<Blog title={blog.title} description={blog.blurb} date={blog.date} link={blog.link} />
			{/each}
		</div>
	</div>
</div>
