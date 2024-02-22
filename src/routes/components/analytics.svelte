<script>
	import '@beyonk/gdpr-cookie-consent-banner/banner.css'; // optional, you can also define your own styles
	import GdprBanner from '@beyonk/gdpr-cookie-consent-banner';

	function initAnalytics() {
		// Google Analytics
		gtag('consent', 'update', {
			ad_storage: 'granted',
			analytics_storage: 'granted'
		});
		// Adobe Analytics
	}
</script>

<GdprBanner cookieName="foo" description="bar" on:analytics={initAnalytics} />

<svelte:head>
	<!-- Google Analytics -->
	<script async src="https://www.googletagmanager.com/gtag/js?id=UA-156955408-1"></script>
	<script>
		window.dataLayer = window.dataLayer || [];
		function gtag() {
			dataLayer.push(arguments);
		}
		gtag('consent', 'default', {
			ad_storage: 'denied',
			analytics_storage: 'denied'
		});
		gtag('js', new Date());
		gtag('config', 'UA-156955408-1');
	</script>
	<!-- Adobe Analytics -->
	<script type="text/javascript">
		const analytics = new oneDS.ApplicationInsights();
		var config = {
			instrumentationKey:
				'360b0e675e0044398fd28c8bdf711b8e-1fe5434d-ee99-4837-99cc-a3a16462d82d-7262',
			channelConfiguration: {
				// Post channel configuration
				eventsLimitInMem: 50
			},
			propertyConfiguration: {
				// Properties Plugin configuration
				env: 'PROD' // Environment can be set to PPE or PROD as needed.
			},
			webAnalyticsConfiguration: {
				// Web Analytics Plugin configuration
				//urlCollectQuery:true,
				autoCapture: {
					scroll: true,
					pageView: true,
					onLoad: true,
					onUnload: true,
					click: true,
					resize: true,
					jsError: true
				}
			}
		};
		//Initialize SDK
		analytics.initialize(config, []);
	</script>
	<!-- MS Analytics -->
	<script
		type="text/javascript"
		src="https://js.monitor.azure.com/scripts/c/ms.analytics-web-3.min.js"
	>
	</script>

	
</svelte:head>
