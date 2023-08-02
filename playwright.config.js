const config = {
  use: {
    channel: "chromium",
    launchOptions: {
        args: [
            '--enable-unsafe-webgpu',
            '--enable-gpu',
            '--use-angle=egl-angle',
            '--use-gl=d3d11',
            '--ignore-gpu-blocklist',
			"--gpu-vendor-id=0x10de"
        ],
        headless: true
    }
  }
}
module.exports = config