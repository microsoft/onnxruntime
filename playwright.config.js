const config = {
  use: {
    channel: "chromium",
    launchOptions: {
        args: [
            '--enable-unsafe-webgpu',
			"--gpu-vendor-id=0x10de"
        ],
        headless: false
    }
  }
}

module.exports = config