const config = {
  use: {
    channel: "chromium",
    launchOptions: {
        args: [
            '--enable-unsafe-webgpu',
            '--ignore-gpu-blocklist',
			"--gpu-vendor-id=0x10de"
        ],
        headless: true
    }
  }
}
module.exports = config