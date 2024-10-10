# WebGPU Execution Provider

This folder is for the WebGPU execution provider(WebGPU EP). Currently, WebGPU EP is working in progress.

## Build WebGPU EP

Just append `--use_webgpu` to the `build.bat`/`build.sh` command line.

For linux, a few dependencies need to be installed:
```sh
apt-get install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libx11-dev libx11-xcb-dev
```

## Troubleshooting

TODO: add solutions to common problems.

## Development Guide

See [How to write WebGPU EP kernel](./docs/How_to_Write_WebGPU_EP_Kernel.md) for more information.

## Conventions

See [Conventions](./docs/Conventions.md) for more information.

## Best Practices

See [Best Practices](./docs/Best_Practices.md) for more information.

## TODO items

The following items are not yet implemented:

- [ ] Validation Switch (allows to change the behavior of whether perform specific validation checks)
- [ ] pushErrorScope/popErrorScope
- [ ] Graph Capture
- [ ] Profiling supported by WebGPU Query Buffer
- [ ] WebGPU resources tracking (mainly for buffers)
- [ ] Global hanlders( unhandled exceptions and device lost )
