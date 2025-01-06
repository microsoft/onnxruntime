## vite-default

### Summary

This is a web application created by `npm create vite@latest` using the following config:

```
<ORT_ROOT>\js\web\test\e2e\exports\testcases>npm create vite@latest

> npx
> create-vite

√ Project name: ... vite-default
√ Select a framework: » Vue
√ Select a variant: » JavaScript

Scaffolding project in <ORT_ROOT>\js\web\test\e2e\exports\testcases\vite-default...

Done. Now run:

  cd vite-default
  npm install
  npm run dev
```

Small changes are made based on the application template, including:

- Add a client side rendering (CSR) component which contains:
  - a checkbox for multi-thread
  - a checkbox for proxy
  - a "Load Model" button
  - a "Run Model" button
  - a state DIV
  - a log DIV
- Add a helper module for creating ORT session and run

### Tests

Uses puppeteer to simulate the following tests:

- Tests on `npm run dev` (dev server)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
- Tests on `npm run dev -- --turbopack` (dev server using TurboPack)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
- Tests on `npm run build` + `npm run serve` (prod)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
