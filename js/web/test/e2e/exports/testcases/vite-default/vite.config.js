import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// https://vite.dev/config/
export default defineConfig({
  // This is a known issue when using WebAssembly with Vite 5.x
  // Need to specify `optimizeDeps.exclude` to NPM packages that uses WebAssembly
  // See: https://github.com/vitejs/vite/issues/8427
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  plugins: [vue()],
});
