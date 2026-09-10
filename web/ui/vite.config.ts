import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const projectRoot = new URL(".", import.meta.url).pathname;

export default defineConfig({
  plugins: [react()],
  root: projectRoot,
  build: {
    outDir: new URL("dist", import.meta.url).pathname,
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api/runs": "http://127.0.0.1:8000",
      "/api/reports": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000"
    }
  }
});
