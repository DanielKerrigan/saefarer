import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import anywidget from "@anywidget/vite";

export default defineConfig({
  build: {
    lib: {
      entry: ["js/widget.ts"],
      formats: ["es"],
    },
    outDir: "src/saefarer/static/",
  },
  plugins: [anywidget(), svelte({ hot: false })],
});
