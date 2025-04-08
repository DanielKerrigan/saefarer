import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import anywidget from "@anywidget/vite";

export default defineConfig({
  server: {
    watch: {
      ignored: [
        "**/src/**",
        "**/examples/**",
        "**/docs/**",
        "**/dist/**",
        "**/.vscode/**",
        "**/.pytest_cache/**",
        "**/.pixi/**",
        "**/.github/**",
      ],
    },
  },
  build: {
    lib: {
      entry: ["js/widget.ts"],
      formats: ["es"],
    },
    outDir: "src/saefarer/static/",
  },
  plugins: [
    anywidget(),
    svelte({
      compilerOptions: {
        hmr: false,
      },
    }),
  ],
});
