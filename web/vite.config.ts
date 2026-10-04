import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// One self-contained HTML file: works offline, from a file:// URL, or as an Artifact page.
export default defineConfig({
  plugins: [preact(), viteSingleFile()],
  build: { outDir: 'dist', assetsInlineLimit: 100_000_000 },
})
