import path from 'node:path'
import { fileURLToPath } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const projectRoot = path.dirname(fileURLToPath(import.meta.url))

// https://vite.dev/config/
export default defineConfig(() => {
  return {
    root: projectRoot,
    cacheDir: path.join(projectRoot, 'node_modules', '.vite'),
    server: {
      port: 5181,
      host: '0.0.0.0',
    },
    resolve: {
      alias: [
        {
          find: /^@\//,
          replacement: `${path.join(projectRoot, 'src')}${path.sep}`,
        },
      ],
      preserveSymlinks: true,
    },
    build: {
      outDir: path.join(projectRoot, 'dist'),
      emptyOutDir: true,
    },
    plugins: [react()],
  }
})
