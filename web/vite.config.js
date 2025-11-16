// ABOUTME: Vite configuration for nuPlan real-time visualization dashboard
// ABOUTME: Configures React Fast Refresh, dev server, and build settings

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: 'localhost',
    open: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
