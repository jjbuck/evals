import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // development only: lets ?devroot=<absolute path to a studies folder> read records through the dev server
  server: { fs: { allow: [".."] } },
});
