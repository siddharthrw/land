/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MISSION_WS_URL?: string;
  readonly VITE_MISSION_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
