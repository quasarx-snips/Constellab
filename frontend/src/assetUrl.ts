/**
 * Resolve a file published from `frontend/public` to a URL the page can load.
 *
 * The Vite dev server serves those files from the site root, but the packaged
 * desktop app opens `frontend/dist/index.html` over file://, where a
 * root-absolute reference such as "/earth-tactical-dark.png" means
 * file:///C:/earth-tactical-dark.png (the drive root) and fails with
 * ERR_FILE_NOT_FOUND. Resolving against the document's own base URL gives
 * http://localhost:5173/... in development and file:///.../dist/... once built.
 */
export function publicAssetUrl(file: string): string {
  const relative = file.replace(/^\/+/, "");
  if (typeof document === "undefined") return `/${relative}`;
  return new URL(relative, document.baseURI).href;
}
