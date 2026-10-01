/** How long a download's object URL is kept: revoking it right after
 * click() can cancel a large download before the browser has read it. */
export const DOWNLOAD_URL_LIFETIME_MS = 60_000

/** Save ``blob`` as ``filename`` through a temporary link. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_LIFETIME_MS)
}

/** Release ``previous`` when it is a blob: URL being replaced by another
 * value (an uploaded image's local preview holds the whole file in memory
 * for as long as its URL lives). */
export function releaseReplacedObjectUrl(previous: string | null | undefined, next: string | null | undefined): void {
  if (previous && previous.startsWith('blob:') && previous !== next) URL.revokeObjectURL(previous)
}
