import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'

/** Builds the auto-preview (heightmap init) whenever there's at least one
 * active filament and the *current* input image hasn't been initialised yet —
 * not just on the rising edge of "ready":
 *  - adding a later filament, or replacing the image while filaments are
 *    already active, must still (re)build it;
 *  - toggling filaments on the same image must not: init only depends on the
 *    photo;
 *  - an image restored from the previous session whose preview (or result)
 *    the server still has is skipped (`initSkipImage`), so a page reload
 *    doesn't redo minutes of GPU work;
 *  - stepping through History back to an image whose preview the server has
 *    since dropped rebuilds it. That case is why "have I initialised this
 *    URL before?" isn't enough on its own: going A -> B -> A leaves the
 *    server holding B's preview while this hook still thinks A is done, and
 *    the panel then showed B's heightmap under A's photo.
 * Mounted once, in App. */
export function useAutoPreviewInit() {
  const activeCount = useAppStore((s) => s.activeFilaments.length)
  const inputImage = useAppStore((s) => s.inputImage)
  const initSkipImage = useAppStore((s) => s.initSkipImage)
  const initStatus = useAppStore((s) => s.initState.status)
  const hasResult = useAppStore((s) => s.currentJob?.status === 'completed')
  const runInit = useAppStore((s) => s.runInit)
  const lastInitImage = useRef<string | null>(null)
  // An init that failed (bad image, out of memory) must not be retried in a
  // loop just because it left the status back at "idle".
  const failedFor = useRef<string | null>(null)

  useEffect(() => {
    if (!inputImage) return
    if (inputImage === initSkipImage) {
      lastInitImage.current = inputImage
      return
    }
    if (activeCount === 0) return
    // A finished run supersedes the preview; don't spend GPU time rebuilding
    // something the result panel is already covering.
    if (hasResult) return
    if (initStatus === 'initializing' || initStatus === 'ready') return
    if (inputImage === lastInitImage.current && failedFor.current === inputImage) return

    lastInitImage.current = inputImage
    const image = inputImage
    runInit().then((outcome) => {
      // 'busy' is a benign race with an init already in flight — retrying is
      // exactly right there, so only a real failure blocks further attempts.
      failedFor.current = outcome === 'error' ? image : null
    })
  }, [activeCount, inputImage, initSkipImage, initStatus, hasResult, runInit])
}
