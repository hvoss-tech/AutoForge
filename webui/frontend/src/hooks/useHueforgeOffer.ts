import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'

/** On the first start, if HueForge's personal filament library is on this
 * computer, open the import dialog with it highlighted. Made once per
 * install (the server remembers it was offered), and only after the
 * first-run tutorial is closed so the two don't open on top of each other. */
export function useHueforgeOffer(): void {
  const tutorialOpen = useAppStore((s) => s.tutorialOpen)
  const checked = useRef(false)

  useEffect(() => {
    if (tutorialOpen || checked.current) return
    checked.current = true
    ;(async () => {
      try {
        const response = await fetch('/api/filaments/hueforge-library')
        if (!response.ok) return
        const info = await response.json()
        if (!info?.found || info.offered) return
        await fetch('/api/filaments/hueforge-library/offered', { method: 'POST' })
        const store = useAppStore.getState()
        // Don't take over a dialog the user already opened themselves.
        if (!store.importModalOpen && !store.settingsModalOpen && !store.tutorialOpen) store.openHueforgeImportOffer()
      } catch {
        // No offer — the Import button still shows the option.
      }
    })()
  }, [tutorialOpen])
}
