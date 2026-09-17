import { useCallback, useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { chooseActiveTab } from '../lib/library'

const MAX_RETRIES = 10
const RETRY_DELAY = 2000 // ms

async function fetchWithRetry(url: string, retries = MAX_RETRIES): Promise<Response> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(url)
      if (resp.ok) return resp
    } catch {
      // Backend not ready yet
    }
    if (attempt < retries - 1) {
      await new Promise(r => setTimeout(r, RETRY_DELAY))
    }
  }
  throw new Error(`Backend unreachable: ${url}`)
}

// Every library reload bumps this; a response that arrives after a newer
// request was sent is dropped. Typing in the search box fires one request per
// keystroke, and an older, slower response used to land last and show results
// for a query the user had already changed.
let filamentsRequestSeq = 0

export async function loadFilaments() {
  const seq = ++filamentsRequestSeq
  try {
    const state = useAppStore.getState()
    const params = new URLSearchParams()
    if (state.activeTab) params.set('filament_type', state.activeTab)
    if (state.filterBrand) params.set('brand', state.filterBrand)
    if (state.filterQuery) params.set('query', state.filterQuery)

    const response = await fetchWithRetry(`/api/filaments?${params.toString()}`, 1)
    const filaments: Filament[] = await response.json()
    if (seq !== filamentsRequestSeq) return
    useAppStore.getState().setFilaments(filaments)
  } catch {
    // Backend not ready or unreachable — keep current state
  }
}

export async function loadFilamentTypes() {
  try {
    const response = await fetchWithRetry('/api/filaments/types', 1)
    const types: string[] = await response.json()
    const store = useAppStore.getState()
    store.setFilamentTypes(types)
    const tab = chooseActiveTab(types, store.activeTab)
    if (tab !== store.activeTab) store.setActiveTab(tab)
  } catch {
    // Backend not ready or unreachable — keep current state
  }
}

/** Reload everything the library panel shows (tabs, the active tab's list,
 * brands) after the library changed. The create/edit/import dialogs each did
 * their own partial refetch: editing reloaded *every* type into the current
 * tab, creating showed the new filament's type under the old tab, and
 * imports never refreshed the tabs, so imported types only appeared after a
 * page reload. `showType` switches to that tab first, so a filament you just
 * created is actually visible. */
export async function refreshLibrary(showType?: string) {
  await loadFilamentTypes()
  if (showType && useAppStore.getState().filamentTypes.includes(showType)) {
    useAppStore.getState().setActiveTab(showType)
  }
  await loadFilaments()
  await loadFilamentBrands()
}

export async function loadFilamentBrands() {
  try {
    const state = useAppStore.getState()
    const params = new URLSearchParams()
    if (state.activeTab) params.set('filament_type', state.activeTab)
    const response = await fetchWithRetry(`/api/filaments/brands?${params.toString()}`, 1)
    const brands: string[] = await response.json()
    useAppStore.getState().setFilamentBrands(brands)
  } catch {
    // Backend unreachable
  }
}

export function useFilamentLoader() {
  const activeTab = useAppStore((s) => s.activeTab)
  const filterBrand = useAppStore((s) => s.filterBrand)
  const filterQuery = useAppStore((s) => s.filterQuery)
  const initialLoadDone = useRef(false)

  const load = useCallback(async () => {
    // On initial load, retry until backend responds
    const retries = initialLoadDone.current ? 1 : 10
    initialLoadDone.current = true

    try {
      const response = await fetchWithRetry('/api/filaments/types', retries)
      const types: string[] = await response.json()
      const store = useAppStore.getState()
      store.setFilamentTypes(types)
      const tab = chooseActiveTab(types, store.activeTab)
      if (tab !== store.activeTab) store.setActiveTab(tab)
    } catch {
      // Backend not ready or unreachable — keep current state
    }

    try {
      await loadFilaments()
      await loadFilamentBrands()
    } catch {
      // Ignore
    }
  }, [])

  useEffect(() => {
    load()
  }, [])

  // Fires on every activeTab/filterBrand/filterQuery change to refetch with
  // the new filters — but those deps also "change" (from undefined to their
  // initial values) on the very first render, which duplicated `load()`'s
  // own loadFilaments()/loadFilamentBrands() calls above with an identical,
  // redundant pair of requests racing them for the same connections. A page
  // load already fires a dozen-plus requests inside its first second (every
  // panel's own on-mount fetch, the project/filament/history loads, the
  // preview WebSocket handshake); HTTP/1.1 caps a browser at 6 concurrent
  // connections per origin, so any request issued shortly after — e.g. a
  // user importing a filament file right away — could queue for seconds
  // behind this avoidable overflow instead of running immediately.
  const isFirstFilterEffect = useRef(true)
  useEffect(() => {
    if (isFirstFilterEffect.current) {
      isFirstFilterEffect.current = false
      return
    }
    loadFilaments()
    loadFilamentBrands()
  }, [activeTab, filterBrand, filterQuery])
}
