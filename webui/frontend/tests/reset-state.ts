export const DEFAULT_SLIDERS = [
  { td: 2.0, layer: 8, depth_mm: 0.32, filament_uuid: '', enabled: true },
  { td: 3.0, layer: 13, depth_mm: 0.52, filament_uuid: '', enabled: true },
  { td: 8.0, layer: 20, depth_mm: 0.8, filament_uuid: '', enabled: true },
  { td: 5.0, layer: 27, depth_mm: 1.08, filament_uuid: '', enabled: true },
  ...Array.from({ length: 11 }, () => ({ td: 5.0, layer: 0, depth_mm: 0.0, filament_uuid: '', enabled: false })),
]

/** Clear the *real* active-filaments list (FilamentService's `_active` map,
 * served at GET/DELETE /api/filaments/active) — a completely separate store
 * from the `active_filaments` field POSTed to /api/project/state below.
 * resetProjectState() only ever touched the latter, so any filament
 * activated through the UI (the library's +/toggle button, drag-onto-slider,
 * ...) stayed active across tests/runs regardless of that reset — e.g. a
 * test that activates the first library filament left it active for every
 * later test in the same run, silently inflating "active filaments" counts. */
export async function resetActiveFilaments(baseURL: string): Promise<void> {
  try {
    const res = await fetch(`${baseURL}/api/filaments/active`)
    if (res.ok) {
      const active: { uuid: string }[] = await res.json()
      for (const f of active) {
        await fetch(`${baseURL}/api/filaments/active/${f.uuid}`, { method: 'DELETE' }).catch(() => {})
      }
    }
  } catch {
    // Backend may not be reachable yet in some setups; tests will surface that.
  }
}

/** Reset server-side project state (color sliders / settings) to the
 * documented defaults, and the real active-filaments list alongside it (see
 * resetActiveFilaments — these are two separate backend stores that both
 * need clearing). The backend persists this across reloads, so without a
 * reset, mutations from one test (or one run) bleed into the next.
 * `settings: {}` resolves to OptimizationSettings' own field defaults
 * (layer_height, background_height, stl_output_size, ...) via Pydantic. */
export async function resetProjectState(baseURL: string): Promise<void> {
  await fetch(`${baseURL}/api/project/state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      color_sliders: DEFAULT_SLIDERS,
      settings: {},
      active_filaments: [],
    }),
  }).catch(() => {})
  await resetActiveFilaments(baseURL)
  await fetch(`${baseURL}/api/state/history`, { method: 'DELETE' }).catch(() => {})
}

/** Remove filaments created by test runs so brand/tab lists and counts stay
 * deterministic across runs, and the (possibly real, shared) filament
 * library doesn't accumulate test data. Matches by substring rather than an
 * exact brand list — the suite uses several ad hoc test-only brand names
 * (E2E, E2ETest, TabTest, DeleteE2E, UpdateE2E, MarkTest, PersistTest,
 * TestBrand, ...) and new ones will keep appearing; every one of them
 * contains "test" or "e2e". */
export async function cleanupTestFilaments(baseURL: string): Promise<void> {
  const isTestBrand = (brand: string) => {
    const b = (brand || '').toLowerCase()
    return b.includes('test') || b.includes('e2e')
  }
  try {
    const res = await fetch(`${baseURL}/api/filaments`)
    if (res.ok) {
      const all = await res.json()
      for (const f of all) {
        if (isTestBrand(f.brand)) {
          await fetch(`${baseURL}/api/filaments/${f.uuid}`, { method: 'DELETE' }).catch(() => {})
          await fetch(`${baseURL}/api/filaments/active/${f.uuid}`, { method: 'DELETE' }).catch(() => {})
        }
      }
    }
  } catch {
    // Backend may not be reachable yet in some setups; tests will surface that.
  }
}
