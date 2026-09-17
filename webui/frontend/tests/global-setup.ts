import type { FullConfig } from '@playwright/test'
import { resetProjectState, cleanupTestFilaments } from './reset-state'

// The backend persists project state and the filament library to disk and
// keeps running across test runs (there is no per-run server/data reset in
// this project's dev workflow). Without this, slider/filament mutations from
// one run bleed into the next: a previous run leaves e.g. all color sliders
// disabled in project_state.json, or accumulates duplicate "E2E"/"TabTest"
// filaments in the library, and later runs silently inherit that state
// instead of the documented defaults.
export default async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL!
  await resetProjectState(baseURL)
  await cleanupTestFilaments(baseURL)
}
