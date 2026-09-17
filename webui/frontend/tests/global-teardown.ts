import type { FullConfig } from '@playwright/test'
import { cleanupTestFilaments } from './reset-state'

// Clean up test-created filaments immediately when the run finishes, not
// just at the start of the next one — otherwise they sit in the (possibly
// real, shared) filament library between runs.
export default async function globalTeardown(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL!
  await cleanupTestFilaments(baseURL)
}
