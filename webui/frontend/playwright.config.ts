import { defineConfig, devices } from '@playwright/test'

// By default the suite starts its own backend with throwaway data
// (tests/test-server.mjs) instead of touching a real server on :8000. Set
// WEBUI_TEST_BASE_URL to run against an already-running server instead.
// The backend serves webui/frontend/dist, so run `npm run build` first
// (`npm test` does both).
const externalBaseURL = process.env.WEBUI_TEST_BASE_URL
const testPort = process.env.WEBUI_TEST_PORT || '8799'
const baseURL = externalBaseURL || `http://127.0.0.1:${testPort}`

export default defineConfig({
  testDir: './tests',
  globalSetup: './tests/global-setup.ts',
  globalTeardown: './tests/global-teardown.ts',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // Must stay 1: every test hits one shared, stateful backend (project
  // state, filament library, job history) with no per-worker isolation, so
  // two spec files running concurrently in separate workers can corrupt
  // each other's state (e.g. a slider mutation from one file's job landing
  // mid-assertion in another file's "defaults" test).
  workers: 1,
  reporter: 'html',
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  webServer: externalBaseURL
    ? undefined
    : {
        command: 'node tests/test-server.mjs',
        url: `${baseURL}/api/system/health`,
        timeout: 120_000,
        reuseExistingServer: false,
        env: { WEBUI_TEST_PORT: testPort },
      },
  // `jobs` run real optimizations and leave finished results behind (which
  // the app restores on load), so they go after everything else.
  projects: [
    {
      name: 'chromium',
      testIgnore: /jobs\//,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'jobs',
      testMatch: /jobs\/.*\.spec\.ts/,
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
