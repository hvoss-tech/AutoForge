import posthog from 'posthog-js'

interface TelemetryConfig {
  enabled: boolean
  apiKey: string | null
  host: string | null
  distinctId: string | null
}

let initPromise: Promise<void> | null = null

/** Fetches the server's telemetry decision and, only if it says to, loads
 * PostHog. The backend is the single source of truth for whether telemetry
 * runs at all (--no-telemetry / AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false, or
 * no project key configured) — this never inits PostHog on a guess, and
 * never falls back to "on" if the request fails. */
export function initTelemetry(): Promise<void> {
  if (initPromise) return initPromise

  initPromise = (async () => {
    try {
      const resp = await fetch('/api/system/telemetry')
      if (!resp.ok) return
      const config: TelemetryConfig = await resp.json()
      if (!config.enabled || !config.apiKey || !config.host) return

      posthog.init(config.apiKey, {
        api_host: config.host,
        capture_pageview: true,
        autocapture: false,
        persistence: 'localStorage',
        // Auto-captures uncaught window errors and unhandled promise
        // rejections as $exception events. React render errors are caught
        // separately by <ErrorBoundary> (React swallows those before they
        // reach window.onerror) and reported via captureException below.
        capture_exceptions: true,
      })

      // Use the backend's per-install anonymous id (same one
      // helpers/telemetry.py attaches to backend $exception events) as this
      // browser's distinct_id too, so a page ping and a backend error from
      // the same install group under one PostHog person instead of two
      // unrelated anonymous ids.
      if (config.distinctId) {
        posthog.identify(config.distinctId)
      }
    } catch {
      // No telemetry backend reachable, or the request failed — stay silent.
    }
  })()

  return initPromise
}

/** Safe to call from anywhere, any time: a no-op until initTelemetry() has
 * actually loaded PostHog (disabled/failed runs never call posthog.init, and
 * posthog-js itself no-ops capture() calls made before init). */
export function captureEvent(name: string, properties?: Record<string, unknown>): void {
  try {
    posthog.capture(name, properties)
  } catch {
    // Telemetry is best-effort and must never break the app.
  }
}

/** Reports a caught error as a PostHog $exception event, same as the
 * automatic window.onerror/unhandledrejection capture above. Use for errors
 * caught explicitly — a React render error in <ErrorBoundary>, a failed
 * WebSocket/fetch call — that would otherwise never reach the window. */
export function captureException(error: unknown, additionalProperties?: Record<string, unknown>): void {
  try {
    posthog.captureException(error, additionalProperties)
  } catch {
    // Telemetry is best-effort and must never break the app.
  }
}
