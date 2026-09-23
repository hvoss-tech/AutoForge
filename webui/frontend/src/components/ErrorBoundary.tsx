import React from 'react'
import { captureException } from '../lib/telemetry'

interface Props {
  children: React.ReactNode
}

interface State {
  error: Error | null
}

/** React render errors never reach window.onerror — React swallows them at
 * the boundary that catches them (or, with no boundary, at the root render
 * call) instead of letting them bubble as an uncaught exception. Without
 * this, a render crash would silently blank the page and never reach
 * PostHog's automatic capture_exceptions. */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    captureException(error, { componentStack: info.componentStack })
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen w-screen flex-col items-center justify-center gap-4 bg-background p-8 text-center text-foreground">
          <h1 className="text-lg font-semibold">Something went wrong</h1>
          <p className="max-w-md text-sm text-muted-foreground">
            AutoForge hit an unexpected error and can't continue. Reloading usually fixes it;
            your project state is saved on the server.
          </p>
          <button
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
            onClick={() => window.location.reload()}
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
