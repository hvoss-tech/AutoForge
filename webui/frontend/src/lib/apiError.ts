/** Human-readable message for a FastAPI error response body.
 *
 * `detail` is a string for errors the API raises itself, but a list of
 * `{loc, msg}` objects for request validation failures (422) — shown raw,
 * that became "[object Object],[object Object]". */
export function describeApiError(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => {
        const field = (d.loc ?? []).filter((p) => p !== 'body' && typeof p === 'string').join('.')
        const msg = d.msg ?? 'invalid value'
        return field ? `${field.replace(/_/g, ' ')}: ${msg}` : msg
      })
      .join('; ')
  }
  return `HTTP ${status}`
}
