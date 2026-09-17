export interface NumberRules {
  integer?: boolean
  min?: number
  max?: number
}

/** The number a (possibly half-typed) input string should commit, or null
 * while it isn't a valid, in-range value yet.
 *
 * Number fields used to parse and coerce on every keystroke
 * (`parseFloat(v) || 0.01`), so clearing a field and typing "0.08" went
 * "0" → coerced to 0.01 → "0.018": most values simply couldn't be typed.
 * Out-of-range drafts aren't committed either (typing "150" into a 1-200
 * field passes through "1" and "15", which are fine, but "0" isn't). */
export function parseNumberDraft(raw: string, rules: NumberRules = {}): number | null {
  const text = raw.trim()
  if (text === '' || text === '-' || text === '.' || text === '-.') return null
  const n = Number(text)
  if (!Number.isFinite(n)) return null
  if (rules.integer && !Number.isInteger(n)) return null
  if (rules.min !== undefined && n < rules.min) return null
  if (rules.max !== undefined && n > rules.max) return null
  return n
}

/** What the field settles on when it loses focus: the typed value clamped
 * into range, or the last committed value if the draft isn't a number. */
export function settleNumberDraft(raw: string, lastValid: number, rules: NumberRules = {}): number {
  const n = Number(raw.trim())
  if (raw.trim() === '' || !Number.isFinite(n)) return lastValid
  let v = rules.integer ? Math.round(n) : n
  if (rules.min !== undefined) v = Math.max(rules.min, v)
  if (rules.max !== undefined) v = Math.min(rules.max, v)
  return v
}
