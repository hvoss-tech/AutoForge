/** Pick the tab to show for a set of library types: keep the current one if
 * it still exists, otherwise the first available. The store starts on "PLA",
 * so a library without PLA filaments used to open on an empty tab. */
export function chooseActiveTab(types: string[], current: string): string {
  if (types.length === 0 || types.includes(current)) return current
  return types[0]
}

export type FilamentSort = 'name' | 'color' | 'td'

function hue(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return [999, 0, 0]
  const n = parseInt(m[1], 16)
  const r = ((n >> 16) & 255) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min
  const l = (max + min) / 2
  // Grays (black/white/silver) sort first, by lightness, before the hue wheel.
  if (d < 0.08) return [-1, l, 0]
  let h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4
  h = (h * 60 + 360) % 360
  return [h, l, d]
}

export function sortFilaments<T extends { name: string; color: string; td: number }>(items: T[], sort: FilamentSort): T[] {
  const copy = [...items]
  if (sort === 'td') return copy.sort((a, b) => a.td - b.td || a.name.localeCompare(b.name))
  if (sort === 'color') {
    return copy.sort((a, b) => {
      const [ha, la] = hue(a.color)
      const [hb, lb] = hue(b.color)
      return ha - hb || la - lb || a.name.localeCompare(b.name)
    })
  }
  return copy.sort((a, b) => a.name.localeCompare(b.name))
}
