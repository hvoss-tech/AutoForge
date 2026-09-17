export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  if (!result) return null
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  }
}

export function luminance(hex: string): number {
  const rgb = hexToRgb(hex)
  if (!rgb) return 0
  const srgb = [rgb.r, rgb.g, rgb.b].map((c) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
  })
  return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2]
}

/** Black or white, whichever has the higher WCAG contrast ratio against the
 * color. The old "luminance > 0.5" cut-off put white text on mid-tones like
 * green, orange or silver, where it was barely readable. */
export function contrastTextColor(hex: string): string {
  const rgb = hexToRgb(hex)
  if (!rgb) return '#ffffff'
  const channel = (v: number) => {
    const c = v / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  const L = 0.2126 * channel(rgb.r) + 0.7152 * channel(rgb.g) + 0.0722 * channel(rgb.b)
  const onBlack = (L + 0.05) / 0.05
  const onWhite = 1.05 / (L + 0.05)
  return onBlack >= onWhite ? '#000000' : '#ffffff'
}

export function hexToTdOverlay(td: number, maxTd: number = 20): string {
  const alpha = Math.min(td / maxTd, 1)
  return `rgba(255, 255, 255, ${alpha * 0.5})`
}

export function lerpColor(a: string, b: string, t: number): string {
  const ar = hexToRgb(a)
  const br = hexToRgb(b)
  if (!ar || !br) return a
  const r1 = Math.round(ar.r + (br.r - ar.r) * t)
  const g1 = Math.round(ar.g + (br.g - ar.g) * t)
  const b1 = Math.round(ar.b + (br.b - ar.b) * t)
  return `#${((1 << 24) | (r1 << 16) | (g1 << 8) | b1).toString(16).slice(1)}`
}

export function normalizeHex(hex: string): string {
  return hex.startsWith('#') ? hex.toUpperCase() : `#${hex.toUpperCase()}`
}

export function luminanceContrast(a: string, b: string): number {
  const la = luminance(a)
  const lb = luminance(b)
  const lighter = Math.max(la, lb)
  const darker = Math.min(la, lb)
  return (lighter + 0.05) / (darker + 0.05)
}
