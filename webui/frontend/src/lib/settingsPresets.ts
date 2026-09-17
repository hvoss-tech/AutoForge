export interface QualityPreset {
  id: 'fast' | 'balanced' | 'best'
  label: string
  iterations: number
  description: string
}

export const QUALITY_PRESETS: QualityPreset[] = [
  { id: 'fast', label: 'Fast', iterations: 2000, description: 'A quick draft to check colors and composition.' },
  { id: 'balanced', label: 'Balanced', iterations: 6000, description: 'Good results for most pictures (default).' },
  { id: 'best', label: 'Best', iterations: 15000, description: 'Slowest; worth it for detailed pictures you are about to print.' },
]

export function presetForIterations(iterations: number): QualityPreset['id'] | 'custom' {
  return QUALITY_PRESETS.find((p) => p.iterations === iterations)?.id ?? 'custom'
}
