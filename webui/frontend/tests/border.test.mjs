// Test runner: node --test tests/border.test.mjs
//
// Borders around the exported model (settings.border, lib/border). The 3D
// view and the downloads are covered by tests/features/border-2026-10-16.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_BORDER,
  borderActive,
  decodeBorderMesh,
  describeBorder,
  formatPrintSize,
  maxCornerRadius,
  normalizeBorder,
} from '../src/lib/border.ts'
import { staleReasons } from '../src/lib/staleResult.ts'

globalThis.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {} }

test('settings and project files without a border mean no border', () => {
  assert.deepEqual(normalizeBorder(undefined), DEFAULT_BORDER)
  assert.deepEqual(normalizeBorder(null), DEFAULT_BORDER)
  assert.equal(borderActive(normalizeBorder({})), false)
})

test('odd values are clamped or dropped, never kept', () => {
  const b = normalizeBorder({ shape: 'star', width: 0, corner_radius: -2, hanger: 'left', hanger_diameter: 'big' })
  assert.deepEqual(b, { shape: 'none', width: 1, corner_radius: 0, hanger: 'none', hanger_diameter: 4 })
  assert.equal(normalizeBorder({ shape: 'oval', width: 500 }).width, 100)
  assert.equal(normalizeBorder({ shape: 'oval', width: '7.5' }).width, 7.5)
})

test('only a shape makes a border; a hanging hole alone does not', () => {
  assert.equal(borderActive({ ...DEFAULT_BORDER, hanger: 'top' }), false)
  assert.equal(borderActive({ ...DEFAULT_BORDER, shape: 'outline' }), true)
})

test('the description names what will be printed', () => {
  assert.equal(describeBorder(DEFAULT_BORDER), null)
  assert.equal(describeBorder({ ...DEFAULT_BORDER, shape: 'rectangle', corner_radius: 0 }), 'Rectangle 5 mm')
  assert.equal(
    describeBorder({ ...DEFAULT_BORDER, shape: 'rectangle', width: 2, corner_radius: 50, hanger: 'corners' }),
    // The radius the backend actually uses at this width.
    `Rectangle 2 mm, rounded ${maxCornerRadius(2)} mm, 2 hanging holes`,
  )
  // A corner radius only applies to rectangles.
  assert.equal(describeBorder({ ...DEFAULT_BORDER, shape: 'oval', hanger: 'top' }), 'Oval 5 mm, hanging hole')
})

test('changing the border does not make the result out of date', () => {
  const run = { settings: { border: DEFAULT_BORDER, iterations: 100 }, filamentUuids: [] }
  assert.deepEqual(staleReasons(run, { border: { ...DEFAULT_BORDER, shape: 'circle' }, iterations: 100 }, []), [])
})

test('print size reads as whole millimetres', () => {
  assert.equal(formatPrintSize([160.4, 311.6]), '160 × 312 mm')
  assert.equal(formatPrintSize(null), null)
})

function b64(typed) {
  return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64')
}

test('the preview mesh decodes, and broken data is refused', () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])
  const indices = new Uint32Array([0, 1, 2])
  const mesh = decodeBorderMesh({ positions: b64(positions), indices: b64(indices) })
  assert.deepEqual(Array.from(mesh.positions), Array.from(positions))
  assert.deepEqual(Array.from(mesh.indices), [0, 1, 2])
  assert.equal(decodeBorderMesh({ positions: '', indices: '' }), null)
  // A triangle pointing past the vertices would crash the renderer.
  assert.equal(decodeBorderMesh({ positions: b64(positions), indices: b64(new Uint32Array([0, 1, 3])) }), null)
})
