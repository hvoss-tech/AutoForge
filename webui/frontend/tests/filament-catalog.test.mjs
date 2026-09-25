// Test runner: node --test tests/filament-catalog.test.mjs
// Search, filtering and sorting behind the filamentcolors.xyz catalog dialog.

import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  COLOR_FAMILIES,
  EMPTY_FILTER,
  activeFilterCount,
  catalogUuid,
  deltaE,
  describeDeltaE,
  facetCounts,
  filterCatalog,
  hexToLab,
  isInLibrary,
  matchesQuery,
  sortCatalog,
  tokenize,
} from '../src/lib/catalog.ts'

const entry = (id, over = {}) => ({
  id,
  brand: 'Acme',
  name: 'Red',
  color_name: 'Red',
  filament_type: 'PLA',
  type_detail: 'PLA',
  color: '#cc0000',
  td: 2,
  color_family: 'Red',
  available: true,
  published: '2026-01-01',
  url: `https://filamentcolors.xyz/swatch/${id}/`,
  ...over,
})

const ENTRIES = [
  entry(1, { brand: 'Bambu Lab', name: 'Jade White', color_name: 'Jade White', color: '#f0f0ea', td: 5, color_family: 'White', published: '2026-03-01' }),
  entry(2, { brand: 'Bambu Lab', name: 'Black', color_name: 'Black', color: '#101010', td: 0.4, color_family: 'Black', filament_type: 'PETG', type_detail: 'PETG HF', published: '2026-05-01' }),
  entry(3, { brand: 'Polymaker', name: 'Galaxy Blue (Panchroma Starlight)', color_name: 'Galaxy Blue', type_detail: 'Panchroma Starlight', color: '#1a3cc8', td: 1.8, color_family: 'Blue', published: '2026-07-05' }),
  entry(4, { brand: 'Elegoo', name: 'Red', color: '#d01010', td: 1.2, published: '2025-11-01' }),
]

test('tokenize lowercases, splits on whitespace and drops a leading #', () => {
  assert.deepEqual(tokenize('  Bambu   #F0F0EA '), ['bambu', 'f0f0ea'])
  assert.deepEqual(tokenize(''), [])
})

test('every word must match, in any order and field', () => {
  assert.ok(matchesQuery(ENTRIES[0], tokenize('white bambu')))
  assert.ok(matchesQuery(ENTRIES[0], tokenize('jade')))
  assert.ok(matchesQuery(ENTRIES[2], tokenize('starlight blue')), 'product line is searchable')
  assert.ok(matchesQuery(ENTRIES[1], tokenize('petg hf')), 'type detail is searchable')
  assert.ok(matchesQuery(ENTRIES[1], tokenize('#101010')), 'hex is searchable')
  assert.ok(matchesQuery(ENTRIES[2], tokenize('blue')), 'color family is searchable')
  assert.ok(!matchesQuery(ENTRIES[0], tokenize('jade black')))
  assert.ok(matchesQuery(ENTRIES[0], []))
})

test('filterCatalog combines type, brand, family, library and query', () => {
  const ids = (f, lib) => filterCatalog(ENTRIES, { ...EMPTY_FILTER, ...f }, lib).map((e) => e.id)
  assert.deepEqual(ids({}), [1, 2, 3, 4])
  assert.deepEqual(ids({ filamentType: 'PETG' }), [2])
  assert.deepEqual(ids({ brand: 'Bambu Lab' }), [1, 2])
  assert.deepEqual(ids({ family: 'Red' }), [4])
  assert.deepEqual(ids({ brand: 'Bambu Lab', query: 'white' }), [1])
  assert.deepEqual(ids({ hideInLibrary: true }, { 1: 'filamentcolors-1', 3: 'mine' }), [2, 4])
  assert.deepEqual(ids({ hideInLibrary: false }, { 1: 'filamentcolors-1' }), [1, 2, 3, 4])
})

test('isInLibrary reads the server map by swatch id', () => {
  assert.ok(isInLibrary(ENTRIES[0], { 1: 'x' }))
  assert.ok(!isInLibrary(ENTRIES[0], { 2: 'x' }))
  assert.ok(!isInLibrary(ENTRIES[0], {}))
})

test('sorts: brand, name, td, newest', () => {
  const ids = (s) => sortCatalog(ENTRIES, s).map((e) => e.id)
  assert.deepEqual(ids('brand'), [2, 1, 4, 3])
  assert.deepEqual(ids('name'), [2, 3, 1, 4])
  assert.deepEqual(ids('td'), [2, 4, 3, 1])
  assert.deepEqual(ids('newest'), [3, 2, 1, 4])
  assert.deepEqual(ENTRIES.map((e) => e.id), [1, 2, 3, 4], 'input is not mutated')
})

test('closest sort ranks by color distance, and falls back without a color', () => {
  assert.deepEqual(sortCatalog(ENTRIES, 'closest', '#0000ff').map((e) => e.id)[0], 3)
  assert.deepEqual(sortCatalog(ENTRIES, 'closest', '#ff0000').map((e) => e.id)[0], 4)
  assert.deepEqual(sortCatalog(ENTRIES, 'closest', '#ffffff').map((e) => e.id)[0], 1)
  assert.deepEqual(sortCatalog(ENTRIES, 'closest', '').map((e) => e.id), sortCatalog(ENTRIES, 'brand').map((e) => e.id))
})

test('Lab conversion and deltaE', () => {
  const [L, a, b] = hexToLab('#ffffff')
  assert.ok(Math.abs(L - 100) < 0.01 && Math.abs(a) < 0.01 && Math.abs(b) < 0.01)
  assert.ok(Math.abs(hexToLab('#000000')[0]) < 0.01)
  assert.equal(hexToLab('nope'), null)
  assert.equal(deltaE('#123456', '#123456'), 0)
  assert.ok(deltaE('#000000', '#ffffff') > 99)
  assert.equal(deltaE('bad', '#ffffff'), Infinity)
  assert.ok(deltaE('#ff0000', '#ee0000') < deltaE('#ff0000', '#00ff00'))
})

test('describeDeltaE puts the number into words', () => {
  assert.equal(describeDeltaE(1), 'Near identical')
  assert.equal(describeDeltaE(4), 'Very close')
  assert.equal(describeDeltaE(10), 'Close')
  assert.equal(describeDeltaE(20), 'Similar')
  assert.equal(describeDeltaE(50), 'Different')
})

test('facetCounts lists values most common first', () => {
  assert.deepEqual(facetCounts(ENTRIES, 'filament_type'), [{ value: 'PLA', count: 3 }, { value: 'PETG', count: 1 }])
  assert.deepEqual(facetCounts(ENTRIES, 'brand')[0], { value: 'Bambu Lab', count: 2 })
  assert.deepEqual(facetCounts([], 'brand'), [])
})

test('activeFilterCount ignores the remembered hide-in-library preference', () => {
  assert.equal(activeFilterCount(EMPTY_FILTER), 0)
  assert.equal(activeFilterCount({ ...EMPTY_FILTER, hideInLibrary: true }), 0)
  assert.equal(activeFilterCount({ ...EMPTY_FILTER, query: '  ' }), 0)
  assert.equal(activeFilterCount({ ...EMPTY_FILTER, query: 'x', brand: 'b', family: 'Red', filamentType: 'PLA', matchColor: '#fff000' }), 5)
})

test('catalogUuid matches the server', () => {
  assert.equal(catalogUuid(4930), 'filamentcolors-4930')
})

test('every color family in the bundled catalog has a chip', () => {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const bundled = JSON.parse(fs.readFileSync(path.join(here, '../../../src/autoforge/data/filamentcolors_catalog.json'), 'utf8'))
  const known = new Set(COLOR_FAMILIES.map((f) => f.name))
  for (const e of bundled.filaments) {
    if (e.color_family) assert.ok(known.has(e.color_family), e.color_family)
  }
  // and the whole catalog filters/sorts fast enough to run on every keystroke
  const t0 = performance.now()
  for (let i = 0; i < 20; i++) sortCatalog(filterCatalog(bundled.filaments, { ...EMPTY_FILTER, query: 'blue pla' }), 'closest', '#3366cc')
  assert.ok((performance.now() - t0) / 20 < 50)
})
