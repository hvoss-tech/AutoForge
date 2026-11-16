// Test runner: node --import tsx/esm --test tests/slicer-printers.test.mjs
//
// The printer picker of File › Slicer project (lib/slicerPrinters).

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  defaultManualSwaps,
  initialChoice,
  modelsOf,
  searchPrinters,
  variantsOf,
  vendorsOf,
  nozzleMatch,
  popularPrinters,
} from '../src/lib/slicerPrinters.ts'

const p = (vendor, model, nozzle, auto_swaps = false) => ({
  name: `${model} ${nozzle} nozzle`,
  vendor,
  model,
  nozzle,
  bed: [0, 0, 200, 200],
  auto_swaps,
})

const printers = [
  p('Prusa', 'Prusa MK4', '0.6'),
  p('Prusa', 'Prusa MK4', '0.4'),
  p('Prusa', 'Prusa MK4 MMU3', '0.4', true),
  p('BBL', 'Bambu Lab X1 Carbon', '0.4', true),
  p('BBL', 'Bambu Lab A1 mini', '0.4', true),
  p('anycubic', 'Anycubic Kobra 3', '0.4'),
  p('Prusa', 'Prusa MK10', '0.4'),
]

test('vendors are sorted without regard to case', () => {
  assert.deepEqual(vendorsOf(printers), ['anycubic', 'BBL', 'Prusa'])
})

test('models sort naturally (MK4 before MK10)', () => {
  assert.deepEqual(modelsOf(printers, 'Prusa'), ['Prusa MK4', 'Prusa MK4 MMU3', 'Prusa MK10'])
})

test('the 0.4 mm nozzle comes first', () => {
  assert.deepEqual(variantsOf(printers, 'Prusa', 'Prusa MK4').map((x) => x.nozzle), ['0.4', '0.6'])
})

test('search matches every word in any order and case', () => {
  assert.deepEqual(searchPrinters(printers, 'carbon X1').map((x) => x.model), ['Bambu Lab X1 Carbon'])
  assert.deepEqual(searchPrinters(printers, 'bbl').length, 2)
  assert.deepEqual(searchPrinters(printers, '   '), [])
  assert.equal(searchPrinters(printers, 'prusa', 2).length, 2)
})

test('the dialog reopens on the remembered printer', () => {
  assert.deepEqual(initialChoice(printers, 'Prusa MK4 0.6 nozzle'), { vendor: 'Prusa', model: 'Prusa MK4', name: 'Prusa MK4 0.6 nozzle' })
  assert.equal(initialChoice(printers, 'Gone 0.4 nozzle'), null)
  assert.equal(initialChoice(printers, ''), null)
})

test('swap mode: the remembered choice for that printer, else what the printer can do', () => {
  const x1 = printers[3]
  const mk4 = printers[1]
  assert.equal(defaultManualSwaps(x1, null, ''), false)
  assert.equal(defaultManualSwaps(mk4, null, ''), true)
  assert.equal(defaultManualSwaps(x1, true, x1.name), true)
  // a choice made for another printer doesn't carry over
  assert.equal(defaultManualSwaps(x1, true, mk4.name), false)
  assert.equal(defaultManualSwaps(undefined, false, ''), true)
})

test('search puts the 0.4 mm nozzle of each printer first (Enter picks it)', () => {
  const list = [p('BBL', 'Bambu Lab X1 Carbon', '0.2', true), p('BBL', 'Bambu Lab X1 Carbon', '0.4', true), p('BBL', 'Bambu Lab X1 Carbon', '0.6', true), p('BBL', 'Bambu Lab X1E', '0.2', true), p('BBL', 'Bambu Lab X1E', '0.4', true)]
  assert.deepEqual(searchPrinters(list, 'x1').map((x) => x.name), [
    'Bambu Lab X1 Carbon 0.4 nozzle', 'Bambu Lab X1 Carbon 0.2 nozzle', 'Bambu Lab X1 Carbon 0.6 nozzle',
    'Bambu Lab X1E 0.4 nozzle', 'Bambu Lab X1E 0.2 nozzle',
  ])
})

test('short names and nozzle labels', async () => {
  const { shortPrinterName, nozzleLabel, popularPrinters } = await import('../src/lib/slicerPrinters.ts')
  assert.equal(shortPrinterName('Bambu Lab X1 Carbon 0.4 nozzle'), 'Bambu Lab X1 Carbon')
  assert.equal(shortPrinterName('Snapmaker U1 (0.4 nozzle)'), 'Snapmaker U1')
  assert.equal(shortPrinterName('Snapmaker U1 (0.4+0.6 nozzle)'), 'Snapmaker U1')
  assert.equal(shortPrinterName('FELIX Pro 3 0.25mm nozzle'), 'FELIX Pro 3')
  assert.equal(shortPrinterName('Raise3D Pro3 0.4 nozzle (Dual)'), 'Raise3D Pro3 0.4 nozzle (Dual)')
  assert.equal(nozzleLabel(p('BBL', 'Bambu Lab X1 Carbon', '0.4')), '0.4 mm')
  assert.equal(nozzleLabel({ ...p('Prusa', 'Prusa MK4S', '0.4'), name: 'Prusa MK4S HF0.4 nozzle' }), 'HF0.4 nozzle')
  // in the curated order, only the ones the server lists
  assert.deepEqual(popularPrinters(printers).map((x) => x.name), ['Bambu Lab A1 mini 0.4 nozzle', 'Bambu Lab X1 Carbon 0.4 nozzle', 'Anycubic Kobra 3 0.4 nozzle'])
})

test('the configured nozzle decides which preset comes first', () => {
  const list = [p('BBL', 'Bambu Lab X1 Carbon', '0.2', true), p('BBL', 'Bambu Lab X1 Carbon', '0.4', true), p('BBL', 'Bambu Lab X1 Carbon', '0.6', true)]
  assert.equal(searchPrinters(list, 'x1', 50, 0.6)[0].nozzle, '0.6')
  assert.equal(variantsOf(list, 'BBL', 'Bambu Lab X1 Carbon', 0.2)[0].nozzle, '0.2')
  // no 0.5 preset: the closest nozzle first, a tie goes to the usual 0.4
  assert.equal(searchPrinters(list, 'x1', 50, 0.5)[0].nozzle, '0.4')
  assert.equal(searchPrinters(list, 'x1', 50, 0.25)[0].nozzle, '0.2')
  assert.deepEqual(variantsOf(list, 'BBL', 'Bambu Lab X1 Carbon', 0.55).map((x) => x.nozzle), ['0.6', '0.4', '0.2'])
  const pop = [p('BBL', 'Bambu Lab A1 mini', '0.4', true), p('BBL', 'Bambu Lab A1 mini', '0.6', true)]
  assert.deepEqual(popularPrinters(pop, 0.6).map((x) => x.name), ['Bambu Lab A1 mini 0.6 nozzle'])
  assert.deepEqual(popularPrinters(pop, 0.8).map((x) => x.name), ['Bambu Lab A1 mini 0.4 nozzle'])
})

test('nozzle match mirrors the server: same, switched (same naming), missing', () => {
  const x04 = p('Prusa', 'Prusa MK4S', '0.4')
  const x06 = p('Prusa', 'Prusa MK4S', '0.6')
  const hf06 = { ...p('Prusa', 'Prusa MK4S', '0.6'), name: 'Prusa MK4S HF0.6 nozzle' }
  const list = [x04, hf06, x06]
  assert.equal(nozzleMatch(list, x04, 0.4).kind, 'same')
  assert.equal(nozzleMatch(list, x04, null).kind, 'same')
  const switched = nozzleMatch(list, x04, 0.6)
  assert.equal(switched.kind, 'switched')
  assert.equal(switched.printer.name, 'Prusa MK4S 0.6 nozzle')
  assert.equal(nozzleMatch(list, hf06, 0.6).kind, 'same')
  const missing = nozzleMatch(list, x04, 0.8)
  assert.equal(missing.kind, 'missing')
  assert.equal(missing.printer.name, x04.name)
  assert.deepEqual(missing.available, [0.4, 0.6])
})

test('picking a printer for slicer projects does not make the result stale', async () => {
  const { staleReasons } = await import('../src/lib/staleResult.ts')
  const run = { settings: { iterations: 100, orca_printer: '', orca_manual_swaps: null }, filamentUuids: [] }
  const now = { iterations: 100, orca_printer: 'Bambu Lab X1 Carbon 0.4 nozzle', orca_manual_swaps: false }
  assert.deepEqual(staleReasons(run, now, []), [])
})
