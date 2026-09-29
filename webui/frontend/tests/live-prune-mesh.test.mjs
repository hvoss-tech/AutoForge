import test from 'node:test'
import assert from 'node:assert/strict'
import { liveMeshUrl } from '../src/lib/pruning.ts'

const live = { forJob: 'job-1', pruneJobId: 'prune-1', url: '/api/outputs/live-ply/prune-1' }

test('the live pruning mesh is shown while that prune runs or is paused', () => {
  for (const status of ['pending', 'running', 'paused']) {
    assert.equal(liveMeshUrl(live, 'job-1', { job_id: 'prune-1', status }), live.url)
  }
})

test('the live pruning mesh is dropped once the prune ends', () => {
  // Completed: the pruned job becomes current and serves its final mesh.
  assert.equal(liveMeshUrl(live, 'prune-1', { job_id: 'prune-1', status: 'completed' }), null)
  // Cancelled/failed: rolled back, so the original's own mesh is right.
  for (const status of ['cancelled', 'failed']) {
    assert.equal(liveMeshUrl(live, 'job-1', { job_id: 'prune-1', status }), null)
  }
})

test('the live pruning mesh never shows for another job or another prune', () => {
  assert.equal(liveMeshUrl(live, 'job-2', { job_id: 'prune-1', status: 'running' }), null)
  assert.equal(liveMeshUrl(live, 'job-1', { job_id: 'prune-2', status: 'running' }), null)
  assert.equal(liveMeshUrl(null, 'job-1', { job_id: 'prune-1', status: 'running' }), null)
  assert.equal(liveMeshUrl(live, 'job-1', null), null)
})
