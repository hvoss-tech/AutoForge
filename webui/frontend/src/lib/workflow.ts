export type StepState = 'done' | 'current' | 'upcoming'

export interface WorkflowStep {
  id: 'image' | 'filaments' | 'run' | 'adjust' | 'export'
  label: string
  state: StepState
  hint: string
}

export interface WorkflowInput {
  hasImage: boolean
  activeFilamentCount: number
  jobStatus: string | null
}

/** The five steps of making a print, and where the user currently is. */
export function getWorkflowSteps({ hasImage, activeFilamentCount, jobStatus }: WorkflowInput): WorkflowStep[] {
  const hasResult = jobStatus === 'completed'
  const running = jobStatus === 'running' || jobStatus === 'paused' || jobStatus === 'pending'
  const done = {
    image: hasImage || hasResult,
    filaments: activeFilamentCount > 0 || hasResult,
    run: hasResult,
  }
  const steps: Omit<WorkflowStep, 'state'>[] = [
    { id: 'image', label: 'Image', hint: 'Upload the picture you want to print' },
    { id: 'filaments', label: 'Filaments', hint: 'Add the filaments you own from the library on the left' },
    { id: 'run', label: 'Run', hint: running ? 'The optimizer is working out the layers' : 'Click Run to let the optimizer pick colors and heights' },
    { id: 'adjust', label: 'Adjust', hint: 'Fine-tune the color layers below, or prune to fewer swaps' },
    { id: 'export', label: 'Export', hint: 'Download the STL and swap instructions from File › Export' },
  ]
  let currentAssigned = false
  return steps.map((step) => {
    let state: StepState
    if (step.id === 'image' || step.id === 'filaments' || step.id === 'run') {
      state = done[step.id] ? 'done' : currentAssigned ? 'upcoming' : 'current'
    } else {
      // Adjusting and exporting stay available (not "done") once there's a result.
      state = hasResult ? 'current' : 'upcoming'
    }
    if (state === 'current') currentAssigned = true
    return { ...step, state }
  })
}
