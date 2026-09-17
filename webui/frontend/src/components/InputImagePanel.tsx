import React, { useCallback, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { describeApiError } from '../lib/apiError'
import { Upload, Image as ImageIcon, RefreshCw } from 'lucide-react'

export const InputImagePanel: React.FC = () => {
  const inputImage = useAppStore((s) => s.inputImage)
  const setInputImage = useAppStore((s) => s.setInputImage)
  const setSettings = useAppStore((s) => s.setSettings)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    setUploadError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/images/upload', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        throw new Error(`Upload failed: ${describeApiError(await response.json().catch(() => null), response.status)}`)
      }
      const data = await response.json()
      if (!data.filename) {
        throw new Error('Upload failed: server did not return a filename')
      }

      // Settings first: both queue one undo step and the last label wins, so
      // this step is recorded as "Input image changed", not "Settings changed".
      const current = useAppStore.getState().settings
      setSettings({ ...current, input_image: data.filename })
      setInputImage(URL.createObjectURL(file))
      // Init is triggered reactively by ActiveFilamentsPanel's effect (it
      // watches for the image+filaments-present transition from both
      // directions) — calling it here too raced that effect whenever
      // filaments were already active, firing runInit() twice and logging
      // a spurious "Already initializing" rejection for the second one.
    } catch (e) {
      // Deliberately NOT falling back to a local blob preview here. That
      // used to set `inputImage` (which alone drives the Run button's
      // enabled state) to an object URL even though the upload never
      // reached the server — `settings.input_image` stayed empty (or kept
      // whatever the previous successful upload was), so Run looked ready
      // but either rejected with "Upload an input image" or silently
      // reran the old image while the preview showed the new one.
      console.error('Failed to upload image:', e)
      setUploadError(e instanceof Error ? e.message : 'Failed to upload image')
    }
  }, [setInputImage, setSettings])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => setIsDragging(false), [])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0])
  }, [handleFile])

  return (
    <div className="flex flex-col h-full bg-gray-800 rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-700">
        <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
          <ImageIcon className="w-3.5 h-3.5" />
          Input Image
        </h3>
      </div>
      {uploadError && (
        <div className="px-3 py-1 bg-red-900/40 text-xs text-red-300" data-testid="image-upload-error">
          {uploadError}
        </div>
      )}
      <div
        className="flex-1 relative p-2 flex items-center justify-center overflow-hidden"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {inputImage ? (
          <>
            <img
              src={inputImage}
              alt="Input"
              className="max-w-full max-h-full object-contain rounded"
              data-testid="input-image"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="absolute top-1 right-1 flex items-center gap-1 px-2 py-1 bg-gray-900/80 hover:bg-gray-900 rounded text-xs text-gray-200"
              title="Change image"
              data-testid="change-image-btn"
            >
              <RefreshCw className="w-3 h-3" />
              Change
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
              data-testid="image-file-input"
            />
          </>
        ) : (
          <div
            className={`w-full h-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
              isDragging ? 'border-blue-400 bg-blue-900/20' : 'border-gray-600 hover:border-gray-500'
            }`}
            onClick={() => fileInputRef.current?.click()}
            data-testid="image-drop-zone"
          >
            <Upload className="w-8 h-8 text-gray-500 mb-2" />
            <p className="text-xs text-gray-400">Drop image or click to upload</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
              data-testid="image-file-input"
            />
          </div>
        )}
      </div>
    </div>
  )
}
