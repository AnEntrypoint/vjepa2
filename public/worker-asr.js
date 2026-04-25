import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2'

env.allowLocalModels = false
env.useBrowserCache = true

let asr = null

self.onmessage = async (e) => {
  const { type, id } = e.data
  if (type === 'load') {
    try {
      self.postMessage({ type: 'progress', id, message: 'downloading Whisper base (~150 MB)…' })
      asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
        dtype: 'q8',
        device: 'webgpu',
        progress_callback: (p) => {
          if (p.status === 'progress') self.postMessage({
            type: 'progress', id,
            message: `${p.file}: ${(p.progress ?? 0).toFixed(0)}%`
          })
        },
      })
      self.postMessage({ type: 'loaded', id })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
    return
  }
  if (type === 'transcribe') {
    const { audio, sampleRate } = e.data
    try {
      const out = await asr(audio, {
        chunk_length_s: 30,
        stride_length_s: 5,
        language: 'english',
        task: 'transcribe',
      })
      self.postMessage({ type: 'result', id, text: out.text || '' })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
  }
}
