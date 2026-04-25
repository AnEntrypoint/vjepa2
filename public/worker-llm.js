import { CreateMLCEngine } from 'https://esm.run/@mlc-ai/web-llm'

const MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
let engine = null

self.onmessage = async (e) => {
  const { type, id } = e.data
  if (type === 'load') {
    try {
      engine = await CreateMLCEngine(MODEL, {
        initProgressCallback: (p) => self.postMessage({
          type: 'progress', id,
          message: p.text ?? '',
          progress: Math.round((p.progress ?? 0) * 100),
        }),
      })
      self.postMessage({ type: 'loaded', id })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
    return
  }
  if (type === 'reset') {
    if (engine) await engine.resetChat().catch(() => {})
    self.postMessage({ type: 'reset', id })
    return
  }
  if (type === 'generate') {
    const { messages, config = {} } = e.data
    try {
      const tokens = []
      const stream = await engine.chat.completions.create({
        messages, stream: true,
        max_tokens: config.max_tokens ?? 400,
        temperature: config.temperature ?? 0.7,
      })
      for await (const chunk of stream) {
        const t = chunk.choices[0]?.delta?.content ?? ''
        if (t) { tokens.push(t); self.postMessage({ type: 'token', id, token: t }) }
      }
      self.postMessage({ type: 'result', id, text: tokens.join('') })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
  }
}
