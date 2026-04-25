import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'

const RELEASE = 'https://github.com/AnEntrypoint/vjepa2/releases/download/model-fp16-ssv2-v1'
const MODEL_URL = `${RELEASE}/model.fp16.onnx`
const CONFIG_URL = `${RELEASE}/config.json`

let session = null, config = null

function f32ToF16(val) {
  const f = new Float32Array(1); f[0] = val
  const u = new Uint32Array(f.buffer)[0]
  const sign = (u >>> 16) & 0x8000
  let exp = (u >>> 23) & 0xff
  let mant = u & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 1 : 0)
  exp = exp - 127 + 15
  if (exp >= 0x1f) return sign | 0x7c00
  if (exp <= 0) {
    if (exp < -10) return sign
    mant = (mant | 0x800000) >>> (1 - exp)
    return sign | (mant >>> 13)
  }
  return sign | (exp << 10) | (mant >>> 13)
}
function f16ToF32(h) {
  const sign = (h & 0x8000) << 16
  const exp = (h & 0x7c00) >>> 10
  const mant = h & 0x3ff
  let f
  if (exp === 0) f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024)
  else if (exp === 0x1f) f = mant ? NaN : Infinity
  else f = Math.pow(2, exp - 15) * (1 + mant / 1024)
  return sign ? -f : f
}
function softmax(a) {
  let m = -Infinity; for (const x of a) if (x > m) m = x
  let s = 0; const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) { out[i] = Math.exp(a[i] - m); s += out[i] }
  for (let i = 0; i < a.length; i++) out[i] /= s
  return out
}

self.onmessage = async (e) => {
  const { type, id } = e.data
  if (type === 'load') {
    try {
      self.postMessage({ type: 'progress', id, message: 'fetching config…' })
      config = await (await fetch(CONFIG_URL)).json()
      self.postMessage({ type: 'progress', id, message: 'downloading vision weights (~720 MB) — cached after first run' })
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      })
      self.postMessage({ type: 'loaded', id, classes: Object.keys(config.id2label).length })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
    return
  }
  if (type === 'classify') {
    const { frames, numFrames, size } = e.data
    try {
      const f16 = new Uint16Array(frames.length)
      for (let i = 0; i < frames.length; i++) f16[i] = f32ToF16(frames[i])
      const tensor = new ort.Tensor('float16', f16, [1, numFrames, 3, size, size])
      const inputName = session.inputNames[0]
      const out = await session.run({ [inputName]: tensor })
      const logits = out[session.outputNames[0]]
      const data = logits.type === 'float16'
        ? Float32Array.from(logits.data, h => f16ToF32(h))
        : logits.data
      const probs = softmax(data)
      const top = Array.from(probs).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5)
      const labels = top.map(([p, i]) => ({ label: config.id2label[i] ?? `class ${i}`, prob: p }))
      self.postMessage({ type: 'result', id, labels })
    } catch (err) {
      self.postMessage({ type: 'error', id, message: err.message })
    }
  }
}
