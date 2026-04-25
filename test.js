const fs = require('fs')
const path = require('path')
const assert = require('assert')

const root = __dirname

for (const f of [
  'public/index.html',
  'public/worker-vision.js',
  'public/worker-asr.js',
  'public/worker-llm.js',
  'serve.js',
  'scripts/convert_fp16.py',
  '.github/workflows/build-fp16.yml',
]) assert.ok(fs.existsSync(path.join(root, f)), `${f} missing`)

const html = fs.readFileSync(path.join(root, 'public/index.html'), 'utf8')
assert.ok(html.includes('worker-vision.js'), 'vision worker not wired')
assert.ok(html.includes('worker-asr.js'), 'asr worker not wired')
assert.ok(html.includes('worker-llm.js'), 'llm worker not wired')
assert.ok(html.includes('NUM_FRAMES = 16'), 'frame count missing')

const v = fs.readFileSync(path.join(root, 'public/worker-vision.js'), 'utf8')
assert.ok(v.includes("executionProviders: ['webgpu']"), 'vision EP missing')
assert.ok(v.includes('model-fp16-ssv2-v1'), 'vision release tag missing')
assert.ok(v.includes('float16'), 'vision fp16 missing')

const a = fs.readFileSync(path.join(root, 'public/worker-asr.js'), 'utf8')
assert.ok(a.includes('whisper-base'), 'asr model missing')
assert.ok(a.includes("dtype: 'q8'"), 'asr dtype missing')

const l = fs.readFileSync(path.join(root, 'public/worker-llm.js'), 'utf8')
assert.ok(l.includes('Llama-3.2-1B-Instruct-q4f16_1-MLC'), 'llm model missing')
assert.ok(l.includes('@mlc-ai/web-llm'), 'mlc import missing')

const wf = fs.readFileSync(path.join(root, '.github/workflows/build-fp16.yml'), 'utf8')
assert.ok(wf.includes('convert_fp16.py'), 'workflow conversion missing')

console.log('OK')
