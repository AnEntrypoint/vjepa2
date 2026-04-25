const fs = require('fs');
const path = require('path');
const assert = require('assert');

const root = __dirname;

assert.ok(fs.existsSync(path.join(root,'public/index.html')), 'index.html missing');
assert.ok(fs.existsSync(path.join(root,'serve.js')), 'serve.js missing');
assert.ok(fs.existsSync(path.join(root,'scripts/convert_fp16.py')), 'convert_fp16.py missing');
assert.ok(fs.existsSync(path.join(root,'.github/workflows/build-fp16.yml')), 'CI workflow missing');

const html = fs.readFileSync(path.join(root,'public/index.html'),'utf8');
assert.ok(html.includes("executionProviders: ['webgpu']"), 'webgpu EP missing in html');
assert.ok(html.includes('float16'), 'fp16 input encoding missing');
assert.ok(html.includes('AnEntrypoint/vjepa2/releases'), 'release URL missing');
assert.ok(html.includes('NUM_FRAMES = 32'), 'frame count missing');

const wf = fs.readFileSync(path.join(root,'.github/workflows/build-fp16.yml'),'utf8');
assert.ok(wf.includes('convert_fp16.py'), 'workflow does not run conversion');
assert.ok(wf.includes('softprops/action-gh-release'), 'workflow does not publish release');

console.log('OK');
