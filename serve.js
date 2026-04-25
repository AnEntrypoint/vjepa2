const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8787;
const ROOT = __dirname;

const TYPES = {
  '.html':'text/html; charset=utf-8',
  '.js':'text/javascript; charset=utf-8',
  '.mjs':'text/javascript; charset=utf-8',
  '.json':'application/json',
  '.onnx':'application/octet-stream',
  '.wasm':'application/wasm',
  '.css':'text/css',
};

const ROUTES = new Map([
  ['/', '/public/index.html'],
]);

http.createServer((req, res) => {
  let url = req.url.split('?')[0];
  url = ROUTES.get(url) ?? url;
  const filePath = path.join(ROOT, url);
  if (!filePath.startsWith(ROOT)) { res.writeHead(403).end(); return; }
  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404).end('not found: '+url); return; }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': TYPES[ext] ?? 'application/octet-stream',
      'Content-Length': st.size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-cache',
    });
    fs.createReadStream(filePath).pipe(res);
  });
}).listen(PORT, () => console.log(`http://localhost:${PORT}`));
