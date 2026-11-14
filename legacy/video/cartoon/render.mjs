// Cartoon shots for the demo video (three.js engine adapted from claude-animation-skills/zach-d-films, MIT).
//   node render.mjs stills <film> 0.5,3,7.2 out_dir     -> PNG stills
//   node render.mjs frames <film> out_dir [fps]          -> out_dir/0001.png ... (the compositor reads these)
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.woff2': 'font/woff2', '.png': 'image/png', '.jpg': 'image/jpeg' };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (e, d) => { if (e) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' }); res.end(d); });
}).listen(0);
const port = server.address().port;

// use the Chromium already installed by Playwright on this machine (no extra download)
const local = process.env.CHROME || path.join(process.env.LOCALAPPDATA || '', 'ms-playwright', 'chromium-1254', 'chrome-win64', 'chrome.exe');
const browser = await chromium.launch({
  executablePath: fs.existsSync(local) ? local : undefined,
  args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=d3d11', '--enable-unsafe-webgpu'],
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.text()); });
page.on('pageerror', e => console.log('[pageerror]', e.message));

const [mode, film, a, b] = process.argv.slice(2);
await page.goto(`http://localhost:${port}/index.html?render=1&film=${film}`);
await page.waitForFunction(() => window.renderAt && window.ready);
await page.evaluate(() => window.ready);
// build every shot once and let textures load before capturing
const DUR = await page.evaluate(() => window.DURATION);
for (let t = 0; t < DUR; t += 0.5) await page.evaluate(t => window.renderAt(t), t);
await page.waitForTimeout(2500);
const shot = () => page.screenshot({ clip: { x: 0, y: 0, width: 1920, height: 1080 }, type: 'png' });

if (mode === 'stills') {
  fs.mkdirSync(b, { recursive: true });
  for (const t of a.split(',').map(Number)) {
    const t0 = Date.now();
    await page.evaluate(t => window.renderAt(t), t);
    fs.writeFileSync(path.join(b, `${film}_${t.toFixed(2)}.png`), await shot());
    console.log('still', t, Date.now() - t0, 'ms');
  }
} else {
  const out = a, fps = Number(b || 30);
  fs.mkdirSync(out, { recursive: true });
  const dur = await page.evaluate(() => window.DURATION);
  const n = Math.ceil(dur * fps);
  const T0 = Date.now();
  for (let i = 0; i < n; i++) {
    await page.evaluate(t => window.renderAt(t), i / fps);
    fs.writeFileSync(path.join(out, String(i + 1).padStart(4, '0') + '.png'), await shot());
    if (i % 60 === 0) console.log(`${film} frame ${i}/${n} ${((Date.now() - T0) / (i + 1)).toFixed(0)} ms/frame`);
  }
}
await browser.close();
server.close();
