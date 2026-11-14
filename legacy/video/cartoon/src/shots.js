// Cartoon story shots for the Track 01 demo (hook + problem). Engine: claude-animation-skills/zach-d-films (MIT),
// adapted to 16:9. Every update(t) is a pure function of t. Times are scene-local seconds and include the
// compositor's 0.5 s lead before the voice starts (cut points come from video/build/voice/*.json word timings).
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { makeHuman } from './human.js';
import { stand, glowSprite } from './props.js';
import { materials } from './materials.js';
import { kf, lerp, smooth, easeOut, easeInOut, range, clamp, V, cam, place, shake } from './anim.js';
import { PUPILS } from './pupils.js';

const LEAD = 0.5;
const at = w => LEAD + w;           // voice word time -> scene time
const CYAN = new THREE.Color(0x48e6ff), RED = new THREE.Color(0xff3b30), GREEN = new THREE.Color(0x4dff9a);

// ---------------------------------------------------------------- helpers
function rnd(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function person(shirt = 0x2f6fd6, pants = 0x2b2f3a) {
  const h = makeHuman('tee');
  const sh = new THREE.MeshStandardMaterial({ color: shirt, roughness: 0.85 });
  const pa = new THREE.MeshStandardMaterial({ color: pants, roughness: 0.9 });
  const M = materials();
  // torso + arms get the shirt, legs + feet the trousers / shoes
  const legRoots = ['lHip', 'rHip'].map(k => h.j[k]);
  h.root.traverse(o => {
    if (!o.isMesh || o.material !== M.cloth) return;
    let p = o, leg = false;
    while (p) { if (legRoots.includes(p)) { leg = true; break; } p = p.parent; }
    o.material = leg ? pa : sh;
  });
  for (const k of ['lAn', 'rAn']) h.j[k].traverse(o => { if (o.isMesh && o.material === M.skin) o.material = new THREE.MeshStandardMaterial({ color: 0x1b1b1f, roughness: 0.6 }); });
  return h;
}

const SEATED = {
  lHip: [-1.5, 0, 0.06], rHip: [-1.5, 0, -0.06], lKn: [1.45, 0, 0], rKn: [1.45, 0, 0],
  lSh: [-0.35, 0, 0.12], rSh: [-0.35, 0, -0.12], lEl: [-0.9, 0, 0], rEl: [-0.9, 0, 0], drop: -0.43,
};

function eyePoint(h) {
  h.root.updateMatrixWorld(true);
  return h.j.head.localToWorld(V(0, 0.11, 0.1));
}

function headset() {
  const g = new THREE.Group();
  const shell = new THREE.MeshPhysicalMaterial({ color: 0xf2f4f7, roughness: 0.35, clearcoat: 0.8 });
  const visor = new THREE.MeshPhysicalMaterial({ color: 0x0b0f17, roughness: 0.1, metalness: 0.2, clearcoat: 1 });
  const body = new THREE.Mesh(new RoundedBoxGeometry(0.19, 0.095, 0.085, 4, 0.03), shell);
  body.position.set(0, 0.11, 0.118); g.add(body);
  const front = new THREE.Mesh(new RoundedBoxGeometry(0.17, 0.075, 0.02, 4, 0.012), visor);
  front.position.set(0, 0.11, 0.163); g.add(front);
  const strap = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.012, 10, 48), new THREE.MeshStandardMaterial({ color: 0x30343c, roughness: 0.8 }));
  strap.rotation.x = Math.PI / 2; strap.scale.set(0.95, 1.12, 1); strap.position.set(0, 0.125, 0.0); g.add(strap);
  const led = glowSprite(0x48e6ff, 0.035); led.position.set(0.06, 0.13, 0.175); g.add(led);
  g.traverse(o => { if (o.isMesh) o.castShadow = true; });
  return g;
}

// glowing gaze beam (a thin additive cylinder + a soft halo)
function beam(color = CYAN) {
  const g = new THREE.Group();
  const geo = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true); geo.translate(0, 0.5, 0);
  const core = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
  const halo = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
  g.add(core, halo);
  const hit = glowSprite(color.getHex(), 0.18); g.add(hit);
  const hit2 = glowSprite(0xffffff, 0.05); g.add(hit2);
  g.set = (from, to, a = 1, col = null, r = 0.004) => {
    const d = to.clone().sub(from), L = d.length();
    for (const m of [core, halo]) { m.position.copy(from); m.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize()); }
    core.scale.set(r, L * a, r); halo.scale.set(r * 3.5, L * a, r * 3.5);
    core.material.opacity = 0.9 * clamp(a * 3); halo.material.opacity = 0.35 * clamp(a * 3);
    if (col) { halo.material.color.copy(col); hit.material.color.copy(col); }
    const tip = from.clone().addScaledVector(d, a);
    hit.position.copy(tip); hit2.position.copy(tip);
    hit.material.opacity = a >= 0.999 ? 0.9 : 0; hit2.material.opacity = a >= 0.999 ? 1 : 0;
  };
  return g;
}

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  tex.redraw = (...args) => { draw(c.getContext('2d'), w, h, ...args); tex.needsUpdate = true; };
  tex.redraw();
  return tex;
}

function spriteFrom(w, h, draw, size) {
  const tex = canvasTex(w, h, draw);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false }));
  s.scale.set(size * w / h, size, 1);
  return s;
}

function drawLock(g, w, h) {
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#ffcc33'; g.strokeStyle = '#ffcc33'; g.lineWidth = 22;
  g.beginPath(); g.arc(w / 2, h * 0.42, w * 0.22, Math.PI, 0); g.stroke();
  g.beginPath(); g.roundRect(w * 0.2, h * 0.42, w * 0.6, h * 0.48, 22); g.fill();
  g.fillStyle = '#5a3d00'; g.beginPath(); g.arc(w / 2, h * 0.62, w * 0.07, 0, 7); g.fill();
  g.fillRect(w / 2 - 8, h * 0.62, 16, h * 0.16);
}
function drawQ(g, w, h) {
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#ff3b30'; g.beginPath(); g.arc(w / 2, h / 2, w * 0.46, 0, 7); g.fill();
  g.fillStyle = '#fff'; g.font = `800 ${Math.round(h * 0.7)}px Arial`; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('?', w / 2, h * 0.53);
}
function drawWarn(g, w, h) {
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#ff3b30'; g.beginPath(); g.moveTo(w / 2, h * 0.06); g.lineTo(w * 0.96, h * 0.92); g.lineTo(w * 0.04, h * 0.92); g.closePath(); g.fill();
  g.fillStyle = '#fff'; g.font = `900 ${Math.round(h * 0.62)}px Arial`; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('!', w / 2, h * 0.6);
}

// a procedurally painted "virtual world" for the foveated-rendering panel
function drawWorld(g, w, h) {
  const R = rnd(7);
  const sky = g.createLinearGradient(0, 0, 0, h * 0.6);
  sky.addColorStop(0, '#1d4e9e'); sky.addColorStop(0.6, '#f08a5d'); sky.addColorStop(1, '#ffd38a');
  g.fillStyle = sky; g.fillRect(0, 0, w, h);
  g.fillStyle = '#fff6d0'; g.beginPath(); g.arc(w * 0.62, h * 0.42, h * 0.09, 0, 7); g.fill();
  for (let i = 0; i < 140; i++) { g.fillStyle = `rgba(255,255,255,${0.3 + R() * 0.6})`; g.fillRect(R() * w, R() * h * 0.3, 2, 2); }
  const layers = [['#6b3f72', 0.45, 0.16], ['#46305e', 0.55, 0.13], ['#24203d', 0.66, 0.1]];
  for (const [col, base, amp] of layers) {
    g.fillStyle = col; g.beginPath(); g.moveTo(0, h);
    for (let x = 0; x <= w; x += 8) g.lineTo(x, h * base - Math.abs(Math.sin(x * 0.004 + base * 9) * amp * h) - Math.sin(x * 0.021 + base) * 10);
    g.lineTo(w, h); g.fill();
  }
  // lake + reflections
  g.fillStyle = '#1a2f55'; g.fillRect(0, h * 0.74, w, h * 0.26);
  for (let i = 0; i < 80; i++) { g.fillStyle = `rgba(255,214,140,${R() * 0.5})`; g.fillRect(w * 0.62 - 80 + R() * 160, h * 0.75 + R() * h * 0.22, 20 + R() * 60, 3); }
  // pine trees + tiny houses with lit windows (fine detail that pixelation destroys)
  for (let i = 0; i < 70; i++) {
    const x = R() * w, y = h * (0.66 + R() * 0.1), s = 18 + R() * 34;
    g.fillStyle = '#0f1b2a'; g.beginPath(); g.moveTo(x, y - s * 2); g.lineTo(x + s * 0.6, y); g.lineTo(x - s * 0.6, y); g.fill();
  }
  for (let i = 0; i < 14; i++) {
    const x = R() * w, y = h * (0.7 + R() * 0.03), s = 26 + R() * 20;
    g.fillStyle = '#2a2236'; g.fillRect(x, y - s, s * 1.3, s);
    g.fillStyle = '#ffcf6b'; for (let k = 0; k < 3; k++) g.fillRect(x + 5 + k * s * 0.4, y - s * 0.7, 6, 6);
  }
}

function foveaMaterial(map) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: map }, uG: { value: new THREE.Vector2(0.5, 0.5) }, uR: { value: 0.16 }, uOn: { value: 0 }, uAsp: { value: 16 / 9 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform sampler2D map; uniform vec2 uG; uniform float uR, uOn, uAsp; varying vec2 vUv;
      void main(){
        vec2 d = vUv - uG; d.x *= uAsp; float r = length(d);
        float inside = 1.0 - smoothstep(uR * 0.86, uR, r);
        vec2 n = vec2(28.0 * uAsp, 28.0);
        vec2 q = (floor(vUv * n) + 0.5) / n;
        vec3 lo = texture2D(map, q).rgb; float gr = dot(lo, vec3(0.3, 0.59, 0.11)); lo = mix(lo, vec3(gr), 0.45) * 0.72;
        vec3 hi = texture2D(map, vUv).rgb;
        vec3 c = mix(hi, mix(lo, hi, inside), uOn);
        float ring = (1.0 - smoothstep(0.0, 0.006, abs(r - uR))) * uOn;
        c = mix(c, vec3(0.35, 0.95, 1.0), ring);
        gl_FragColor = vec4(c, 1.0);
        #include <colorspace_fragment>
      }`,
  });
}

// ================================================================= HOOK
// S1 0 -> 3.29 : "What if your headset knew exactly where you were looking?"
function hook1({ makeStage }) {
  const scene = makeStage({ shadowSize: 2.5 });
  const man = person(0x2f6fd6); man.root.rotation.y = 0.35; scene.add(man.root);
  const hs = headset(); man.j.head.add(hs);
  const panelTex = canvasTex(1600, 900, drawWorld);
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.9), new THREE.MeshBasicMaterial({ map: panelTex, toneMapped: false, transparent: true }));
  panel.position.set(0.95, 1.75, 1.7); panel.lookAt(0, 1.7, 0); scene.add(panel);
  const frame = new THREE.Mesh(new THREE.PlaneGeometry(1.68, 0.98), new THREE.MeshBasicMaterial({ color: 0x48e6ff, transparent: true, opacity: 0.25, toneMapped: false }));
  frame.position.copy(panel.position).addScaledVector(V(0, 1.7, 0).sub(panel.position).normalize(), -0.005); frame.lookAt(0, 1.7, 0); scene.add(frame);
  const b = beam(); scene.add(b);
  const c = cam(32);
  const tOn = at(0.91);
  return {
    scene, camera: c,
    update(t) {
      man.setPose({ neck: [0.0, 0.25 + Math.sin(t * 0.8) * 0.03, 0], head: [-0.06, 0.12, 0], lSh: [0.05, 0, 0.16], rSh: [0.05, 0, -0.16], lEl: [-0.25, 0, 0], rEl: [-0.25, 0, 0] });
      const k = easeInOut(range(t, 0, 3.4));
      place(c, [lerp(-2.7, -2.35, k), lerp(1.9, 1.85, k), lerp(0.7, 0.85, k)], [0.25, 1.68, lerp(0.75, 0.85, k)]);
      const from = eyePoint(man);
      const p = V(0.95, 1.75, 1.7).add(V(lerp(-0.35, 0.18, smooth(range(t, tOn, 3.3))), lerp(0.12, -0.05, smooth(range(t, tOn, 3.3))), 0));
      b.set(from, p, easeOut(range(t, tOn, tOn + 0.45)));
      panel.material.opacity = smooth(range(t, 0.1, 0.7));
      frame.material.opacity = 0.25 * smooth(range(t, 0.1, 0.7));
      return { dof: { focus: 1.6, aperture: 0.0008, maxblur: 0.006 }, bloom: 0.6 };
    },
  };
}

// S2 3.29 -> 5.88 : "It could draw only that spot in full detail."
function hook2({ makeStage }) {
  const scene = makeStage();
  const tex = canvasTex(1920, 1080, drawWorld);
  const mat = foveaMaterial(tex);
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.9), mat); panel.position.set(0, 1.5, 0); scene.add(panel);
  const c = cam(30);
  const T0 = 3.29;
  return {
    scene, camera: c,
    update(t) {
      const lt = t;   // shot-local
      place(c, [0, 1.5, lerp(1.72, 1.62, easeInOut(range(lt, 0, 2.6)))], [0, 1.5, 0]);
      mat.uniforms.uOn.value = smooth(range(lt, 0.15, 0.55));
      // gaze wanders across the scene: sun -> houses -> lake
      const gx = kf(lt, [[0, 0.62], [0.9, 0.62], [1.6, 0.36], [2.6, 0.28]], easeInOut);
      const gy = kf(lt, [[0, 0.58], [0.9, 0.58], [1.6, 0.3], [2.6, 0.24]], easeInOut);
      mat.uniforms.uG.value.set(gx, gy);
      mat.uniforms.uR.value = 0.15 + 0.01 * Math.sin(lt * 5);
      return { bloom: 0.2 };
    },
  };
}

function car() {
  const g = new THREE.Group();
  const paint = new THREE.MeshPhysicalMaterial({ color: 0xd63a2f, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1c1d22, roughness: 0.7 });
  const seatM = new THREE.MeshStandardMaterial({ color: 0x3b2a22, roughness: 0.7 });
  const chrome = new THREE.MeshPhysicalMaterial({ color: 0xdde2e8, metalness: 1, roughness: 0.2 });
  const add = (geo, m, p, r = [0, 0, 0]) => { const o = new THREE.Mesh(geo, m); o.position.set(...p); o.rotation.set(...r); o.castShadow = o.receiveShadow = true; g.add(o); return o; };
  add(new RoundedBoxGeometry(1.75, 0.42, 3.9, 4, 0.12), paint, [0, 0.52, 0.2]);          // lower body
  add(new RoundedBoxGeometry(1.75, 0.28, 1.35, 4, 0.1), paint, [0, 0.8, 1.45]);          // hood
  add(new RoundedBoxGeometry(1.75, 0.28, 0.95, 4, 0.1), paint, [0, 0.8, -1.25]);         // rear deck
  for (const s of [-1, 1]) add(new RoundedBoxGeometry(0.1, 0.3, 1.5, 3, 0.04), paint, [s * 0.83, 0.82, 0.05]); // doors
  add(new RoundedBoxGeometry(1.6, 0.08, 1.5, 2, 0.03), dark, [0, 0.72, 0.05]);           // floor
  const glass = new THREE.MeshPhysicalMaterial({ color: 0xbfe6ff, transmission: 0.9, roughness: 0.05, transparent: true, opacity: 0.35 });
  add(new THREE.BoxGeometry(1.6, 0.42, 0.02), glass, [0, 1.13, 0.74], [-0.45, 0, 0]);     // windshield
  add(new RoundedBoxGeometry(1.5, 0.12, 0.22, 3, 0.05), dark, [0, 0.98, 0.72]);          // dashboard
  for (const x of [-0.42, 0.42]) {
    add(new RoundedBoxGeometry(0.55, 0.16, 0.55, 3, 0.06), seatM, [x, 0.84, -0.25]);
    add(new RoundedBoxGeometry(0.55, 0.62, 0.14, 3, 0.06), seatM, [x, 1.15, -0.5], [-0.12, 0, 0]);
  }
  const wheel = add(new THREE.TorusGeometry(0.17, 0.022, 10, 32), dark, [0.42, 1.12, 0.47], [-0.95, 0, 0]);
  add(new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8), dark, [0.42, 1.04, 0.6], [-0.95 + Math.PI / 2, 0, 0]);
  for (const [x, z] of [[-0.86, 1.35], [0.86, 1.35], [-0.86, -1.15], [0.86, -1.15]]) {
    add(new THREE.CylinderGeometry(0.34, 0.34, 0.26, 28), dark, [x, 0.34, z], [0, 0, Math.PI / 2]);
    add(new THREE.CylinderGeometry(0.18, 0.18, 0.27, 20), chrome, [x, 0.34, z], [0, 0, Math.PI / 2]);
  }
  for (const s of [-1, 1]) add(new THREE.SphereGeometry(0.09, 16, 12), new THREE.MeshBasicMaterial({ color: 0xfff6d8 }), [s * 0.6, 0.7, 2.15]);
  g.userData.wheel = wheel;
  return g;
}

function road() {
  const tex = canvasTex(256, 1024, (g, w, h) => {
    g.fillStyle = '#3a3d44'; g.fillRect(0, 0, w, h);
    const R = rnd(3); for (let i = 0; i < 1600; i++) { g.fillStyle = `rgba(255,255,255,${R() * 0.06})`; g.fillRect(R() * w, R() * h, 2, 2); }
    g.fillStyle = '#f2f2f2'; for (let y = 0; y < h; y += 256) g.fillRect(w / 2 - 6, y, 12, 140);
    g.fillStyle = '#e8e8e8'; g.fillRect(8, 0, 8, h); g.fillRect(w - 16, 0, 8, h);
  });
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(1, 12);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(7, 120), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9 }));
  m.rotation.x = -Math.PI / 2; m.position.set(0.9, 0.005, 40); m.receiveShadow = true;
  m.userData.tex = tex;
  return m;
}

// S3 5.88 -> 9.12 : "A car could notice the moment a driver's eyes drift off the road."
function hook3({ makeStage }) {
  const scene = makeStage({ shadowSize: 4, sunDir: [-4, 7, 3] });
  const rd = road(); scene.add(rd);
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x7fb069, roughness: 1 }));
  grass.rotation.x = -Math.PI / 2; grass.receiveShadow = true; scene.add(grass);
  const R = rnd(11);
  for (let i = 0; i < 40; i++) {   // roadside trees
    const s = R() < 0.5 ? -1 : 1, z = -10 + R() * 80;
    const tr = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.9, 8), new THREE.MeshStandardMaterial({ color: 0x6b4a2f }));
    trunk.position.y = 0.45; tr.add(trunk);
    const top = new THREE.Mesh(new THREE.SphereGeometry(0.6 + R() * 0.3, 16, 12), new THREE.MeshStandardMaterial({ color: 0x3f8f4a, roughness: 0.9 }));
    top.position.y = 1.3; tr.add(top);
    tr.position.set(0.9 + s * (4.5 + R() * 4), 0, z); tr.traverse(o => { if (o.isMesh) o.castShadow = true; });
    tr.userData.z = z; scene.add(tr); (scene.userData.trees ||= []).push(tr);
  }
  const cg = car(); cg.position.x = 0.9; scene.add(cg);
  const man = person(0xf0b429); man.root.position.set(1.32, 0.25, -0.18); scene.add(man.root);
  const phone = new THREE.Group();
  const pb = new THREE.Mesh(new RoundedBoxGeometry(0.075, 0.15, 0.012, 3, 0.008), new THREE.MeshStandardMaterial({ color: 0x15161a }));
  const ps = new THREE.Mesh(new THREE.PlaneGeometry(0.065, 0.135), new THREE.MeshBasicMaterial({ color: 0x7fd1ff, toneMapped: false }));
  ps.position.z = 0.007; phone.add(pb, ps); scene.add(phone);
  const b = beam(); scene.add(b);
  const warn = spriteFrom(256, 230, drawWarn, 0.22); scene.add(warn);
  const wglow = glowSprite(0xff3b30, 0.9); scene.add(wglow);
  const c = cam(34);
  const tDrift = at(7.32) - at(5.38);   // shot-local
  return {
    scene, camera: c,
    update(t) {
      // world scrolls past the parked car
      rd.userData.tex.offset.y = -t * 1.6;
      for (const tr of scene.userData.trees) tr.position.z = ((tr.userData.z - t * 19.2) % 80 + 90) % 80 - 10;
      const bob = Math.sin(t * 9) * 0.004;
      cg.position.y = bob; man.root.position.y = 0.25 + bob;
      const look = smooth(range(t, tDrift - 0.1, tDrift + 0.45));
      man.setPose({ ...SEATED, lSh: [-1.05, 0, 0.12], rSh: [lerp(-1.05, -0.55, look), lerp(0, 0.2, look), -0.12], lEl: [-0.55, 0, 0], rEl: [lerp(-0.55, -1.25, look), 0, 0],
                    neck: [lerp(0, 0.35, look), lerp(0, -0.15, look), 0], head: [lerp(0, 0.3, look), 0, 0] });
      man.root.updateMatrixWorld(true);
      const hand = man.j.rWr.localToWorld(V(0, -0.06, 0.03));
      phone.position.copy(hand); phone.lookAt(eyePoint(man)); phone.rotateY(Math.PI); phone.rotateY(Math.PI);
      const from = eyePoint(man);
      const ahead = V(1.32, 1.05, 14);
      const tgt = ahead.clone().lerp(phone.position, look);
      b.set(from, tgt, easeOut(range(t, 0.15, 0.5)), CYAN.clone().lerp(RED, look));
      const blink = look > 0.5 ? (Math.sin((t - tDrift) * 14) > 0 ? 1 : 0.35) : 0;
      warn.position.set(1.32, 1.42, 0.74); warn.material.opacity = blink * smooth(range(t, tDrift + 0.3, tDrift + 0.5));
      warn.scale.setScalar(0.22 * (1 + 0.3 * easeOut(range(t, tDrift + 0.3, tDrift + 0.6))));
      warn.scale.x *= 256 / 230;
      wglow.position.set(1.32, 1.0, 0.75); wglow.material.opacity = 0.7 * blink * smooth(range(t, tDrift + 0.3, tDrift + 0.5));
      const k = easeInOut(range(t, 0, 3.3));
      const sh = shake(t, 0.8, 2);
      place(c, [lerp(-0.15, 0.1, k) + sh[0], 1.95 + sh[1], lerp(2.35, 2.05, k)], [1.25, 1.3, 0.05]);
      return { dof: { focus: 3.0, aperture: 0.0006, maxblur: 0.006 }, bloom: 0.5 };
    },
  };
}

// S4 9.12 -> end : "And someone who can't move their hands could type, with just their eyes."
const KEYS = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];
function keyRect(ch, w, h) {
  for (let r = 0; r < 3; r++) {
    const i = KEYS[r].indexOf(ch);
    if (i < 0) continue;
    const kw = w / 11, kh = h * 0.16, x0 = (w - KEYS[r].length * kw) / 2 + kw * 0.05, y0 = h * 0.42 + r * kh * 1.15;
    return [x0 + i * kw, y0, kw * 0.9, kh];
  }
}
function drawKeyboard(g, w, h, typed = '', active = '', prog = 0) {
  g.fillStyle = '#0d1420'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#e8f4ff'; g.font = `700 ${Math.round(h * 0.17)}px Arial`; g.textAlign = 'left'; g.textBaseline = 'middle';
  g.fillText(typed, w * 0.08, h * 0.2);
  const tw = g.measureText(typed).width;
  g.fillStyle = '#48e6ff'; g.fillRect(w * 0.08 + tw + 8, h * 0.12, 6, h * 0.16);
  for (const row of KEYS) for (const ch of row) {
    const [x, y, kw, kh] = keyRect(ch, w, h);
    const on = ch === active;
    g.fillStyle = on ? `rgba(72,230,255,${0.35 + 0.65 * prog})` : '#1e2a3c';
    g.beginPath(); g.roundRect(x, y, kw, kh, 10); g.fill();
    g.fillStyle = on ? '#001018' : '#b9c6d8'; g.font = `700 ${Math.round(kh * 0.5)}px Arial`; g.textAlign = 'center';
    g.fillText(ch, x + kw / 2, y + kh / 2 + 2);
  }
}
function hook4({ makeStage }) {
  const scene = makeStage({ shadowSize: 3 });
  const woman = person(0x8e44ad, 0x34495e); woman.root.position.set(0, 0.07, 0); scene.add(woman.root);
  const chair = new THREE.Group();
  const cm = new THREE.MeshStandardMaterial({ color: 0x2d3436, roughness: 0.6 });
  const seat = new THREE.Mesh(new RoundedBoxGeometry(0.55, 0.08, 0.5, 3, 0.03), cm); seat.position.set(0, 0.5, -0.05); chair.add(seat);
  const back = new THREE.Mesh(new RoundedBoxGeometry(0.55, 0.6, 0.06, 3, 0.03), cm); back.position.set(0, 0.85, -0.3); chair.add(back);
  for (const s of [-1, 1]) { const wh = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.03, 10, 32), cm); wh.rotation.y = Math.PI / 2; wh.position.set(s * 0.33, 0.32, -0.1); chair.add(wh); }
  chair.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = true; }); scene.add(chair);
  const W2 = 1280, H2 = 720;
  let state = ['', '', 0];
  const tex = canvasTex(W2, H2, (g, w, h) => drawKeyboard(g, w, h, ...state));
  const mon = new THREE.Group();
  const bez = new THREE.Mesh(new RoundedBoxGeometry(1.1, 0.66, 0.05, 3, 0.02), new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.4 }));
  const scr = new THREE.Mesh(new THREE.PlaneGeometry(1.04, 0.585), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
  scr.position.z = 0.027; mon.add(bez, scr);
  const st = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.5, 12), new THREE.MeshStandardMaterial({ color: 0x15171c })); st.position.y = -0.55; mon.add(st);
  const desk = new THREE.Mesh(new RoundedBoxGeometry(1.4, 0.05, 0.6, 3, 0.02), new THREE.MeshStandardMaterial({ color: 0xc8a27a, roughness: 0.6 }));
  desk.position.set(0, 0.76, 1.0); desk.castShadow = desk.receiveShadow = true; scene.add(desk);
  mon.position.set(0, 1.35, 1.1); mon.rotation.y = Math.PI; scene.add(mon);
  const b = beam(); scene.add(b);
  const c = cam(34);
  const word = 'HELLO', dwell = 0.62, t0 = 0.55;
  return {
    scene, camera: c,
    update(t) {
      woman.setPose({ ...SEATED, lSh: [-0.25, 0, 0.1], rSh: [-0.25, 0, -0.1], lEl: [-1.0, 0, 0], rEl: [-1.0, 0, 0], neck: [0.06, 0, 0], head: [0.05, 0, 0] });
      const idx = Math.floor((t - t0) / dwell);
      const prog = clamp(((t - t0) % dwell) / (dwell * 0.85));
      const typed = t < t0 ? '' : word.slice(0, clamp(idx, 0, word.length));
      const active = t >= t0 && idx < word.length ? word[idx] : '';
      state = [typed, active, prog]; tex.redraw();
      const ch = active || word[word.length - 1];
      const [x, y, kw, kh] = keyRect(ch, W2, H2);
      const u = (x + kw / 2) / W2, v = (y + kh / 2) / H2;
      mon.updateMatrixWorld(true);
      const p = scr.localToWorld(V((u - 0.5) * 1.04, (0.5 - v) * 0.585, 0.002));
      b.set(eyePoint(woman), p, easeOut(range(t, 0.1, 0.45)), null, 0.0035);
      const k = easeInOut(range(t, 0, 4.2));
      place(c, [lerp(-1.75, -1.45, k), 1.75, lerp(-0.95, -0.7, k)], [0.1, 1.22, 0.7]);
      return { dof: { focus: 1.9, aperture: 0.0007, maxblur: 0.006 }, bloom: 0.55 };
    },
  };
}

// ================================================================= PROBLEM
function eyeWall(scene, cols = 16, rows = 9, seed = 5) {
  const loader = new THREE.TextureLoader();
  const texs = Array.from({ length: 10 }, (_, i) => { const t = loader.load(`tex/eye${String(i).padStart(2, '0')}.jpg`); t.colorSpace = THREE.SRGBColorSpace; return t; });
  const R = rnd(seed);
  const cards = [];
  const cw = 0.5, ch = 0.25, gap = 0.05;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const k = Math.floor(R() * 10);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(cw, ch), new THREE.MeshStandardMaterial({ map: texs[k], roughness: 0.6 }));
    const x = (c - (cols - 1) / 2) * (cw + gap), y = 1.6 + (r - (rows - 1) / 2) * (ch + gap);
    m.position.set(x, y, (R() - 0.5) * 0.08); m.rotation.z = (R() - 0.5) * 0.04; m.castShadow = true;
    m.userData = { k, x, y, delay: R() }; scene.add(m); cards.push(m);
  }
  return cards;
}
function tagSprite(color) {
  return spriteFrom(128, 128, (g, w, h) => {
    g.clearRect(0, 0, w, h); g.strokeStyle = color; g.lineWidth = 9;
    g.beginPath(); g.arc(w / 2, h / 2, w * 0.3, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(w / 2, 4); g.lineTo(w / 2, h * 0.3); g.moveTo(w / 2, h * 0.7); g.lineTo(w / 2, h - 4);
    g.moveTo(4, h / 2); g.lineTo(w * 0.3, h / 2); g.moveTo(w * 0.7, h / 2); g.lineTo(w - 4, h / 2); g.stroke();
  }, 0.07);
}

// P1 0 -> 6.66 : "But to learn this, a computer needs millions of eye pictures, each tagged with exactly where..."
function prob1({ makeStage }) {
  const scene = makeStage();
  scene.remove(scene.children.find(o => o.material && o.material.isShadowMaterial));
  const cards = eyeWall(scene, 22, 13);
  const tags = [];
  for (const m of cards) for (const [u, v] of PUPILS[m.userData.k]) {
    const s = tagSprite('#4dff9a'); s.userData = { m, u, v }; scene.add(s); tags.push(s);
  }
  const c = cam(30);
  const tTag = at(3.8);
  return {
    scene, camera: c,
    update(t) {
      const k = easeInOut(range(t, 0, 6.5));
      const z = lerp(1.15, 7.5, Math.pow(k, 1.4));
      place(c, [lerp(-0.02, 0.6, k), lerp(1.6, 1.75, k), z], [lerp(-0.02, 0, k), lerp(1.6, 1.6, k), 0]);
      for (const m of cards) {
        const a = easeOut(range(t, 0.2 + m.userData.delay * 1.6, 0.6 + m.userData.delay * 1.6));
        m.material.opacity = 1; m.scale.setScalar(0.6 + 0.4 * a); m.visible = a > 0.01 || Math.abs(m.userData.x) < 0.3 && Math.abs(m.userData.y - 1.6) < 0.2;
        if (Math.abs(m.userData.x) < 0.3 && Math.abs(m.userData.y - 1.6) < 0.2) m.scale.setScalar(1);
      }
      for (const s of tags) {
        const { m, u, v } = s.userData;
        const a = easeOut(range(t, tTag + m.userData.delay * 0.9, tTag + 0.25 + m.userData.delay * 0.9));
        s.visible = m.visible;
        s.position.set(m.position.x + (u - 0.5) * 0.5 * m.scale.x, m.position.y + (0.5 - v) * 0.25 * m.scale.y, m.position.z + 0.01);
        s.material.opacity = a; s.scale.setScalar(0.07 * (1.6 - 0.6 * a));
      }
      return { bloom: 0.25 };
    },
  };
}

// P2 6.66 -> 10.44 : "Collecting them from real people is slow, expensive and private."
function prob2({ makeStage }) {
  const scene = makeStage({ shadowSize: 5 });
  const shirts = [0xe74c3c, 0x27ae60, 0x2980b9, 0xf39c12, 0x8e44ad, 0x16a085, 0xd35400];
  const people = shirts.map((s, i) => { const h = person(s); h.root.position.set(-i * 0.75, 0, 0); h.root.rotation.y = Math.PI / 2; scene.add(h.root); return h; });
  // studio camera on a tripod in front of the queue
  const rig = stand(); rig.position.set(1.6, 0, 0); scene.add(rig);
  const body = new THREE.Mesh(new RoundedBoxGeometry(0.32, 0.22, 0.2, 3, 0.03), new THREE.MeshStandardMaterial({ color: 0x1d1f24, roughness: 0.5 }));
  body.position.set(1.6, 1.62, 0); scene.add(body);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.18, 24), new THREE.MeshStandardMaterial({ color: 0x0c0d10, roughness: 0.2, metalness: 0.6 }));
  lens.rotation.z = Math.PI / 2; lens.position.set(1.42, 1.62, 0); scene.add(lens);
  const flash = glowSprite(0xffffff, 1.4); flash.position.set(1.38, 1.75, 0); scene.add(flash);
  const flashL = new THREE.PointLight(0xffffff, 0, 6); flashL.position.set(1.3, 1.7, 0); scene.add(flashL);
  const locks = people.map(() => { const s = spriteFrom(256, 256, drawLock, 0.26); scene.add(s); return s; });
  const c = cam(30);
  const tPriv = at(9.09) - at(6.16);
  return {
    scene, camera: c,
    update(t) {
      people.forEach((h, i) => {
        const fid = Math.sin(t * 1.3 + i * 1.7);
        h.setPose({ lSh: [0.05, 0, 0.16], rSh: [0.05, 0, -0.16], lEl: [-0.25, 0, 0], rEl: [-0.25, 0, 0],
                    neck: [0, 0.15 * fid, 0], head: [i ? 0.05 : 0, 0, 0], chest: [Math.sin(t * 1.6 + i) * 0.012, 0, 0] });
      });
      const f = [0.35, 1.5, 2.6].reduce((acc, ft) => Math.max(acc, t >= ft ? Math.exp(-(t - ft) * 9) : 0), 0);
      flash.material.opacity = f; flash.scale.setScalar(0.3 + 1.2 * f); flashL.intensity = 18 * f;
      locks.forEach((s, i) => {
        const a = easeOut(range(t, tPriv + i * 0.07, tPriv + 0.25 + i * 0.07));
        s.position.set(-i * 0.75, 2.05 + 0.04 * Math.sin(t * 3 + i), 0);
        s.material.opacity = a; s.scale.setScalar(0.26 * (0.4 + 0.6 * a));
      });
      const k = easeInOut(range(t, 0, 3.9));
      place(c, [lerp(3.9, 3.5, k), 1.95, lerp(3.4, 3.1, k)], [lerp(-0.9, -0.5, k), 1.3, -0.2]);
      return { dof: { focus: 4.2, aperture: 0.0005, maxblur: 0.005 }, bloom: 0.5 };
    },
  };
}

// P3 10.44 -> end : "And the tags? Mostly a guess."
function prob3({ makeStage }) {
  const scene = makeStage();
  scene.remove(scene.children.find(o => o.material && o.material.isShadowMaterial));
  const cards = eyeWall(scene, 7, 5, 9);
  const items = [];
  for (const m of cards) for (const [u, v] of PUPILS[m.userData.k]) {
    const g = tagSprite('#4dff9a'), r = tagSprite('#ff3b30');
    scene.add(g, r); items.push({ m, u, v, g, r, j: [Math.sin(m.userData.x * 37 + u * 11), Math.cos(m.userData.y * 23 + v * 7)] });
  }
  const qs = cards.filter((_, i) => i % 3 === 0).map(m => { const s = spriteFrom(128, 128, drawQ, 0.12); scene.add(s); s.userData.m = m; return s; });
  const c = cam(30);
  const tQ = at(10.98) - at(9.94);
  return {
    scene, camera: c,
    update(t) {
      const drift = smooth(range(t, 0.15, 0.9));
      for (const it of items) {
        const { m, u, v, g, r, j } = it;
        const x = m.position.x + (u - 0.5) * 0.5, y = m.position.y + (0.5 - v) * 0.25;
        const dx = j[0] * 0.05 * drift + Math.sin(t * 7 + j[0] * 5) * 0.004 * drift, dy = j[1] * 0.03 * drift;
        g.position.set(x + dx, y + dy, 0.02); r.position.copy(g.position);
        g.material.opacity = 1 - drift; r.material.opacity = drift;
      }
      qs.forEach((s, i) => {
        const a = easeOut(range(t, tQ + i * 0.05, tQ + 0.25 + i * 0.05));
        s.position.set(s.userData.m.position.x + 0.2, s.userData.m.position.y + 0.12, 0.05);
        s.material.opacity = a; s.scale.setScalar(0.12 * (0.3 + 0.7 * a) * (1 + 0.15 * Math.sin((t - tQ) * 9) * (1 - range(t, tQ + 0.3, tQ + 0.9))));
      });
      const sh = shake(t, 1.2, 4);
      place(c, [0.0 + sh[0], 1.6 + sh[1], lerp(2.5, 2.2, easeInOut(range(t, 0, 2.6)))], [0, 1.6, 0]);
      return { bloom: 0.3 };
    },
  };
}

// ================================================================= films
const HOOK_CUTS = [0, at(2.79), at(5.38), at(8.62), 13.2];
const PROB_CUTS = [0, at(6.16), at(9.94), 13.1];
export const FILMS = {
  hook: {
    DURATION: 13.2, FADES: [],
    SHOTS: [hook1, hook2, hook3, hook4].map((build, i) => ({ start: HOOK_CUTS[i], end: HOOK_CUTS[i + 1], build })),
  },
  problem: {
    DURATION: 13.1, FADES: [],
    SHOTS: [prob1, prob2, prob3].map((build, i) => ({ start: PROB_CUTS[i], end: PROB_CUTS[i + 1], build })),
  },
};
