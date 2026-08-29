/** Canonical 2D rig math. X right, Y down; clockwise rotation deltas in degrees. */
export const DEG = Math.PI / 180;
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const wrap = d => ((d + 180) % 360 + 360) % 360 - 180;
export const lerp = (a, b, t) => a + (b - a) * t;
export const clone = value => structuredClone(value);
export function uid(){if(globalThis.crypto?.randomUUID)return crypto.randomUUID();const bytes=crypto.getRandomValues(new Uint8Array(16));bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;const h=[...bytes].map(b=>b.toString(16).padStart(2,'0')).join('');return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20)}`;}

export function setupTransforms(rig) {
  const joints = Object.fromEntries(rig.joints.map(j => [j.id, j]));
  const out = {};
  for (const b of rig.bones) {
    const a = joints[b.start], z = joints[b.end];
    if (!a || !z) throw new Error(`Unknown joint on ${b.id}`);
    out[b.id] = { x: a.x, y: a.y, angle: b.start === b.end ? 0 : Math.atan2(z.y-a.y, z.x-a.x), length: Math.hypot(z.x-a.x, z.y-a.y) };
  }
  return out;
}
export function localPoint(p, t) {
  const dx = p[0]-t.x, dy = p[1]-t.y, c = Math.cos(t.angle), s = Math.sin(t.angle);
  return [c*dx+s*dy, -s*dx+c*dy];
}
export function worldPoint(p, t) {
  const c = Math.cos(t.angle), s = Math.sin(t.angle);
  return [t.x+c*p[0]-s*p[1], t.y+s*p[0]+c*p[1]];
}
export function poseTransforms(rig, frame = {}) {
  const bind = setupTransforms(rig), posed = {}, angles = frame.angles || {}, root = frame.root || [0, 0];
  for (const b of rig.bones) {
    const own = bind[b.id], delta = (angles[b.id] || 0)*DEG;
    if (!b.parent) posed[b.id] = {...own, x: own.x+root[0], y: own.y+root[1], angle: own.angle+delta};
    else {
      if (!posed[b.parent]) throw new Error('Parents must precede children in the rig');
      const offset = localPoint([own.x, own.y], bind[b.parent]);
      const [x, y] = worldPoint(offset, posed[b.parent]);
      posed[b.id] = {...own, x, y, angle: posed[b.parent].angle+own.angle-bind[b.parent].angle+delta};
    }
  }
  return posed;
}
export function skinMatrices(rig, frame = {}) {
  const bind = setupTransforms(rig), pose = poseTransforms(rig, frame), matrices = {};
  for (const b of rig.bones) {
    const a = bind[b.id], p = pose[b.id], c = Math.cos(p.angle-a.angle), s = Math.sin(p.angle-a.angle);
    matrices[b.id] = [c, s, -s, c, p.x-c*a.x+s*a.y, p.y-s*a.x-c*a.y];
  }
  return matrices;
}
export function deformMesh(mesh, matrices) {
  const out = new Float32Array(mesh.vertices.length*2);
  mesh.vertices.forEach((p, i) => {
    for (const influence of mesh.weights[i]) {
      const m = matrices[influence.bone];
      if (!m) throw new Error(`Mesh references unknown bone ${influence.bone}`);
      const w = influence.weight;
      out[2*i] += w*(m[0]*p[0]+m[2]*p[1]+m[4]);
      out[2*i+1] += w*(m[1]*p[0]+m[3]*p[1]+m[5]);
    }
  });
  return out;
}
export function interpolateFrame(clip, time) {
  const frames = clip?.frames;
  if (!frames?.length) return {time, angles: {}, root: [0, 0]};
  if (time <= frames[0].time) return frames[0];
  if (time >= frames.at(-1).time) return frames.at(-1);
  let lo = 0, hi = frames.length-1;
  while (hi-lo > 1) { const m = (lo+hi)>>1; if (frames[m].time <= time) lo = m; else hi = m; }
  const a = frames[lo], b = frames[hi], t = (time-a.time)/(b.time-a.time), angles = {};
  for (const name of new Set([...Object.keys(a.angles || {}), ...Object.keys(b.angles || {})])) {
    // Retargeted tracks are unwrapped before storage. Linear interpolation then
    // preserves genuine multi-turn motion instead of introducing +/-180 jumps.
    angles[name] = lerp(a.angles?.[name] || 0, b.angles?.[name] || 0, t);
  }
  return {time, angles, root: [lerp(a.root?.[0] || 0, b.root?.[0] || 0, t), lerp(a.root?.[1] || 0, b.root?.[1] || 0, t)]};
}
export function boneEnd(t) { return [t.x+Math.cos(t.angle)*t.length, t.y+Math.sin(t.angle)*t.length]; }
export function solveTwoBone(a, target, l1, l2, bend = 1) {
  const dx = target[0]-a[0], dy = target[1]-a[1];
  const d = clamp(Math.hypot(dx, dy), Math.abs(l1-l2)+1e-5, l1+l2-1e-5);
  const direction = Math.atan2(dy, dx);
  const offset = Math.acos(clamp((l1*l1+d*d-l2*l2)/(2*l1*d), -1, 1));
  const upper = direction + Math.sign(bend || 1)*offset;
  const knee = [a[0]+Math.cos(upper)*l1, a[1]+Math.sin(upper)*l1];
  const reachable = [a[0]+Math.cos(direction)*d, a[1]+Math.sin(direction)*d];
  const lower = Math.atan2(reachable[1]-knee[1], reachable[0]-knee[0]);
  return {upper, lower, knee, target: reachable, unreachable: Math.hypot(dx, dy) > l1+l2};
}
export function demoClip(rig, kind = 'idle') {
  const duration = kind === 'walk' ? 2 : 4, fps = 30, frames = [];
  const known = new Set(rig.bones.map(b => b.id));
  for (let i=0; i<=duration*fps; i++) {
    const time = i/fps, phase = time/duration*Math.PI*2, angles = {}, root = [0, 0];
    const put = (id, v) => { if (known.has(id)) angles[id] = v; };
    put('torso', Math.sin(phase)*1.4); put('head', -Math.sin(phase)*1.2);
    for (const side of ['l', 'r']) {
      const sign = side === 'l' ? 1 : -1;
      put(`upper_arm_${side}`, sign*Math.sin(phase)*1.6);
      put(`forearm_${side}`, sign*Math.sin(phase)*2);
      if (kind === 'walk') {
        put(`upper_arm_${side}`, sign*Math.sin(phase)*27);
        put(`thigh_${side}`, -sign*Math.sin(phase)*23);
        put(`shin_${side}`, Math.sin(phase)*sign*17);
        put(`foot_${side}`, Math.sin(phase)*sign*-10);
        root[1] = -Math.sin(phase*2)*3;
      }
    }
    if (kind === 'wave') {
      const raise = Math.sin(Math.PI*time/duration)**2;
      put('upper_arm_l', -82*raise);
      put('forearm_l', (-58+Math.sin(phase*4)*15)*raise);
      put('hand_l', Math.sin(phase*4)*14*raise);
    }
    for (const id of known) if (id.startsWith('tail')) put(id, Math.sin(phase+(id==='tail_tip'?.5:0))*8);
    frames.push({time, angles, root});
  }
  return {id: uid(), name: {idle:'Idle · procedural', walk:'Walk · procedural', wave:'Wave · procedural'}[kind], fps, duration, source_type:'procedural', frames, diagnostics:{note:'Procedural test motion, not extracted mocap.'}};
}
export function distanceToSegment(p, a, b) {
  const dx=b[0]-a[0], dy=b[1]-a[1], d=dx*dx+dy*dy;
  const t=d?clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d,0,1):0;
  return Math.hypot(p[0]-a[0]-dx*t,p[1]-a[1]-dy*t);
}
