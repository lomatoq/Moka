/** BVH, canonical motion, confidence-aware projection and proportion-preserving retargeting. */
import {DEG, clamp, wrap, lerp, setupTransforms, poseTransforms, boneEnd, solveTwoBone, interpolateFrame, uid} from './core.js';

const ALIASES = {
  pelvis: ['hips','hip','pelvis','root'], chest: ['spine2','upperchest','chest','spine1','spine'],
  neck: ['neck','neck1'], head_tip: ['headtopend','headend','head'],
  shoulder_l: ['leftarm','leftupperarm','lupperarm','upperarml','shoulderleft','leftshoulder'],
  elbow_l: ['leftforearm','leftlowerarm','lforearm','forearml','lowerarml','leftelbow'],
  wrist_l: ['lefthand','lhand','handl','leftwrist'], hand_l: ['lefthandmiddle1','leftmiddleproximal','leftmiddlefinger1','leftmiddle1'],
  shoulder_r: ['rightarm','rightupperarm','rupperarm','upperarmr','shoulderright','rightshoulder'],
  elbow_r: ['rightforearm','rightlowerarm','rforearm','forearmr','lowerarmr','rightelbow'],
  wrist_r: ['righthand','rhand','handr','rightwrist'], hand_r: ['righthandmiddle1','rightmiddleproximal','rightmiddlefinger1','rightmiddle1'],
  hip_l: ['leftupleg','leftupperleg','lthigh','thighl','leftthigh','lefthip'],
  knee_l: ['leftleg','leftlowerleg','lshin','shinl','calfl','leftknee'],
  ankle_l: ['leftfoot','lfoot','footl','leftankle'], toe_l: ['lefttoebase','lefttoes','lefttoe','toel'],
  hip_r: ['rightupleg','rightupperleg','rthigh','thighr','rightthigh','righthip'],
  knee_r: ['rightleg','rightlowerleg','rshin','shinr','calfr','rightknee'],
  ankle_r: ['rightfoot','rfoot','footr','rightankle'], toe_r: ['righttoebase','righttoes','righttoe','toer'],
};
export const ROLES = Object.keys(ALIASES);
export function normalizeName(name) { return name.toLowerCase().replace(/.*[:|]/,'').replace(/^mixamorig\d*/, '').replace(/[^a-z0-9]/g,''); }
export function autoMap(names, targetRoles = ROLES) {
  const normalized = names.map(name => [name, normalizeName(name)]), mapping = {};
  for (const role of targetRoles) {
    const exact = normalized.find(([n]) => n === role);
    if (exact) { mapping[role] = exact[0]; continue; }
    let found;
    for (const alias of ALIASES[role] || [normalizeName(role)]) {
      found = normalized.find(([, n]) => n === alias);
      if (found) break;
    }
    mapping[role] = found?.[0] || '';
  }
  return mapping;
}
const vec = (a,b,t) => [lerp(a[0],b[0],t),lerp(a[1],b[1],t),lerp(a[2]||0,b[2]||0,t),Math.min(a[3]??1,b[3]??1)];
export function canonicalFrame(frame, mapping) {
  const raw = frame.joints || {}, out = {};
  for (const [role, source] of Object.entries(mapping)) if (source && raw[source]) out[role] = [...raw[source]];
  if (!out.pelvis && out.hip_l && out.hip_r) out.pelvis = vec(out.hip_l,out.hip_r,.5);
  if (!out.chest && out.shoulder_l && out.shoulder_r) out.chest = vec(out.shoulder_l,out.shoulder_r,.5);
  if (!out.neck && out.chest && out.head_tip) out.neck = vec(out.chest,out.head_tip,.42);
  if (!out.head_tip && out.neck && out.chest) out.head_tip = vec(out.chest,out.neck,2.2);
  for (const s of ['l','r']) {
    if (!out[`hand_${s}`] && out[`wrist_${s}`] && out[`elbow_${s}`]) out[`hand_${s}`] = vec(out[`elbow_${s}`],out[`wrist_${s}`],1.28);
    if (!out[`toe_${s}`] && out[`ankle_${s}`] && out[`knee_${s}`]) {
      const a=out[`ankle_${s}`],k=out[`knee_${s}`],len=Math.hypot(a[0]-k[0],a[1]-k[1],a[2]-k[2]);
      out[`toe_${s}`] = [a[0]+(s==='l'?1:-1)*len*.22,a[1],a[2]||0,(a[3]??1)*.5];
    }
  }
  return out;
}
export function projectFrame(frame, mapping, options = {}) {
  const source = canonicalFrame(frame,mapping), out={}, yaw=(options.yaw||0)*DEG;
  for (const key of Object.keys(source)) {
    const opposite = key.endsWith('_l')?key.slice(0,-2)+'_r':key.endsWith('_r')?key.slice(0,-2)+'_l':key;
    const p = options.mirror ? (source[opposite] || source[key]) : source[key];
    // Input coordinates are consistently Y-up; projection returns image Y-down.
    const horizontal = p[0]*Math.cos(yaw)+(p[2]||0)*Math.sin(yaw);
    out[key] = [(options.mirror?-1:1)*horizontal,-p[1],-(p[0])*Math.sin(yaw)+(p[2]||0)*Math.cos(yaw),p[3]??1];
  }
  return out;
}
export function validateMotion(motion) {
  if (motion?.schema !== 'moka.motion/1' || !Array.isArray(motion.frames) || !motion.frames.length) throw new Error('Unsupported or empty motion data');
  if (motion.frames.length>18001 || (motion.joints?.length||0)>256) throw new Error('Motion exceeds the frame or joint limit');
  let previous=-1;
  for (const f of motion.frames) {
    if (!Number.isFinite(f.time) || f.time<0 || f.time<=previous || f.time>600) throw new Error('Motion timestamps must increase and stay within 10 minutes');
    previous=f.time;
    for (const p of Object.values(f.joints||{})) if (!Array.isArray(p) || p.length<3 || !p.slice(0,3).every(Number.isFinite)) throw new Error('Motion contains an invalid joint coordinate');
  }
  return motion;
}

function qmul(a,b) {
  const [ax,ay,az,aw]=a,[bx,by,bz,bw]=b;
  return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz];
}
function qrotate(q,v) {
  const p=qmul(q,[...v,0]),r=qmul(p,[-q[0],-q[1],-q[2],q[3]]);
  return r.slice(0,3);
}
function qaxis(axis,degrees) {
  const v=[0,0,0,Math.cos(degrees*DEG/2)]; v[axis]=Math.sin(degrees*DEG/2); return v;
}
export function parseBVH(text, filename='motion.bvh') {
  if (text.length>48_000_000) throw new Error('BVH exceeds the input size budget');
  const tokens=text.match(/[{}]|[^\s{}]+/g)||[]; let pos=0, channelCount=0;
  const joints=[], channels=[];
  const next=()=>{if(pos>=tokens.length)throw new Error('Truncated BVH');return tokens[pos++];};
  const expect=value=>{const got=next();if(got!==value)throw new Error(`Expected ${value}, got ${got}`);};
  const number=()=>{const n=Number(next());if(!Number.isFinite(n))throw new Error('Invalid BVH number');return n;};
  function joint(parent, end=false) {
    const index=joints.length;
    if(index>=256)throw new Error('BVH exceeds 256 joints');
    const name=end?`${joints[parent].name}_EndSite`:next();
    if(joints.some(j=>j.name===name))throw new Error(`Duplicate joint ${name}`);
    const j={name,parent,offset:[0,0,0],channels:[]};joints.push(j);expect('{');
    while(tokens[pos]!=='}') {
      const token=next();
      if(token==='OFFSET')j.offset=[number(),number(),number()];
      else if(token==='CHANNELS') {
        const count=number();if(!Number.isInteger(count)||count<0||count>6)throw new Error('Invalid BVH channel count');
        for(let i=0;i<count;i++){const c=next();if(!/^[XYZ](position|rotation)$/.test(c))throw new Error(`Unknown BVH channel ${c}`);j.channels.push({name:c,index:channelCount++});}
      } else if(token==='JOINT')joint(index);
      else if(token==='End'){expect('Site');joint(index,true);}
      else throw new Error(`Unexpected BVH token ${token}`);
    }
    expect('}');return index;
  }
  expect('HIERARCHY');expect('ROOT');joint(-1);expect('MOTION');
  const header=next();if(!/^Frames:?$/i.test(header))throw new Error('BVH has no frame count');
  if(header==='Frames')expect(':');
  const count=number();if(!Number.isInteger(count)||count<1||count>18001)throw new Error('BVH must contain 1–18001 frames');
  expect('Frame');const timeToken=next();if(timeToken!=='Time:'&&timeToken!=='Time')throw new Error('BVH has no frame time');if(timeToken==='Time')expect(':');
  const dt=number();if(dt<=0||dt*count>600)throw new Error('Invalid BVH frame time/duration');
  function evaluate(values) {
    const world=[], output={};
    joints.forEach((j,i)=>{
      let p=[...j.offset],q=[0,0,0,1];
      for(const ch of j.channels){const axis='XYZ'.indexOf(ch.name[0]),v=values[ch.index]||0;if(ch.name.endsWith('position'))p[axis]+=v;else q=qmul(q,qaxis(axis,v));}
      if(j.parent>=0){const parent=world[j.parent],r=qrotate(parent.q,p);p=r.map((v,k)=>v+parent.p[k]);q=qmul(parent.q,q);}
      world[i]={p,q};output[j.name]=[...p,1];
    });return output;
  }
  const frames=[];
  for(let i=0;i<count;i++){const values=[];for(let c=0;c<channelCount;c++)values.push(number());frames.push({time:i*dt,joints:evaluate(values)});}
  if(pos<tokens.length && tokens.slice(pos).some(t=>t.trim()))throw new Error('BVH has unexpected extra motion values');
  return validateMotion({schema:'moka.motion/1',name:filename.replace(/\.bvh$/i,''),type:'bvh',coordinate_space:'world',fps:1/dt,
    joints:joints.map(j=>({name:j.name,parent:j.parent<0?null:joints[j.parent].name})),frames,reference:{time:0,joints:evaluate(Array(channelCount).fill(0))},
    diagnostics:{note:'BVH declared Euler channel order and hierarchy were evaluated before retargeting.'}});
}

class OneEuro {
  constructor(cutoff=1.7,beta=.015){this.cutoff=cutoff;this.beta=beta;this.x=null;this.dx=0;}
  alpha(dt,cutoff){return 1/(1+1/(2*Math.PI*cutoff*Math.max(dt,1e-6)));}
  next(x,dt){if(this.x===null){this.x=x;return x;}const dx=(x-this.x)/Math.max(dt,1e-6),ad=this.alpha(dt,1);this.dx=lerp(this.dx,dx,ad);this.x=lerp(this.x,x,this.alpha(dt,this.cutoff+this.beta*Math.abs(this.dx)));return this.x;}
}
export function unwrapTrack(values) {
  const out=[];let prev=0;
  values.forEach((v,i)=>{prev=i?prev+wrap(v-prev):v;out.push(prev);});return out;
}
export function conditionFrames(projected, frames, threshold=.35) {
  const keys=new Set(projected.flatMap(p=>Object.keys(p))),out=projected.map(p=>({...p}));let missing=0;
  for(const key of keys){
    const valid=[];for(let i=0;i<out.length;i++)if(out[i][key] && out[i][key][3]>=threshold)valid.push(i);
    if(!valid.length){for(const f of out)delete f[key];continue;}
    let vpos=0;
    for(let i=0;i<out.length;i++){
      if(out[i][key] && out[i][key][3]>=threshold)continue;
      missing++;while(vpos+1<valid.length&&valid[vpos+1]<i)vpos++;
      const before=valid[vpos]<=i?valid[vpos]:null,after=valid[vpos]>=i?valid[vpos]:valid[vpos+1];
      if(before!==null&&after!==undefined&&frames[after].time-frames[before].time<=.35){
        const t=(frames[i].time-frames[before].time)/(frames[after].time-frames[before].time||1);out[i][key]=vec(out[before][key],out[after][key],t);
      }else{out[i][key]=[...out[before??after??valid[0]][key]];out[i][key][3]=0;}
    }
  }
  return {frames:out,missing};
}

export function retarget(motion, rig, mapping, options={}) {
  validateMotion(motion);
  const bind=setupTransforms(rig), bones=rig.bones, fps=motion.fps||30;
  const useScreen=options.projection==='screen'&&motion.screen_frames?.length===motion.frames.length;
  const sourceFrames=useScreen?motion.screen_frames:motion.frames;
  const raw=sourceFrames.map(f=>projectFrame(f,mapping,{...options,yaw:useScreen?0:options.yaw}));
  const conditioned=conditionFrames(raw,sourceFrames,options.confidence??.35), projected=conditioned.frames;
  const referenceIndex=clamp(Math.round(options.reference||0),0,sourceFrames.length-1);
  const reference=options.useBind && motion.reference && !useScreen?projectFrame(motion.reference,mapping,options):projected[referenceIndex];
  function directions(p){const dirs={root:0};for(const b of bones){if(b.id==='root')continue;const a=p[b.start],z=p[b.end];if(a&&z&&Math.hypot(z[0]-a[0],z[1]-a[1])>1e-5)dirs[b.id]=Math.atan2(z[1]-a[1],z[0]-a[0])/DEG;}return dirs;}
  const refDir=directions(reference),mode=options.mode||'absolute';
  const tracks=Object.fromEntries(bones.map(b=>[b.id,[]])),frames=[];
  const refPelvis=projected[referenceIndex].pelvis||[0,0,0,1];
  const bodyScale=(()=>{
    const lens=[];for(const b of bones){if(!['torso','thigh_l','thigh_r','shin_l','shin_r'].includes(b.id))continue;const a=reference[b.start],z=reference[b.end];if(a&&z){const len=Math.hypot(z[0]-a[0],z[1]-a[1]);if(len>1e-5)lens.push(bind[b.id].length/len);}}
    lens.sort((a,b)=>a-b);return lens.length?lens[Math.floor(lens.length/2)]:1;
  })();
  let weakDirections=0;
  sourceFrames.forEach((f,i)=>{
    const dirs=directions(projected[i]),angles={};
    for(const b of bones){
      if(b.id==='root'){angles.root=0;tracks.root.push(0);continue;}
      let value=tracks[b.id].at(-1)||0;
      if(dirs[b.id]!==undefined && dirs[b.parent]!==undefined){
        const local=wrap(dirs[b.id]-dirs[b.parent]);
        if(mode==='relative'){
          if(refDir[b.id]!==undefined&&refDir[b.parent]!==undefined)value=wrap(local-wrap(refDir[b.id]-refDir[b.parent]));
        }else value=wrap(local-(bind[b.id].angle-bind[b.parent].angle)/DEG);
      }else weakDirections++;
      tracks[b.id].push(value);angles[b.id]=value;
    }
    const pelvis=projected[i].pelvis||refPelvis;
    const root=options.rootMotion?[(pelvis[0]-refPelvis[0])*bodyScale,(pelvis[1]-refPelvis[1])*bodyScale]:[0,0];
    frames.push({time:f.time,angles,root});
  });
  const smoothing=clamp(Number(options.smoothing??.45),0,1);
  for(const b of bones){
    const values=unwrapTrack(tracks[b.id]);
    if(smoothing>0){const filter=new OneEuro(lerp(10,.9,smoothing),.008);values.forEach((v,i)=>{values[i]=filter.next(v,i?frames[i].time-frames[i-1].time:1/fps);});}
    values.forEach((v,i)=>{frames[i].angles[b.id]=v;});
  }
  const rootFilters=[new OneEuro(2,.015),new OneEuro(2,.015)];
  frames.forEach((f,i)=>f.root=f.root.map((v,k)=>rootFilters[k].next(v,i?f.time-frames[i-1].time:1/fps)));
  let contacts=0;
  if(options.footLock)contacts=lockFeet(rig,frames,projected);
  if(options.loop && frames.length>2){
    const first=frames[0],last=frames.at(-1),duration=last.time-first.time||1;
    const delta=Object.fromEntries(bones.map(b=>[b.id,last.angles[b.id]-first.angles[b.id]]));
    const rootDelta=last.root.map((v,k)=>v-first.root[k]);
    for(const f of frames){const t=(f.time-first.time)/duration,s=t*t*(3-2*t);for(const b of bones)f.angles[b.id]-=delta[b.id]*s;f.root=f.root.map((v,k)=>v-rootDelta[k]*s);}
  }
  const missingRoles=rig.joints.filter(j=>!reference[j.id]).map(j=>j.id);
  return {id:uid(),name:motion.name||'Retargeted motion',fps,duration:frames.at(-1).time,source_type:motion.type,
    frames,retarget:{mapping,options},diagnostics:{weakDirections,filledLowConfidenceSamples:conditioned.missing,missingRoles,contactFrames:contacts,
      sourceFrames:frames.length,proportions:'target bone lengths preserved',projection:useScreen?'observed-2D':'projected-3D',
      warning:rig.preset==='quadruped'?'Human arm roles drive front legs unless mapping is changed.':null}};
}

export function lockFeet(rig,frames,projected) {
  const bind=setupTransforms(rig),byId=Object.fromEntries(rig.bones.map(b=>[b.id,b]));let count=0;
  for(const side of ['l','r']){
    const upper=`thigh_${side}`,lower=`shin_${side}`,ankle=`ankle_${side}`;
    if(!byId[upper]||!byId[lower])continue;
    const points=projected.map(p=>p[ankle]).filter(Boolean);if(points.length<3)continue;
    const heights=points.map(p=>p[1]).sort((a,b)=>a-b),ground=heights[Math.floor(heights.length*.9)];
    const sourceLengths=projected.map(p=>p[ankle]&&p[`hip_${side}`]?Math.hypot(p[ankle][0]-p[`hip_${side}`][0],p[ankle][1]-p[`hip_${side}`][1]):0).filter(v=>v>0);
    const scale=sourceLengths.reduce((a,b)=>a+b,0)/Math.max(sourceLengths.length,1)||1;
    let anchor=null,stable=0;
    for(let i=1;i<frames.length;i++){
      const p=projected[i][ankle],prev=projected[i-1][ankle];if(!p||!prev||p[3]<.35){anchor=null;stable=0;continue;}
      const dt=frames[i].time-frames[i-1].time,speed=Math.hypot(p[0]-prev[0],p[1]-prev[1])/Math.max(dt,1e-4);
      const planted=speed<scale*(anchor?.13:.085)&&p[1]>ground-scale*.07;
      if(!planted){anchor=null;stable=0;continue;}
      stable++;const pose=poseTransforms(rig,frames[i]);
      if(!anchor){if(stable<2)continue;anchor=boneEnd(pose[lower]);}
      const a=[pose[upper].x,pose[upper].y],bend=Math.sign(Math.sin(pose[lower].angle-pose[upper].angle))||1;
      const ik=solveTwoBone(a,anchor,bind[upper].length,bind[lower].length,-bend);
      if(ik.unreachable){anchor=null;stable=0;continue;}
      const parent=byId[upper].parent;
      frames[i].angles[upper]=(ik.upper-pose[parent].angle-bind[upper].angle+bind[parent].angle)/DEG;
      frames[i].angles[lower]=(ik.lower-ik.upper-bind[lower].angle+bind[upper].angle)/DEG;
      count++;
    }
  }
  // IK atan2 may cross the branch cut; re-unwrap all tracks before playback.
  for(const b of rig.bones){const v=unwrapTrack(frames.map(f=>f.angles[b.id]||0));frames.forEach((f,i)=>f.angles[b.id]=v[i]);}
  return count;
}

export function landmarksToFrame(landmarks, time, width=1, height=1, world=false) {
  const p=landmarks.map(l=>[l.x*(world?1:width),-l.y*(world?1:height),(l.z||0)*(world?1:width),Math.min(l.visibility??1,l.presence??1)]);
  const pairs={shoulder_l:11,shoulder_r:12,elbow_l:13,elbow_r:14,wrist_l:15,wrist_r:16,
    hip_l:23,hip_r:24,knee_l:25,knee_r:26,ankle_l:27,ankle_r:28,toe_l:31,toe_r:32};
  const joints=Object.fromEntries(Object.entries(pairs).map(([k,i])=>[k,p[i]]));
  joints.pelvis=vec(p[23],p[24],.5);joints.chest=vec(p[11],p[12],.5);
  joints.neck=vec(joints.chest,p[0],.42);joints.head_tip=vec(joints.chest,p[0],1.5);
  joints.hand_l=vec(p[17],p[19],.5);joints.hand_r=vec(p[18],p[20],.5);
  return {time,joints};
}

export function clipToBVH(rig,clip) {
  // Export the actual animated 2D hierarchy embedded in the XY plane. This is
  // intentionally NOT advertised as recovered volumetric motion or FBX.
  const bind=setupTransforms(rig),byId=Object.fromEntries(rig.bones.map(b=>[b.id,b])),children={};
  for(const b of rig.bones){const p=b.parent||'';(children[p]??=[]).push(b);}
  const lines=['HIERARCHY'],order=[];
  function emit(b,depth){
    order.push(b);const indent='  '.repeat(depth),p=b.parent?bind[b.parent]:null;
    lines.push(`${indent}${b.parent?'JOINT':'ROOT'} ${b.id}`,`${indent}{`);
    let offset=[0,0];if(p){const dx=bind[b.id].x-p.x,dy=bind[b.id].y-p.y;offset=[dx,-dy];}
    // BVH rest offsets are expressed in a common zero-rotation hierarchy.
    lines.push(`${indent}  OFFSET ${offset[0].toFixed(5)} ${offset[1].toFixed(5)} 0`);
    lines.push(`${indent}  CHANNELS ${b.parent?'3 Zrotation Xrotation Yrotation':'6 Xposition Yposition Zposition Zrotation Xrotation Yrotation'}`);
    if(children[b.id]?.length)children[b.id].forEach(c=>emit(c,depth+1));
    else{const t=bind[b.id],dx=Math.cos(t.angle)*t.length,dy=-Math.sin(t.angle)*t.length;lines.push(`${indent}  End Site`,`${indent}  {`,`${indent}    OFFSET ${dx.toFixed(5)} ${dy.toFixed(5)} 0`,`${indent}  }`);}
    lines.push(`${indent}}`);
  }
  emit(rig.bones[0],0);
  const fps=clip.fps||30,count=Math.max(1,Math.round(clip.duration*fps)+1);
  lines.push('MOTION',`Frames: ${count}`,`Frame Time: ${(1/fps).toFixed(9)}`);
  // Use pose-world rotation DELTAS to cancel setup orientations correctly.
  for(let i=0;i<count;i++){
    const f=interpolateFrame(clip,Math.min(i/fps,clip.duration));
    const pose=poseTransforms(rig,f),values=[];
    for(const b of order){
      if(!b.parent)values.push((f.root?.[0]||0).toFixed(6),(-(f.root?.[1]||0)).toFixed(6),'0');
      const delta=(pose[b.id].angle-bind[b.id].angle)-(b.parent?pose[b.parent].angle-bind[b.parent].angle:0);
      values.push((-delta/DEG).toFixed(6),'0','0');
    }
    lines.push(values.join(' '));
  }
  return lines.join('\n');
}
