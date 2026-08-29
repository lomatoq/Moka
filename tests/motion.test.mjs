import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {autoMap,parseBVH,retarget,unwrapTrack,conditionFrames,projectFrame,clipToBVH,validateMotion} from '../web/motion.js';
import {poseTransforms,setupTransforms,demoClip,boneEnd,solveTwoBone,interpolateFrame,deformMesh,skinMatrices} from '../web/core.js';
const rig=JSON.parse(fs.readFileSync(new URL('./fixtures/rig.json',import.meta.url),'utf8'));
const near=(a,b,eps=1e-5)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const motionFromRig=(count=4)=>({schema:'moka.motion/1',name:'test',type:'fixture',fps:30,joints:rig.joints.map(j=>({name:j.id,parent:null})),frames:Array.from({length:count},(_,i)=>({time:i/30,joints:Object.fromEntries(rig.joints.map(j=>[j.id,[j.x/100,-j.y/100,0,1]]))}))});
const mapping=Object.fromEntries(rig.joints.map(j=>[j.id,j.id]));
const bvh=`HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftArm
 {
  OFFSET 1 0 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  JOINT LeftForeArm
  {
   OFFSET 2 0 0
   CHANNELS 3 Zrotation Xrotation Yrotation
   End Site { OFFSET 1 0 0 }
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.033333333
0 0 0 0 0 0 0 0 0 0 0 0
3 4 0 90 0 0 0 0 0 0 0 0
`;

test('Mixamo and namespace aliases resolve physical joint positions',()=>{
 const map=autoMap(['mixamorig:Hips','mixamorig:Spine2','mixamorig:LeftArm','mixamorig:LeftForeArm','mixamorig:LeftHand','mixamorig:LeftUpLeg','mixamorig:LeftLeg','mixamorig:LeftFoot']);
 assert.equal(map.pelvis,'mixamorig:Hips');assert.equal(map.shoulder_l,'mixamorig:LeftArm');assert.equal(map.elbow_l,'mixamorig:LeftForeArm');assert.equal(map.hip_l,'mixamorig:LeftUpLeg');assert.equal(map.knee_l,'mixamorig:LeftLeg');
});
test('BVH hierarchy, root translation, and rotation are evaluated',()=>{
 const m=parseBVH(bvh);near(m.frames[0].joints.LeftForeArm[0],3);near(m.frames[1].joints.LeftArm[0],3);near(m.frames[1].joints.LeftArm[1],5);near(m.frames[1].joints.LeftForeArm[1],7);
});
test('BVH respects declared Euler rotation order',()=>{
 const header=`HIERARCHY ROOT Hips { OFFSET 0 0 0 CHANNELS 3 Zrotation Xrotation Yrotation JOINT Child { OFFSET 0 1 0 CHANNELS 0 End Site { OFFSET 0 1 0 } } } MOTION Frames: 1 Frame Time: 0.1 `;
 const m=parseBVH(header+'90 90 0');near(m.frames[0].joints.Child[0],0);near(m.frames[0].joints.Child[1],0);near(m.frames[0].joints.Child[2],1);
 const other=parseBVH(header.replace('Zrotation Xrotation','Xrotation Zrotation')+'90 90 0');near(other.frames[0].joints.Child[0],-1);near(other.frames[0].joints.Child[2],0);
});
test('Malformed and truncated BVH fail explicitly',()=>{assert.throws(()=>parseBVH(bvh.slice(0,-25)));assert.throws(()=>parseBVH(bvh.replace('Frames: 2','Frames: 999999')));});
test('Circular angle tracks are unwrapped before interpolation',()=>{
 assert.deepEqual(unwrapTrack([170,179,-179,-160]),[170,179,181,200]);
 near(interpolateFrame({frames:[{time:0,angles:{x:179}},{time:1,angles:{x:181}}]},.5).angles.x,180);
});
test('Matching source and target bind poses produce zero offsets',()=>{
 const clip=retarget(motionFromRig(),rig,mapping,{smoothing:0});
 for(const f of clip.frames)for(const angle of Object.values(f.angles))near(angle,0);
});
test('Relative reference starts from the target setup pose',()=>{
 const m=motionFromRig();m.frames.forEach(f=>{f.joints.elbow_l[0]+=.5;});
 const clip=retarget(m,rig,mapping,{mode:'relative',reference:0,smoothing:0});
 for(const angle of Object.values(clip.frames[0].angles))near(angle,0);
});
test('Retarget preserves target lengths under source proportion changes',()=>{
 const m=motionFromRig(8);m.frames.forEach((f,i)=>{f.joints.wrist_l[0]+=i*.3;f.joints.elbow_l[1]+=i*.1;});
 const clip=retarget(m,rig,mapping,{smoothing:0});const bind=setupTransforms(rig);
 for(const f of clip.frames){const p=poseTransforms(rig,f);for(const b of rig.bones){const end=boneEnd(p[b.id]);near(Math.hypot(end[0]-p[b.id].x,end[1]-p[b.id].y),bind[b.id].length);}}
});
test('Root motion is scaled from source units to target pixels',()=>{
 const m=motionFromRig();m.frames.forEach((f,i)=>Object.values(f.joints).forEach(p=>p[0]+=i));
 const clip=retarget(m,rig,mapping,{smoothing:0,rootMotion:true});
 assert.ok(clip.frames.at(-1).root[0]>100);assert.ok(clip.frames.at(-1).root[0]<301);
});
test('Projection supports view angle and anatomical mirroring',()=>{
 const frame={joints:{L:[1,2,3,1],R:[-2,2,1,1]}};
 const p=projectFrame(frame,{wrist_l:'L',wrist_r:'R'},{yaw:90});near(p.wrist_l[0],3);near(p.wrist_l[1],-2);
 const mirror=projectFrame(frame,{wrist_l:'L',wrist_r:'R'},{mirror:true});near(mirror.wrist_l[0],2);near(mirror.wrist_r[0],-1);
});
test('Short confidence gaps interpolate, long gaps are explicitly held',()=>{
 const frames=[0,.1,.2,.8,1.4].map(time=>({time}));const p=[{x:[0,0,0,1]},{},{x:[2,0,0,1]},{},{x:[6,0,0,1]}];
 const c=conditionFrames(p,frames);near(c.frames[1].x[0],1);near(c.frames[3].x[0],2);assert.equal(c.frames[3].x[3],0);assert.equal(c.missing,2);
});
test('Two-bone IK hits reachable targets without changing segment length',()=>{
 const solution=solveTwoBone([0,0],[2,2],2,2,1);near(Math.hypot(...solution.knee),2);near(Math.hypot(solution.knee[0]-solution.target[0],solution.knee[1]-solution.target[1]),2);near(solution.target[0],2);near(solution.target[1],2);
 assert.equal(solveTwoBone([0,0],[10,0],2,2).unreachable,true);
});
test('Loop closure exactly matches first and last pose',()=>{
 const m=motionFromRig(30);m.frames.forEach((f,i)=>{f.joints.elbow_l[1]+=.02*i;});const clip=retarget(m,rig,mapping,{smoothing:.4,loop:true});
 for(const key of Object.keys(clip.frames[0].angles))near(clip.frames[0].angles[key],clip.frames.at(-1).angles[key]);
});
test('Procedural clips are labeled and close cleanly',()=>{
 for(const kind of ['idle','walk','wave']){const c=demoClip(rig,kind);assert.equal(c.source_type,'procedural');for(const key of Object.keys(c.frames[0].angles))near(c.frames[0].angles[key],c.frames.at(-1).angles[key]);}
});
test('2D BVH export roundtrips the actual planar hierarchy',()=>{
 const clip=demoClip(rig,'wave');const motion=parseBVH(clipToBVH(rig,clip));const bind=setupTransforms(rig);
 for(const i of [0,23,60,97,120]){const pose=poseTransforms(rig,clip.frames[i]);for(const bone of rig.bones){const p=motion.frames[i].joints[bone.id];near(p[0],pose[bone.id].x-bind.root.x,.002);near(p[1],bind.root.y-pose[bone.id].y,.002);}}
});
test('CPU and browser skinning conventions preserve rest vertices',()=>{
 const mesh={vertices:[[230,340],[260,380]],weights:[[{bone:'upper_arm_r',weight:.7},{bone:'forearm_r',weight:.3}],[{bone:'torso',weight:1}]]};
 assert.deepEqual([...deformMesh(mesh,skinMatrices(rig))],[230,340,260,380]);
});
test('Non-finite and duplicate-time motion is rejected',()=>{
 const m=motionFromRig();m.frames[1].time=0;assert.throws(()=>validateMotion(m));m.frames[1].time=1/30;m.frames[1].joints.pelvis[0]=NaN;assert.throws(()=>validateMotion(m));
});
