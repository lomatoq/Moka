import {validateMotion} from './motion.js';

/** Sample world-space bone positions; do not assume a source bone's Euler axes. */
export async function load3D(file){
  if(file.size>128*1024*1024)throw new Error('3D file exceeds 128 MB');
  const THREE=await import('three');
  const manager=new THREE.LoadingManager();
  manager.setURLModifier(url=>{
    if(url.startsWith('blob:')||url.startsWith('data:'))return url;
    throw new Error('External assets are disabled. Export a self-contained GLB or embedded FBX.');
  });
  let object,animations;const ext=file.name.split('.').at(-1).toLowerCase();
  if(ext==='fbx'){
    const {FBXLoader}=await import('three/addons/loaders/FBXLoader.js');object=new FBXLoader(manager).parse(await file.arrayBuffer(),'');animations=object.animations;
  }else{
    const {GLTFLoader}=await import('three/addons/loaders/GLTFLoader.js');const bytes=await file.arrayBuffer();
    if(ext==='gltf'){const json=JSON.parse(new TextDecoder().decode(bytes));for(const entry of [...(json.buffers||[]),...(json.images||[])])if(entry.uri&&!entry.uri.startsWith('data:'))throw new Error('Use GLB, or a glTF with embedded data URIs. External files are not fetched.');}
    const parsed=await new GLTFLoader(manager).parseAsync(ext==='gltf'?new TextDecoder().decode(bytes):bytes,'');object=parsed.scene;animations=parsed.animations;
  }
  const bones=[];object.traverse(node=>{if(node.isBone)bones.push(node);});
  if(!bones.length)throw new Error('This file has no skeleton. Import a skinned, animated model.');
  if(bones.length>256)throw new Error('Source skeleton exceeds 256 bones. Remove unused facial/finger bones before importing.');
  if(!animations?.length)throw new Error('The file contains no animation clips.');
  const names=new Set();for(const bone of bones){if(!bone.name||names.has(bone.name))throw new Error('Source bones need unique, non-empty names');names.add(bone.name);}
  object.updateMatrixWorld(true);const v=new THREE.Vector3();
  const read=()=>Object.fromEntries(bones.map(b=>{b.getWorldPosition(v);return [b.name,[v.x,v.y,v.z,1]];}));
  const reference={time:0,joints:read()};
  return {names:animations.map((a,i)=>({index:i,name:a.name||`Animation ${i+1}`,duration:a.duration})),
    async sample(index=0,{fps=30,onProgress=()=>{},signal}={}){
      const animation=animations[index];if(!animation)throw new Error('Choose an existing animation');
      if(animation.duration>600||animation.duration<=0)throw new Error('Animation duration must be within 0–600 seconds');
      const count=Math.min(18001,Math.ceil(animation.duration*fps)+1),frames=[],mixer=new THREE.AnimationMixer(object);
      mixer.clipAction(animation).setLoop(THREE.LoopOnce,1).play();
      try{for(let i=0;i<count;i++){if(signal?.aborted)throw new Error('Motion import cancelled');const time=Math.min(i/fps,animation.duration);mixer.setTime(Math.min(time,animation.duration-1e-7));object.updateMatrixWorld(true);frames.push({time,joints:read()});if(i%45===0){onProgress(i/count,`Sampling skeletal motion ${i+1}/${count}`);await new Promise(r=>setTimeout(r,0));}}}
      finally{mixer.stopAllAction();mixer.uncacheRoot(object);}
      return validateMotion({schema:'moka.motion/1',name:animation.name||file.name.replace(/\.[^.]+$/,''),type:ext,fps,coordinate_space:'world',joints:bones.map(b=>({name:b.name,parent:b.parent?.isBone?b.parent.name:null})),reference,frames,
        diagnostics:{sourceFile:file.name,sourceAnimation:index,totalAnimations:animations.length,note:'Evaluated source hierarchy and sampled world positions; target proportions are applied during retargeting.'}});
    },
    dispose(){object.traverse(n=>{n.geometry?.dispose();const materials=Array.isArray(n.material)?n.material:n.material?[n.material]:[];for(const m of materials){for(const value of Object.values(m))if(value?.isTexture)value.dispose();m.dispose();}});}
  };
}
