import {Stage} from './stage.js';
import {clone,demoClip,interpolateFrame,setupTransforms,DEG,clamp,uid} from './core.js';
import {autoMap,validateMotion,landmarksToFrame,clipToBVH,canonicalFrame} from './motion.js';
import {PoseCapture} from './capture.js';

const $=id=>document.getElementById(id),$$=q=>[...document.querySelectorAll(q)];
const state={project:null,capabilities:null,clipId:'',time:0,playing:false,pose:{angles:{},root:[0,0]},selectedBone:'torso',selectedLayer:null,motion:null,mapping:{},busy:false,job:null,customStart:null,dirty:false,poseDirty:false};
const source=new Stage($('source-stage'),{source:true}),preview=new Stage($('preview-stage'));
let capture=null,captureController=null,captureURL=null,toastTimer=null,lastTick=0,saveChain=Promise.resolve();
const engineDescriptions={cpu:'Geometry-guided cut. Extends texture under joints; does not invent hidden anatomy.',imported:'Preserves imported PSD artwork, then splits coarse semantic layers at the rig joints.',sam2:'Prompted SAM 2 masks + rig-conditioned anatomical split. Hidden joint overlap uses texture extension.',seethrough:'Local See-through generates inpainted semantic layers; Moka splits them into actual articulated parts.',qwen:'Local Qwen Image Layered generates RGBA layers. Moka adds anatomical subdivision. Requires a compatible CUDA setup.'};
const on=(id,event,fn)=>$(id).addEventListener(event,guard(fn));
function guard(fn){return async event=>{try{await fn(event);}catch(error){report(error);}};}
function report(error){console.error(error);toast(error.message||String(error),true);status(error.message||'Operation failed');}
function toast(message,error=false){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,error?14000:5000);}
function status(message){$('status-message').textContent=message;}
async function api(path,options={}){const response=await fetch(path,{...options,headers:options.body instanceof FormData?{}:{'Content-Type':'application/json',...options.headers}});if(!response.ok){let reason;try{reason=(await response.json()).detail;}catch{reason=response.statusText;}throw new Error(typeof reason==='string'?reason:JSON.stringify(reason));}return response.json();}
const asset=path=>`/api/projects/${state.project.id}/assets/${path.split('/').map(encodeURIComponent).join('/')}?r=${state.project.revision}`;
const activeClip=()=>state.project?.clips.find(c=>c.id===state.clipId)||null;
const duration=()=>activeClip()?.duration||4;
function busy(value,message=''){state.busy=value;document.body.classList.toggle('busy',value);source.disabled=preview.disabled=value;refreshButtons();if(message)status(message);}
function refreshButtons(){const p=state.project,ready=!!p&&!state.busy;
  for(const id of ['save-button','export-button','apply-preset','detect-pose','cut-button','import-motion','add-key','reset-pose','rebuild-mesh'])$(id).disabled=!ready;
  $('project-name').disabled=!ready;$('background-button').disabled=!ready||!!p?.layers.length||!state.capabilities?.engines.rembg.available;
  $('play-button').disabled=!ready||!activeClip();$('reset-time').disabled=!ready;$('retarget-button').disabled=!ready||!state.motion;
  $$('[data-motion]').forEach(b=>b.disabled=!ready);for(const id of ['demo-button','import-character','welcome-demo'])$(id).disabled=state.busy;
}
function markDirty(){state.dirty=true;$('save-state').textContent='Unsaved edits';}
function save(patch={}){
  if(!state.project)return Promise.resolve();
  saveChain=saveChain.catch(()=>{}).then(async()=>{
    $('save-state').textContent='Saving…';
    const p=state.project;const data={revision:p.revision,rig:clone(p.rig),clips:clone(p.clips),name:$('project-name').value,...patch};
    const result=await api(`/api/projects/${p.id}`,{method:'PUT',body:JSON.stringify(data)});
    await adopt(result,{fit:false,preserveTime:true});state.dirty=false;$('save-state').textContent='Saved locally';return result;
  });return saveChain;
}
async function adopt(project,{fit=false,preserveTime=false}={}){
  state.project=project;state.motion=project.source_motion||state.motion;
  if(!project.clips.some(c=>c.id===state.clipId))state.clipId='';
  if(!preserveTime){state.time=0;state.playing=false;state.pose={angles:{},root:[0,0]};}
  if(!project.rig.bones.some(b=>b.id===state.selectedBone))state.selectedBone='torso';
  if(!project.layers.some(l=>l.id===state.selectedLayer))state.selectedLayer=project.layers[0]?.id||null;
  $('welcome').hidden=true;$('project-name').value=project.name;$('save-state').textContent='Saved locally';$('preset').value=project.rig.preset;
  $('source-size').textContent=`${project.width} × ${project.height}`;$('part-count').textContent=project.layers.length?`${project.layers.length} PARTS`:'SETUP ONLY';
  try{localStorage.setItem('moka:lastProject',project.id);}catch{}
  await Promise.all([source.setProject(project,asset,{fit}),preview.setProject(project,asset,{fit})]);
  renderLayers();renderRig();renderClips();renderQuality();refreshButtons();setSelection();updateTimeline();
}
function tab(name){$$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));for(const n of ['layers','rig','motion'])$(n+'-panel').hidden=n!==name;}
function setTool(tool){source.tool=tool;preview.tool=tool==='pan'?'pan':'rig';state.customStart=null;$$('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));if(['brush','erase'].includes(tool)){source.showFill=true;$('show-fill').checked=true;tab('layers');}if(tool==='bone'){tab('rig');status('Custom bone: click the start, then the tip in the source view.');}source.render();}
function setSelection(){source.selectedBone=preview.selectedBone=state.selectedBone;source.selectedLayer=preview.selectedLayer=state.selectedLayer;
  const layer=state.project?.layers.find(l=>l.id===state.selectedLayer);$('layer-properties').hidden=!layer;
  if(layer){$('layer-name').value=layer.name;$('layer-bone').value=layer.bone;$('layer-order').value=layer.order;$('layer-opacity').value=layer.opacity;
    $('layer-quality').textContent=`${layer.mesh.vertices.length} vertices · ${layer.mesh.triangles.length/3} triangles\n${((layer.fill_fraction||0)*100).toFixed(1)}% completed overlap · ${layer.completion||layer.provenance}`;$('layer-quality').classList.toggle('warning',!!layer.needs_review);}
  $('bone-select').value=state.selectedBone;const frame=!state.poseDirty&&activeClip()?interpolateFrame(activeClip(),state.time):state.pose;const angle=frame.angles?.[state.selectedBone]||0;$('bone-angle').value=clamp(angle,-180,180);$('bone-angle-value').textContent=`${angle.toFixed(1)}°`;
  const b=state.project?.rig.bones.find(b=>b.id===state.selectedBone);if(b){const t=setupTransforms(state.project.rig)[b.id];$('bone-info').textContent=`${b.id} → ${b.parent||'world'}\nLength: ${t.length.toFixed(1)} px · ${b.layer?'artwork part':'connector'}\n${b.start} → ${b.end}`;}
  $('track-name').textContent=state.selectedBone||'Bone rotation';$$('.layer-row').forEach(e=>e.classList.toggle('active',e.dataset.id===state.selectedLayer));source.render();preview.render(frame);drawCurve();
}
function renderLayers(){const list=$('layer-list');list.replaceChildren();const layers=state.project.layers;$('layers-count').textContent=layers.length;
  if(!layers.length){const e=document.createElement('div');e.className='empty-state';e.textContent='Align the skeleton, then separate the artwork.';list.append(e);return;}
  for(const layer of [...layers].sort((a,b)=>b.order-a.order)){
    const row=document.createElement('div');row.className='layer-row';row.dataset.id=layer.id;row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label',`Select ${layer.name}`);
    const img=document.createElement('img');img.src=asset(layer.image);img.alt='';const text=document.createElement('span');text.className='layer-label';const strong=document.createElement('strong');strong.textContent=layer.name;const small=document.createElement('small');small.textContent=layer.bone;text.append(strong,small);row.append(img,text);
    if(layer.needs_review){const dot=document.createElement('span');dot.className='review-dot';dot.textContent='●';dot.title='Inspect the completed area';row.append(dot);}
    const eye=document.createElement('button');eye.className='eye';eye.textContent=layer.visible?'◉':'○';eye.title=layer.visible?'Hide part':'Show part';eye.addEventListener('click',guard(async e=>{e.stopPropagation();if(state.busy)return;await save({layer_edits:[{id:layer.id,visible:!layer.visible}]});}));row.append(eye);
    const select=()=>{state.selectedLayer=layer.id;state.selectedBone=layer.bone;setSelection();};row.addEventListener('click',select);row.addEventListener('keydown',e=>{if(e.key==='Enter')select();});list.append(row);
  }
}
function renderRig(){const options=state.project.rig.bones.map(b=>({value:b.id,text:b.id}));for(const id of ['bone-select','layer-bone'])fillSelect($(id),options);$('bones-count').textContent=options.length;}
function fillSelect(element,options,selected){element.replaceChildren(...options.map(o=>{const e=document.createElement('option');e.value=o.value;e.textContent=o.text;return e;}));if(selected!==undefined)element.value=selected;}
function renderClips(){const clips=state.project.clips;fillSelect($('clip-select'),[{value:'',text:'Setup / manual pose'},...clips.map(c=>({value:c.id,text:c.name}))],state.clipId);const list=$('clip-list');list.replaceChildren();
  for(const c of clips){const button=document.createElement('button');button.textContent=c.name;button.classList.toggle('active',c.id===state.clipId);button.addEventListener('click',()=>selectClip(c.id));list.append(button);}
  if(!clips.length){const text=document.createElement('span');text.className='muted';text.textContent='No clips yet';list.append(text);}
}
function selectClip(id){state.poseDirty=false;state.clipId=id;state.time=0;state.playing=false;state.pose={angles:{},root:[0,0]};renderClips();refreshButtons();updateTimeline();setSelection();}
function renderQuality(){const q=state.project.quality||{},warnings=state.project.warnings||[];const lines=[];
  if(typeof q.visible_coverage==='number')lines.push(`Visible pixels retained: ${(q.visible_coverage*100).toFixed(2)}%`);
  if(q.produced_parts!==undefined)lines.push(`Parts: ${q.produced_parts} / ${q.expected_parts}`);
  if(q.ambiguous_geometry_fraction!==undefined)lines.push(`Geometrically ambiguous: ${(q.ambiguous_geometry_fraction*100).toFixed(1)}%`);
  if(q.method)lines.push(`Method: ${q.method}`);lines.push(...warnings.map(w=>'• '+w));$('quality-content').textContent=lines.join('\n\n')||'No decomposition yet.';
}
function updateTimeline(){const d=duration();state.time=clamp(state.time,0,d);$('scrubber').max=d;$('scrubber').value=state.time;$('time-label').replaceChildren(document.createTextNode(state.time.toFixed(2)+' '));const small=document.createElement('small');small.textContent=`/ ${d.toFixed(2)} s`;$('time-label').append(small);$('play-button').textContent=state.playing?'Ⅱ':'▶';
  const frame=activeClip()?interpolateFrame(activeClip(),state.time):state.pose;preview.render(frame);drawCurve();
  if(state.motion&&state.project){const i=Math.min(state.motion.frames.length-1,Math.round(state.time*(state.motion.fps||30))),raw=state.motion.frames[i];const frame={time:raw.time,joints:canonicalFrame(raw,state.mapping)};const links=state.project.rig.bones.map(b=>[b.start,b.end]);source.sourceMotion={frame,links};source.render();}
}
function drawCurve(){const canvas=$('curve-canvas'),box=canvas.getBoundingClientRect();if(!box.width)return;const dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(box.width*dpr)||canvas.height!==Math.round(box.height*dpr)){canvas.width=box.width*dpr;canvas.height=box.height*dpr;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,box.width,box.height);const d=duration(),w=box.width,h=box.height;
  ctx.font='8px system-ui';for(let i=0;i<=8;i++){const x=i/8*w;ctx.strokeStyle='#293b44';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,17);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle='#708d9b';ctx.fillText((i/8*d).toFixed(1),x+3,11);}
  const clip=activeClip();if(clip?.frames.length){const values=clip.frames.map(f=>f.angles[state.selectedBone]||0),max=Math.max(10,...values.map(Math.abs));ctx.strokeStyle='#a8dfbf';ctx.lineWidth=1.5;ctx.beginPath();clip.frames.forEach((f,i)=>{const x=f.time/d*w,y=45-values[i]/max*21;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  const x=state.time/d*w;ctx.strokeStyle='#f1c785';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle='#f1c785';ctx.beginPath();ctx.moveTo(x-3,0);ctx.lineTo(x+3,0);ctx.lineTo(x,5);ctx.fill();
}
function tick(timestamp){const dt=lastTick?Math.min(.1,(timestamp-lastTick)/1000):0;lastTick=timestamp;if(state.playing&&activeClip()){state.time+=dt*Number($('speed').value);if(state.time>duration()){if($('play-loop').checked)state.time%=duration();else{state.time=duration();state.playing=false;}}updateTimeline();}requestAnimationFrame(tick);}
async function importCharacter(file){if(!file)return;busy(true,'Importing character…');try{const form=new FormData();form.append('file',file);form.append('preset',$('preset').value);const project=await api('/api/projects',{method:'POST',body:form});state.motion=null;state.mapping={};state.clipId='';await adopt(project,{fit:true});if(project.suggested_engine)$('cut-engine').value=project.suggested_engine;updateEngineHelp();status('Character imported. Align or detect the setup joints before cutting.');}finally{busy(false);}}
async function demo(){busy(true,'Loading the synthetic demo…');try{state.motion=null;state.mapping={};state.clipId='';const p=await api('/api/demo',{method:'POST'});await adopt(p,{fit:true});$('cut-engine').value='cpu';updateEngineHelp();status('Demo with known joints loaded. Click Separate & build rig.');}finally{busy(false);}}
function updateEngineHelp(){$('engine-help').textContent=engineDescriptions[$('cut-engine').value];}
async function cut(){await save();busy(true,'Separating artwork…');state.playing=false;$('job-bar').hidden=false;try{
  const job=await api(`/api/projects/${state.project.id}/cut`,{method:'POST',body:JSON.stringify({revision:state.project.revision,engine:$('cut-engine').value,padding:Number($('padding').value),rigid:$('rigid').checked})});state.job=job.id;
  while(true){const j=await api('/api/jobs/'+job.id);$('job-message').textContent=j.message;$('job-progress').value=j.progress;
    if(j.status==='done'){await adopt(await api('/api/projects/'+state.project.id),{preserveTime:true});tab('layers');toast(`${state.project.layers.length} parts created. Inspect overlaps with the Fill overlay.`);status('Layers built. Visible pixels and completed overlap have separate masks.');break;}
    if(j.status==='failed')throw new Error(j.error||j.message);if(j.status==='cancelled'){toast(j.message);break;}await new Promise(r=>setTimeout(r,500));
  }
}finally{state.job=null;$('job-bar').hidden=true;busy(false);}}
function workerTask(data){return new Promise((resolve,reject)=>{const worker=new Worker('/static/motion-worker.js',{type:'module'});worker.onmessage=({data})=>{worker.terminate();data.error?reject(new Error(data.error)):resolve(data.result);};worker.onerror=e=>{worker.terminate();reject(new Error(e.message));};worker.postMessage(data);});}
async function importMotion(file){if(!file||!state.project)return;if(file.size>128*1024*1024)throw new Error('Motion files must be below 128 MB');const ext=file.name.split('.').at(-1).toLowerCase();
  if(file.type.startsWith('video/')||['mp4','webm','mov','m4v'].includes(ext)){openCapture(file);return;}
  busy(true,'Reading source motion…');try{let motion;
    if(ext==='bvh')motion=await workerTask({type:'bvh',text:await file.text(),name:file.name});
    else if(ext==='json')motion=validateMotion(JSON.parse(await file.text()));
    else if(['fbx','glb','gltf'].includes(ext)){
      const {load3D}=await import('./import3d.js');const imported=await load3D(file);try{const index=imported.names.length>1?await chooseAnimation(imported.names):0;if(index===null)return;motion=await imported.sample(index,{onProgress:(_,text)=>status(text)});}finally{imported.dispose();}
    }else throw new Error('Use BVH, FBX, GLB, motion JSON, MP4, or WebM.');
    await installMotion(motion);
  }finally{busy(false);}}
async function installMotion(motion){state.motion=validateMotion(motion);state.mapping=autoMap(motion.joints.map(j=>typeof j==='string'?j:j.name),state.project.rig.joints.map(j=>j.id));await save({source_motion:motion});renderMapping();tab('motion');refreshButtons();status('Source motion loaded. Review the mapping and apply it to the character.');toast('Motion loaded. The source skeleton has not changed your character proportions.');}
function renderMapping(){if(!state.motion||!state.project)return;const motion=state.motion,names=motion.joints.map(j=>typeof j==='string'?j:j.name);const list=$('mapping-list');list.replaceChildren();
  for(const joint of state.project.rig.joints){const row=document.createElement('div');row.className='mapping-row';const label=document.createElement('label');label.textContent=joint.id;const select=document.createElement('select');select.setAttribute('aria-label',`Source for ${joint.id}`);fillSelect(select,[{value:'',text:'Derived / unmapped'},...names.map(n=>({value:n,text:n}))],state.mapping[joint.id]||'');select.addEventListener('change',()=>{state.mapping[joint.id]=select.value;});row.append(label,select);list.append(row);}
  $('mapping-count').textContent=`${Object.values(state.mapping).filter(Boolean).length}/${state.project.rig.joints.length}`;$('reference-frame').max=motion.frames.length-1;
  const d=motion.frames.at(-1).time;$('motion-summary').textContent=`${motion.name}\n${motion.frames.length} frames · ${d.toFixed(2)} s · ${motion.type}\n${motion.diagnostics?.note||'Source skeleton evaluated before 2D retargeting.'}`;
}
async function applyMotion(){if(!state.motion)return;busy(true,'Retargeting world-space motion…');try{
  const options={projection:$('projection').value,yaw:Number($('yaw').value),mode:$('retarget-mode').value,reference:Number($('reference-frame').value),smoothing:Number($('smoothing').value)/100,mirror:$('mirror-motion').checked,rootMotion:$('root-motion').checked,footLock:$('foot-lock').checked,loop:$('close-loop').checked};
  const clip=await workerTask({type:'retarget',motion:state.motion,rig:state.project.rig,mapping:state.mapping,options});state.project.clips.push(clip);state.clipId=clip.id;state.motion.target_mapping={...state.mapping};await save({source_motion:state.motion});selectClip(clip.id);
  const d=clip.diagnostics;$('motion-diagnostics').textContent=`${d.sourceFrames} frames retargeted. Target bone lengths preserved.\nLow-confidence samples filled/held: ${d.filledLowConfidenceSamples}.\nUnmapped roles: ${d.missingRoles.join(', ')||'none'}.\nFoot-contact frames: ${d.contactFrames}.`;toast('Motion applied. Play it, inspect crossings, then export.');status('Retarget complete · target proportions preserved');
}finally{busy(false);}}
async function detectPose(){if(state.project.rig.preset==='quadruped')throw new Error('The configured pose detectors are trained for humans. Adjust the quadruped scaffold or use a manually prepared skeleton.');busy(true,'Detecting character joints…');try{
  if($('pose-engine').value==='dwpose'){await adopt(await api(`/api/projects/${state.project.id}/pose`,{method:'POST'}),{preserveTime:true});}
  else{capture??=new PoseCapture(state.capabilities);const result=await capture.image(source.original),points=result.landmarks?.[0];if(!points||points.length!==33)throw new Error('No human skeleton was detected. Existing joints were preserved.');const torso=[11,12,23,24].map(i=>points[i].visibility??0);if(torso.reduce((a,b)=>a+b,0)/4<.4)throw new Error('Detection confidence is too low. Existing joints were preserved.');const frame=landmarksToFrame(points,0,state.project.width,state.project.height),rig=clone(state.project.rig);for(const j of rig.joints){const p=frame.joints[j.id];if(p&&p[3]>.25){j.x=p[0];j.y=-p[1];j.confidence=p[3];j.source='mediapipe';}}rig.provenance='mediapipe-with-derived-endpoints';rig.confidence=torso.reduce((a,b)=>a+b,0)/4;state.project.rig=rig;await save({rebuild_mesh:true});}
  tab('rig');toast('Detected joints applied. Review hands, feet, and the head tip before cutting.');
}finally{busy(false);}}
function openCapture(file){captureController?.abort();capture?.stop();capture=new PoseCapture(state.capabilities);if(captureURL)URL.revokeObjectURL(captureURL);captureURL=URL.createObjectURL(file);$('capture-video').src=captureURL;$('capture-title').textContent=file.name;$('capture-status').textContent='Choose the sample rate and processed duration. Full input remains local.';$('capture-progress').value=0;$('capture-start').disabled=false;$('capture-dialog').showModal();}
async function startCapture(){const video=$('capture-video');if(!video.videoWidth)throw new Error('The browser has not decoded this video. Try MP4/H.264 or WebM.');captureController=new AbortController();$('capture-start').disabled=true;busy(true,'Capturing video skeleton locally…');try{
  const motion=await capture.video(video,{fps:Number($('capture-fps').value),maxDuration:clamp(Number($('capture-duration').value),1,180),signal:captureController.signal,onProgress:(value,message)=>{$('capture-progress').value=value;$('capture-status').textContent=message;}});motion.name=$('capture-title').textContent.replace(/\.[^.]+$/,'');$('capture-dialog').close();await installMotion(motion);
}finally{$('capture-start').disabled=false;busy(false);}}
function closeCapture(){captureController?.abort();capture?.stop();$('capture-video').pause();$('capture-dialog').close();}
function chooseAnimation(names){return new Promise(resolve=>{const dialog=$('animation-dialog');$('animation-options').replaceChildren();let done=false;const finish=value=>{if(done)return;done=true;dialog.close();resolve(value);};for(const a of names){const b=document.createElement('button');b.textContent=`${a.name} · ${a.duration.toFixed(2)} s`;b.onclick=()=>finish(a.index);$('animation-options').append(b);}$('animation-close').onclick=()=>finish(null);dialog.addEventListener('cancel',()=>finish(null),{once:true});dialog.showModal();});}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);}
async function exportFile(kind){await save();if(kind==='bvh'){const clip=activeClip();if(!clip)throw new Error('Select an animation clip before exporting BVH.');download(new Blob([clipToBVH(state.project.rig,clip)],{type:'text/plain'}),'moka-planar-motion.bvh');return;}
  const response=await fetch(`/api/projects/${state.project.id}/export/${kind}`);if(!response.ok){const e=await response.json();throw new Error(e.detail||'Export failed');}const filename=response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1]||`moka-${kind}.zip`;download(await response.blob(),filename);toast(`${filename} exported.`);
}
async function addKey(){const frame=clone(state.poseDirty?state.pose:(activeClip()?interpolateFrame(activeClip(),state.time):state.pose)),existing=activeClip();let clip=existing;
  if(!clip){clip={id:uid(),name:'Manual pose keys',fps:30,duration:4,source_type:'manual',frames:[{time:0,angles:{},root:[0,0]},{time:4,angles:{},root:[0,0]}]};state.project.clips.push(clip);state.clipId=clip.id;}
  const key={time:state.time,angles:frame.angles||{},root:frame.root||[0,0]};const i=clip.frames.findIndex(f=>Math.abs(f.time-key.time)<.001);if(i>=0)clip.frames[i]=key;else clip.frames.push(key);clip.frames.sort((a,b)=>a.time-b.time);clip.duration=Math.max(clip.duration,state.time);await save();state.poseDirty=false;updateTimeline();toast(`Pose key saved at ${state.time.toFixed(2)} s.`);
}
source.onRigChange=final=>{markDirty();preview.project=state.project;preview.render();if(final)save({rebuild_mesh:true}).catch(report);};
source.onJointSelect=id=>{$('joint-info').textContent=`Selected joint: ${id}. Drag in the source view. Re-run separation when the anatomical boundaries need to change.`;};
source.onBoneSelect=preview.onBoneSelect=id=>{state.selectedBone=id;const l=state.project.layers.find(l=>l.bone===id);if(l)state.selectedLayer=l.id;setSelection();};
source.onStroke=guard(async stroke=>{if(!state.selectedLayer)return;busy(true,'Saving visible-mask edit…');try{await adopt(await api(`/api/projects/${state.project.id}/mask`,{method:'POST',body:JSON.stringify({revision:state.project.revision,layer:state.selectedLayer,strokes:[{...stroke,erase:stroke.mode==='erase'}]})}),{preserveTime:true});}finally{busy(false);}});
source.onPick=guard(async point=>{if(!state.customStart){state.customStart=point;status('Custom bone: now click its tip.');return;}const id=$('custom-bone-name').value.trim();if(!/^[A-Za-z][A-Za-z0-9_]{0,39}$/.test(id))throw new Error('Use a short bone name with letters, digits, and underscores.');if(state.project.rig.bones.some(b=>b.id===id))throw new Error('This bone name already exists.');const start=state.customStart;state.customStart=null;const rig=state.project.rig;rig.joints.push({id:id+'_start',x:start[0],y:start[1],source:'manual',confidence:1},{id:id+'_end',x:point[0],y:point[1],source:'manual',confidence:1});rig.bones.push({id,parent:state.selectedBone||'root',start:id+'_start',end:id+'_end',layer:$('custom-layer').checked});await save();state.selectedBone=id;setTool('rig');setSelection();toast('Custom bone added. Separate again to allocate artwork to it.');});

on('demo-button','click',demo);on('welcome-demo','click',demo);on('import-character','click',()=>$('character-file').click());on('character-file','change',async e=>{await importCharacter(e.target.files[0]);e.target.value='';});on('save-button','click',()=>save());on('project-name','change',()=>save());
on('apply-preset','click',async()=>{if(state.project.layers.length&&!confirm('Replace setup joints? Existing artwork is retained, but meshes will be rebound and motion should be reviewed.'))return;const p=await api(`/api/projects/${state.project.id}/preset`,{method:'POST',body:JSON.stringify({revision:state.project.revision,preset:$('preset').value})});await adopt(p,{preserveTime:true});toast('Scaffold applied. This is not detected anatomy.');});
on('background-button','click',async()=>{busy(true,'Removing the background locally…');try{await adopt(await api(`/api/projects/${state.project.id}/background`,{method:'POST'}),{fit:true});}finally{busy(false);}});
on('detect-pose','click',detectPose);on('cut-button','click',cut);on('cut-engine','change',updateEngineHelp);on('padding','input',()=>$('padding-value').textContent=$('padding').value+' px');on('cancel-job','click',()=>state.job&&api('/api/jobs/'+state.job+'/cancel',{method:'POST'}));
$$('[data-tab]').forEach(b=>b.addEventListener('click',()=>tab(b.dataset.tab)));$$('[data-tool]').forEach(b=>b.addEventListener('click',()=>setTool(b.dataset.tool)));
on('fit-button','click',()=>{source.fit();preview.fit();});on('show-bones','change',()=>{source.showBones=preview.showBones=$('show-bones').checked;source.render();preview.render();});on('show-mesh','change',()=>{preview.showMesh=$('show-mesh').checked;preview.render();});on('show-fill','change',()=>{source.showFill=$('show-fill').checked;source.render();});
on('brush-size','input',()=>{source.brushRadius=Number($('brush-size').value);$('brush-size-value').textContent=source.brushRadius+' px';});
for(const id of ['layer-name','layer-bone','layer-order','layer-opacity'])on(id,'change',()=>{if(!state.selectedLayer)return;const edit={id:state.selectedLayer,name:$('layer-name').value,bone:$('layer-bone').value,order:Number($('layer-order').value),opacity:Number($('layer-opacity').value)};return save({layer_edits:[edit]});});
on('bone-select','change',()=>{state.selectedBone=$('bone-select').value;setSelection();});on('bone-angle','input',()=>{state.playing=false;if(activeClip()&&!state.poseDirty)state.pose=clone(interpolateFrame(activeClip(),state.time));state.poseDirty=true;state.pose.angles[state.selectedBone]=Number($('bone-angle').value);$('bone-angle-value').textContent=$('bone-angle').value+'°';preview.render(state.pose);$('play-button').textContent='▶';});
on('reset-pose','click',()=>{selectClip('');state.pose={angles:{},root:[0,0]};setSelection();});on('rebuild-mesh','click',()=>save({rebuild_mesh:true,rigid:$('rigid').checked}));on('add-key','click',addKey);
$$('[data-motion]').forEach(b=>b.addEventListener('click',guard(async()=>{const clip=demoClip(state.project.rig,b.dataset.motion);state.project.clips.push(clip);state.clipId=clip.id;await save();selectClip(clip.id);state.playing=true;updateTimeline();})));
on('clip-select','change',()=>selectClip($('clip-select').value));on('play-button','click',()=>{state.playing=!state.playing;updateTimeline();});on('reset-time','click',()=>{state.time=0;updateTimeline();});on('scrubber','input',()=>{state.time=Number($('scrubber').value);state.playing=false;state.poseDirty=false;updateTimeline();});
on('import-motion','click',()=>$('motion-file').click());on('motion-file','change',async e=>{await importMotion(e.target.files[0]);e.target.value='';});on('retarget-button','click',applyMotion);on('yaw','input',()=>$('yaw-value').textContent=$('yaw').value+'°');on('smoothing','input',()=>$('smoothing-value').textContent=$('smoothing').value+'%');
on('capture-start','click',startCapture);on('capture-close','click',closeCapture);$('capture-dialog').addEventListener('cancel',closeCapture);
on('export-button','click',()=>$('export-dialog').showModal());$$('[data-export]').forEach(b=>b.addEventListener('click',guard(()=>exportFile(b.dataset.export))));
on('projects-button','click',async()=>{const projects=await api('/api/projects');$('projects-list').replaceChildren();for(const p of projects){const button=document.createElement('button');const name=document.createElement('span');name.textContent=p.name;const size=document.createElement('small');size.textContent=`${p.width} × ${p.height}`;button.append(name,size);button.addEventListener('click',guard(async()=>{state.motion=null;state.clipId='';await adopt(await api('/api/projects/'+p.id),{fit:true});if(state.motion){state.mapping=state.motion.target_mapping||autoMap(state.motion.joints.map(j=>typeof j==='string'?j:j.name));renderMapping();}$('projects-dialog').close();}));$('projects-list').append(button);}if(!projects.length)$('projects-list').textContent='No saved projects yet.';$('projects-dialog').showModal();});
on('engines-button','click',()=>{const list=$('engines-list');list.replaceChildren();for(const e of Object.values(state.capabilities.engines)){const row=document.createElement('div');row.className='engine-row';const title=document.createElement('strong');title.textContent=e.label;row.append(title,document.createTextNode(`${e.available?'Available / configured':'Not installed'} · ${e.kind}${e.tested_inference?'':' · inference not verified in this build'}`));list.append(row);}const note=document.createElement('p');note.className='field-help';note.textContent=state.capabilities.browser_assets_cached?'Browser pose model and 3D loaders are cached locally.':'First browser pose/3D use downloads official library/model assets. Use tools/cache_assets.py for offline operation.';list.append(note);$('engines-dialog').showModal();});
window.addEventListener('keydown',guard(async e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.code==='Space'){e.preventDefault();if(activeClip()){state.playing=!state.playing;updateTimeline();}}if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();await save();}if(e.key==='v')setTool('rig');if(e.key==='Escape')setTool('rig');}));
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
let dragDepth=0;window.addEventListener('dragenter',e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();dragDepth++;document.body.classList.add('drag-over');}});window.addEventListener('dragleave',()=>{dragDepth--;if(dragDepth<=0)document.body.classList.remove('drag-over');});window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',guard(async e=>{e.preventDefault();dragDepth=0;document.body.classList.remove('drag-over');if(state.busy)return;const file=e.dataTransfer.files[0];if(!file)return;const ext=file.name.split('.').at(-1).toLowerCase();if(['bvh','fbx','glb','gltf','json','mp4','mov','webm'].includes(ext)){if(!state.project)throw new Error('Import a character before loading its motion.');await importMotion(file);}else await importCharacter(file);}));
new ResizeObserver(drawCurve).observe($('curve-canvas'));requestAnimationFrame(tick);
async function init(){state.capabilities=await api('/api/capabilities');for(const option of $('cut-engine').options){const e=state.capabilities.engines[option.value];if(e&&!e.available){option.disabled=true;option.textContent+=' · not installed';}}$('pose-engine').options[1].disabled=!state.capabilities.engines.dwpose.available;$('render-status').textContent=preview.gl?'WEBGL 2 · WEIGHTED MESHES':'CANVAS · SOFTWARE MESHES';let id;try{id=localStorage.getItem('moka:lastProject');}catch{}if(id){try{await adopt(await api('/api/projects/'+id),{fit:true});if(state.motion){state.mapping=state.motion.target_mapping||autoMap(state.motion.joints.map(j=>typeof j==='string'?j:j.name),state.project.rig.joints.map(j=>j.id));renderMapping();}}catch(e){status('Previous project unavailable. Import a character or try the demo.');}}refreshButtons();}
init().catch(report);
