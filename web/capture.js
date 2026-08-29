import {landmarksToFrame,validateMotion} from './motion.js';

export class PoseCapture {
  constructor(settings){this.settings=settings;this.pending=new Map();this.counter=0;this.worker=null;this.ready=null;}
  async init(){
    if(this.ready)return this.ready;
    // MediaPipe's pinned WASM bootstrap calls importScripts; use a classic worker.
    // pose-worker.js still loads the public ES module with dynamic import().
    this.worker=new Worker('/static/pose-worker.js');
    this.worker.onmessage=({data})=>{const item=this.pending.get(data.id);if(!item)return;this.pending.delete(data.id);clearTimeout(item.timer);data.error?item.reject(new Error(data.error)):item.resolve(data.result);};
    this.worker.onerror=e=>{for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(new Error(e.message||'Pose worker could not start'));}this.pending.clear();this.ready=null;};
    this.ready=this.request('init',{settings:this.settings},[],180000).catch(e=>{this.ready=null;throw e;});return this.ready;
  }
  request(type,extra={},transfer=[],timeout=45000){
    return new Promise((resolve,reject)=>{const id=++this.counter,timer=setTimeout(()=>{this.pending.delete(id);reject(new Error('Pose inference timed out. Check model downloads, then retry.'));},timeout);this.pending.set(id,{resolve,reject,timer});this.worker.postMessage({id,type,...extra},transfer);});
  }
  async image(image){await this.init();const bitmap=await createImageBitmap(image);return this.request('image',{bitmap},[bitmap]);}
  stop(){for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(new Error('Capture cancelled'));}this.pending.clear();this.worker?.terminate();this.worker=null;this.ready=null;}
  async video(video,{fps=24,maxDuration=30,onProgress=()=>{},signal}={}){
    await this.init();video.pause();
    const duration=Math.min(video.duration,maxDuration);if(!Number.isFinite(duration)||duration<=0)throw new Error('Video duration is unavailable');
    if(duration>180||fps<1||fps>30)throw new Error('Capture limit: 180 seconds, up to 30 fps');
    const count=Math.max(2,Math.floor(duration*fps)),frames=[],screen_frames=[];let found=0,missing=0;
    const timestampBase=performance.now()+1000;
    for(let i=0;i<count;i++){
      if(signal?.aborted)throw new Error('Capture cancelled');const time=i/fps;
      await seek(video,Math.min(time,video.duration-.002));
      const bitmap=await createImageBitmap(video);
      const result=await this.request('video',{bitmap,timestamp:timestampBase+i*1000/fps},[bitmap]);
      const imagePose=result.landmarks?.[0],worldPose=result.worldLandmarks?.[0];
      if(imagePose?.length===33 && worldPose?.length===33){frames.push(landmarksToFrame(worldPose,time,1,1,true));screen_frames.push(landmarksToFrame(imagePose,time,video.videoWidth,video.videoHeight));found++;}
      else {frames.push({time,joints:{}});screen_frames.push({time,joints:{}});missing++;}
      onProgress((i+1)/count,`Tracked ${i+1}/${count} frames · ${missing} without a detected body`);
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    if(found<Math.min(3,count))throw new Error('No reliable body track was found. Use a single, fully visible performer. No placeholder motion was generated.');
    const first=frames.find(f=>Object.keys(f.joints).length),names=Object.keys(first.joints);
    return validateMotion({schema:'moka.motion/1',name:'Video performance',type:'mediapipe-video',coordinate_space:'hip-centered-estimated-3d',fps,joints:names.map(name=>({name,parent:null})),frames,screen_frames,
      diagnostics:{detectedFrames:found,missingFrames:missing,totalFrames:count,processedDuration:duration,sourceDuration:video.duration,truncated:duration<video.duration-.1,
        note:'World landmarks are monocular hip-centered estimates, not global root motion. Observed 2D uses video pixels and preserves screen-space translation.'}});
  }
}
export function seek(video,time){return new Promise((resolve,reject)=>{
  if(video.readyState>=2 && Math.abs(video.currentTime-time)<.0005){resolve();return;}
  const timer=setTimeout(()=>finish(new Error('Could not decode the requested video frame')),15000);
  const finish=error=>{clearTimeout(timer);video.removeEventListener('seeked',done);video.removeEventListener('error',bad);error?reject(error):resolve();};
  const done=()=>finish(),bad=()=>finish(new Error('The browser cannot decode this video codec'));
  video.addEventListener('seeked',done,{once:true});video.addEventListener('error',bad,{once:true});video.currentTime=time;
});}
