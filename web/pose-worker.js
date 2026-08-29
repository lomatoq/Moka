/* MediaPipe runs off the UI thread. Only model/library URLs are fetched. */
let detector=null,mode=null,settings=null;
self.onmessage=async({data})=>{
  const {id,type,bitmap}=data;
  try{
    if(type==='init'){
      settings=data.settings;
      const {PoseLandmarker,FilesetResolver}=await import(settings.pose_module);
      const files=await FilesetResolver.forVisionTasks(settings.pose_wasm);
      detector=await PoseLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:settings.pose_model,delegate:'CPU'},runningMode:'IMAGE',numPoses:1,minPoseDetectionConfidence:.4,minPosePresenceConfidence:.4,minTrackingConfidence:.4});
      mode='IMAGE';self.postMessage({id,result:{ready:true}});return;
    }
    if(!detector)throw new Error('Pose detector has not been initialized');
    const desired=type==='video'?'VIDEO':'IMAGE';if(mode!==desired){await detector.setOptions({runningMode:desired});mode=desired;}
    const result=desired==='VIDEO'?detector.detectForVideo(bitmap,data.timestamp):detector.detect(bitmap);
    self.postMessage({id,result:{landmarks:result.landmarks,worldLandmarks:result.worldLandmarks}});
  }catch(error){self.postMessage({id,error:error?.message||String(error)});}finally{bitmap?.close();}
};
