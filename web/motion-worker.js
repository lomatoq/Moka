import {parseBVH,retarget} from './motion.js';
self.onmessage=({data})=>{try{const result=data.type==='bvh'?parseBVH(data.text,data.name):retarget(data.motion,data.rig,data.mapping,data.options);self.postMessage({result});}catch(error){self.postMessage({error:error.message||String(error)});}};
