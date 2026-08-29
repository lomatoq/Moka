import {skinMatrices, deformMesh, poseTransforms, setupTransforms, boneEnd, distanceToSegment, clamp} from './core.js';

const loadImage = async url => {const response=await fetch(url);if(!response.ok)throw new Error('Could not load image asset');const blob=URL.createObjectURL(await response.blob());return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{URL.revokeObjectURL(blob);resolve(image);};image.onerror=()=>{URL.revokeObjectURL(blob);reject(new Error('Could not decode image asset'));};image.src=blob;});};

/** Independent WebGL mesh renderer. No Spine runtime is redistributed. */
export class Stage {
  constructor(container,{source=false}={}) {
    this.container=container;this.source=source;this.scale=1;this.offset=[0,0];this.frame={};this.resources=[];
    this.glCanvas=document.createElement('canvas');this.overlay=document.createElement('canvas');
    this.glCanvas.className='stage-canvas';this.overlay.className='stage-overlay';container.append(this.glCanvas,this.overlay);
    this.ctx=this.overlay.getContext('2d');this.gl=this.glCanvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:true});
    this.showBones=true;this.showMesh=false;this.showFill=false;this.tool='rig';this.selectedBone=null;this.selectedLayer=null;
    this.loadSerial=0;this.pointer=null;this.disabled=false;
    if(this.gl)this.initGL();else {this.glCanvas.remove();this.fallback=true;}
    new ResizeObserver(()=>this.resize()).observe(container);
    this.overlay.addEventListener('pointerdown',e=>this.pointerDown(e));
    this.overlay.addEventListener('pointermove',e=>this.pointerMove(e));
    this.overlay.addEventListener('pointerup',e=>this.pointerUp(e));
    this.overlay.addEventListener('pointercancel',e=>this.pointerUp(e));
    this.overlay.addEventListener('wheel',e=>{e.preventDefault();const p=this.screenPoint(e),factor=Math.exp(-e.deltaY*.001);const old=this.scale;this.scale=clamp(this.scale*factor,.04,10);this.offset=p.map((v,i)=>v-(v-this.offset[i])*this.scale/old);this.render();},{passive:false});
    this.overlay.addEventListener('dblclick',()=>this.fit());
    this.overlay.addEventListener('contextmenu',e=>e.preventDefault());
  }
  initGL(){
    const gl=this.gl;
    const shader=(type,code)=>{const s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    const vs=shader(gl.VERTEX_SHADER,`#version 300 es
      in vec2 a_position;in vec2 a_uv;uniform vec2 u_size;uniform vec2 u_offset;uniform float u_scale;out vec2 v_uv;
      void main(){vec2 p=(a_position*u_scale+u_offset)/u_size*2.0-1.0;gl_Position=vec4(p.x,-p.y,0,1);v_uv=a_uv;}`);
    const fs=shader(gl.FRAGMENT_SHADER,`#version 300 es
      precision mediump float;in vec2 v_uv;uniform sampler2D u_image;uniform float u_opacity;out vec4 outColor;
      void main(){vec4 c=texture(u_image,v_uv);outColor=vec4(c.rgb,c.a*u_opacity);}`);
    this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
    gl.deleteShader(vs);gl.deleteShader(fs);gl.useProgram(this.program);
    this.loc={};for(const key of ['size','offset','scale','image','opacity'])this.loc[key]=gl.getUniformLocation(this.program,'u_'+key);
    this.attrs={pos:gl.getAttribLocation(this.program,'a_position'),uv:gl.getAttribLocation(this.program,'a_uv')};
    gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);
  }
  resource(image,positions,uvs,triangles,layer=null){
    const r={image,positions:new Float32Array(positions),uvs:new Float32Array(uvs),triangles:new Uint32Array(triangles),layer};
    if(this.gl){const gl=this.gl;r.vao=gl.createVertexArray();gl.bindVertexArray(r.vao);
      r.pos=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,r.pos);gl.bufferData(gl.ARRAY_BUFFER,r.positions,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(this.attrs.pos);gl.vertexAttribPointer(this.attrs.pos,2,gl.FLOAT,false,0,0);
      r.uv=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,r.uv);gl.bufferData(gl.ARRAY_BUFFER,r.uvs,gl.STATIC_DRAW);gl.enableVertexAttribArray(this.attrs.uv);gl.vertexAttribPointer(this.attrs.uv,2,gl.FLOAT,false,0,0);
      r.indices=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,r.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,r.triangles,gl.STATIC_DRAW);
      r.texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,r.texture);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    }return r;
  }
  release(){if(!this.gl)return;const gl=this.gl;for(const r of this.resources){gl.deleteBuffer(r.pos);gl.deleteBuffer(r.uv);gl.deleteBuffer(r.indices);gl.deleteTexture(r.texture);gl.deleteVertexArray(r.vao);}this.resources=[];}
  async setProject(project,assetUrl,{fit=false}={}) {
    const serial=++this.loadSerial;this.project=project;this.assetUrl=assetUrl;
    const original=await loadImage(assetUrl(project.source));
    const items=await Promise.all((project.layers||[]).map(async l=>{
      const image=await loadImage(assetUrl(l.image));let mask=null;
      if(l.fill_mask){const raw=await loadImage(assetUrl(l.fill_mask));const c=document.createElement('canvas');c.width=raw.width;c.height=raw.height;const ctx=c.getContext('2d');ctx.drawImage(raw,0,0);const d=ctx.getImageData(0,0,c.width,c.height);for(let i=0;i<d.data.length;i+=4){const a=d.data[i];d.data[i]=255;d.data[i+1]=173;d.data[i+2]=69;d.data[i+3]=a*.7;}ctx.putImageData(d,0,0);mask=c;}
      return {layer:l,image,mask};
    }));
    if(serial!==this.loadSerial)return;
    this.release();this.original=original;this.items=items;
    if(this.source||!items.length)this.resources=[this.resource(original,[0,0,project.width,0,project.width,project.height,0,project.height],[0,0,1,0,1,1,0,1],[0,1,2,0,2,3])];
    else this.resources=items.map(({layer:l,image,mask})=>{const r=this.resource(image,l.mesh.vertices.flat(),l.mesh.uvs.flat(),l.mesh.triangles,l);r.mask=mask;return r;});
    if(fit)this.fit();else this.render();
  }
  resize(){const box=this.container.getBoundingClientRect();this.width=Math.max(1,box.width);this.height=Math.max(1,box.height);this.dpr=Math.min(window.devicePixelRatio||1,2);
    for(const c of [this.glCanvas,this.overlay]){c.width=Math.round(this.width*this.dpr);c.height=Math.round(this.height*this.dpr);}
    if(!this.fitted&&this.project)this.fit();else this.render();
  }
  fit(){if(!this.project)return;this.scale=Math.min(this.width/this.project.width,this.height/this.project.height)*.86;this.offset=[(this.width-this.project.width*this.scale)/2,(this.height-this.project.height*this.scale)/2];this.fitted=true;this.render();}
  screenPoint(e){const r=this.overlay.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
  worldPoint(p){return [(p[0]-this.offset[0])/this.scale,(p[1]-this.offset[1])/this.scale];}
  toScreen(p){return [p[0]*this.scale+this.offset[0],p[1]*this.scale+this.offset[1]];}
  pointerDown(e){if(!this.project||this.disabled)return;this.overlay.setPointerCapture(e.pointerId);const p=this.screenPoint(e),world=this.worldPoint(p);
    if(e.button===1||e.button===2||e.altKey||this.tool==='pan'){this.pointer={type:'pan',start:p,offset:[...this.offset]};return;}
    if(this.source&&['brush','erase'].includes(this.tool)&&this.selectedLayer){this.pointer={type:'brush',points:[world]};this.render();return;}
    if(this.source&&this.tool==='bone'){this.onPick?.(world);return;}
    if(this.source&&this.tool==='rig'){
      const joint=this.project.rig.joints.map(j=>({j,d:Math.hypot(j.x-world[0],j.y-world[1])*this.scale})).sort((a,b)=>a.d-b.d)[0];
      if(joint?.d<12){this.pointer={type:'joint',joint:joint.j.id};this.onJointSelect?.(joint.j.id);return;}
    }
    const transforms=this.source?setupTransforms(this.project.rig):poseTransforms(this.project.rig,this.frame);
    const closest=this.project.rig.bones.filter(b=>b.start!==b.end).map(b=>({b,d:distanceToSegment(world,[transforms[b.id].x,transforms[b.id].y],boneEnd(transforms[b.id]))*this.scale})).sort((a,b)=>a.d-b.d)[0];
    if(closest?.d<15)this.onBoneSelect?.(closest.b.id);
  }
  pointerMove(e){const p=this.screenPoint(e);this.hover=this.worldPoint(p);if(!this.pointer){if(['brush','erase'].includes(this.tool))this.render();return;}
    if(this.pointer.type==='pan')this.offset=p.map((v,i)=>this.pointer.offset[i]+v-this.pointer.start[i]);
    if(this.pointer.type==='joint'){
      const j=this.project.rig.joints.find(j=>j.id===this.pointer.joint),w=this.worldPoint(p);j.x=clamp(w[0],-this.project.width*.5,this.project.width*1.5);j.y=clamp(w[1],-this.project.height*.5,this.project.height*1.5);j.source='manual';j.confidence=1;this.onRigChange?.(false);
    }
    if(this.pointer.type==='brush'){const w=this.worldPoint(p),last=this.pointer.points.at(-1);if(Math.hypot(w[0]-last[0],w[1]-last[1])*this.scale>2)this.pointer.points.push(w);}
    this.render();
  }
  pointerUp(e){if(!this.pointer)return;const action=this.pointer;this.pointer=null;
    if(action.type==='joint')this.onRigChange?.(true);
    if(action.type==='brush')this.onStroke?.({mode:this.tool==='erase'?'erase':'add',radius:this.brushRadius||12,points:action.points});
    if(this.overlay.hasPointerCapture(e.pointerId))this.overlay.releasePointerCapture(e.pointerId);this.render();
  }
  render(frame){if(frame)this.frame=frame;const {gl,ctx}=this;if(!ctx||!this.width)return;
    ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);
    if(gl){gl.viewport(0,0,this.glCanvas.width,this.glCanvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(this.program);gl.uniform2f(this.loc.size,this.width,this.height);gl.uniform2f(this.loc.offset,...this.offset);gl.uniform1f(this.loc.scale,this.scale);gl.uniform1i(this.loc.image,0);}
    if(!this.project)return;
    const matrices=skinMatrices(this.project.rig,this.source?{}:this.frame);
    const resources=[...this.resources].sort((a,b)=>(a.layer?.order||0)-(b.layer?.order||0));
    for(const r of resources){if(r.layer?.visible===false)continue;const opacity=r.layer?.opacity??1;
      r.current=r.layer?deformMesh(r.layer.mesh,matrices):r.positions;
      if(gl){gl.bindVertexArray(r.vao);gl.bindBuffer(gl.ARRAY_BUFFER,r.pos);gl.bufferSubData(gl.ARRAY_BUFFER,0,r.current);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,r.texture);gl.uniform1f(this.loc.opacity,opacity);gl.drawElements(gl.TRIANGLES,r.triangles.length,gl.UNSIGNED_INT,0);}
      else this.drawTriangles(ctx,r,opacity);
      if(this.showMesh&&!this.source&&r.layer)this.drawMesh(ctx,r.current,r.triangles,r.layer.id===this.selectedLayer);
    }
    if(this.showFill&&this.source)for(const item of this.items||[]){if(!item.mask||item.layer.visible===false)continue;if(this.selectedLayer&&item.layer.id!==this.selectedLayer)continue;const [x,y,w,h]=item.layer.bbox,[sx,sy]=this.toScreen([x,y]);ctx.drawImage(item.mask,sx,sy,w*this.scale,h*this.scale);}
    if(this.showBones){const t=this.source?setupTransforms(this.project.rig):poseTransforms(this.project.rig,this.frame);
      for(const b of this.project.rig.bones){if(b.start===b.end)continue;const a=this.toScreen([t[b.id].x,t[b.id].y]),z=this.toScreen(boneEnd(t[b.id])),selected=b.id===this.selectedBone;ctx.strokeStyle=selected?'#fff4a8':b.layer?'#b4f1d8':'#8994a2';ctx.lineWidth=selected?3:1.5;ctx.globalAlpha=b.layer?1:.55;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();}
      ctx.globalAlpha=1;
      const joints=this.source?this.project.rig.joints.map(j=>({p:[j.x,j.y],confidence:j.confidence,source:j.source})):Object.values(t).map(v=>({p:[v.x,v.y],confidence:1}));
      for(const j of joints){const p=this.toScreen(j.p);ctx.beginPath();ctx.arc(...p,4.3,0,Math.PI*2);ctx.fillStyle=j.confidence<.45?'#e7ac60':'#172a27';ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle=j.confidence<.45?'#ffd198':'#c1fce3';ctx.stroke();}
    }
    if(this.source&&this.hover&&['brush','erase'].includes(this.tool)){const p=this.toScreen(this.hover);ctx.beginPath();ctx.arc(...p,(this.brushRadius||12)*this.scale,0,Math.PI*2);ctx.strokeStyle=this.tool==='erase'?'#ff8e89':'#b3f2d7';ctx.lineWidth=1.3;ctx.stroke();}
    if(this.pointer?.type==='brush'){ctx.beginPath();this.pointer.points.forEach((p,i)=>{const s=this.toScreen(p);i?ctx.lineTo(...s):ctx.moveTo(...s);});ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=(this.brushRadius||12)*this.scale*2;ctx.strokeStyle=this.tool==='erase'?'#ff8e8966':'#b3f2d755';ctx.stroke();}
    if(this.sourceMotion)this.drawMotion(ctx);
    if(this.fallback){ctx.fillStyle='#ecb47e';ctx.font='11px system-ui';ctx.fillText('Canvas renderer · WebGL2 unavailable',12,this.height-12);}
  }
  drawMesh(ctx,vertices,triangles,selected=false){ctx.strokeStyle=selected?'#fff1a480':'#b6e4d340';ctx.lineWidth=.7;ctx.beginPath();for(let i=0;i<triangles.length;i+=3){for(let k=0;k<3;k++){const index=triangles[i+k],p=this.toScreen([vertices[index*2],vertices[index*2+1]]);k?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.closePath();}ctx.stroke();}
  drawTriangles(destination,r,opacity){
    const pos=r.current,uv=r.uvs,img=r.image;
    // Accumulate triangle coverage within ONE layer, then composite that layer.
    // Source-over on each anti-aliased triangle otherwise creates dark seams.
    if(!r.layer){const [x,y]=this.toScreen([0,0]);destination.globalAlpha=opacity;destination.drawImage(img,x,y,this.project.width*this.scale,this.project.height*this.scale);destination.globalAlpha=1;return;}
    if(!this.layerCanvas){this.layerCanvas=document.createElement('canvas');this.layerCtx=this.layerCanvas.getContext('2d');}
    const canvas=this.layerCanvas,ctx=this.layerCtx;
    if(canvas.width!==this.overlay.width||canvas.height!==this.overlay.height){canvas.width=this.overlay.width;canvas.height=this.overlay.height;}
    ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);ctx.globalAlpha=1;ctx.globalCompositeOperation='lighter';
    for(let i=0;i<r.triangles.length;i+=3){const ids=[...r.triangles.slice(i,i+3)],s=ids.map(n=>[uv[n*2]*img.width,uv[n*2+1]*img.height]),d=ids.map(n=>this.toScreen([pos[n*2],pos[n*2+1]]));
      const [s0,s1,s2]=s,[d0,d1,d2]=d,det=(s1[0]-s0[0])*(s2[1]-s0[1])-(s2[0]-s0[0])*(s1[1]-s0[1]);if(Math.abs(det)<1e-8)continue;
      const a=((d1[0]-d0[0])*(s2[1]-s0[1])-(d2[0]-d0[0])*(s1[1]-s0[1]))/det,b=((d1[1]-d0[1])*(s2[1]-s0[1])-(d2[1]-d0[1])*(s1[1]-s0[1]))/det,c=((s1[0]-s0[0])*(d2[0]-d0[0])-(s2[0]-s0[0])*(d1[0]-d0[0]))/det,e=((s1[0]-s0[0])*(d2[1]-d0[1])-(s2[0]-s0[0])*(d1[1]-d0[1]))/det;
      ctx.save();ctx.beginPath();ctx.moveTo(...d0);ctx.lineTo(...d1);ctx.lineTo(...d2);ctx.closePath();ctx.clip();ctx.transform(a,b,c,e,d0[0]-a*s0[0]-c*s0[1],d0[1]-b*s0[0]-e*s0[1]);ctx.drawImage(img,0,0);ctx.restore();
    }
    destination.globalAlpha=opacity;destination.drawImage(canvas,0,0,this.width,this.height);destination.globalAlpha=1;
  }
  drawMotion(ctx){const {frame,links}=this.sourceMotion;if(!frame)return;const pts=Object.values(frame.joints);if(!pts.length)return;const minX=Math.min(...pts.map(p=>p[0])),maxX=Math.max(...pts.map(p=>p[0])),minY=Math.min(...pts.map(p=>p[1])),maxY=Math.max(...pts.map(p=>p[1]));const scale=Math.min(155/(maxX-minX||1),170/(maxY-minY||1));const point=p=>[22+(p[0]-minX)*scale,this.height-25-(p[1]-minY)*scale];ctx.fillStyle='#0c151dda';ctx.fillRect(10,this.height-218,185,208);ctx.fillStyle='#acc0c5';ctx.font='10px system-ui';ctx.fillText('SOURCE MOTION',22,this.height-198);ctx.strokeStyle='#e9c27b';ctx.lineWidth=1.5;
    for(const [a,b] of links){if(!frame.joints[a]||!frame.joints[b])continue;ctx.beginPath();ctx.moveTo(...point(frame.joints[a]));ctx.lineTo(...point(frame.joints[b]));ctx.stroke();}for(const p of pts){ctx.beginPath();ctx.arc(...point(p),2,0,Math.PI*2);ctx.fillStyle='#f4d59c';ctx.fill();}
  }
}
