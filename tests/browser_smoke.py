"""Real-server browser smoke tests. No API or animation outputs are mocked.

Normal: python tests/browser_smoke.py --assets
Managed offline renderer: --source-mode --browser /usr/bin/chromium
Source mode only bundles our ES modules and bridges HTTP to the same real API;
it does not test external model/Three imports or pretend to enable WebGL.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import httpx
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]

def embedded_gltf(binary=False):
    """Original two-bone, one-triangle fixture with a known 90-degree rotation."""
    chunks=[];views=[];accessors=[]
    def add(values,fmt,kind,component,count,minimum=None,maximum=None):
        offset=sum(len(c) for c in chunks)
        data=struct.pack('<'+fmt*len(values),*values)
        views.append({'buffer':0,'byteOffset':offset,'byteLength':len(data)})
        chunks.append(data+b'\0'*((-len(data))%4))
        a={'bufferView':len(views)-1,'componentType':component,'count':count,'type':kind}
        if minimum is not None:a['min']=minimum
        if maximum is not None:a['max']=maximum
        accessors.append(a);return len(accessors)-1
    pos=add([0,0,0, .1,0,0, 0,.1,0],'f','VEC3',5126,3,[0,0,0],[.1,.1,0])
    joints=add([0,0,0,0]*3,'H','VEC4',5123,3)
    weights=add([1,0,0,0]*3,'f','VEC4',5126,3)
    indices=add([0,1,2],'H','SCALAR',5123,3)
    ident=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];second=ident.copy();second[13]=-1
    inverse=add(ident+second,'f','MAT4',5126,2)
    times=add([0,1],'f','SCALAR',5126,2,[0],[1])
    quat=add([0,0,0,1,0,0,math.sqrt(.5),math.sqrt(.5)],'f','VEC4',5126,2)
    data=b''.join(chunks)
    doc={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
         'nodes':[{'name':'Scene','children':[1,3]},{'name':'Hips','children':[2]},
                  {'name':'Head','translation':[0,1,0]},{'name':'Triangle','mesh':0,'skin':0}],
         'meshes':[{'primitives':[{'attributes':{'POSITION':pos,'JOINTS_0':joints,'WEIGHTS_0':weights},'indices':indices}]}],
         'skins':[{'joints':[1,2],'skeleton':1,'inverseBindMatrices':inverse}],
         'animations':[{'name':'Known rotation','samplers':[{'input':times,'output':quat,'interpolation':'LINEAR'}],
                        'channels':[{'sampler':0,'target':{'node':1,'path':'rotation'}}]}],
         'buffers':[{'byteLength':len(data)}],'bufferViews':views,'accessors':accessors}
    if not binary:
        doc['buffers'][0]['uri']='data:application/octet-stream;base64,'+base64.b64encode(data).decode()
        return json.dumps(doc).encode()
    js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((-len(js))%4)
    return struct.pack('<III',0x46546c67,2,12+8+len(js)+8+len(data))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(data),0x004e4942)+data


def source_page(page,client,base):
    html=(ROOT/'web/index.html').read_text().replace('<head>',f'<head><base href="{base}/">')
    html=re.sub(r'<link rel="stylesheet"[^>]+>','<style>'+(ROOT/'web/styles.css').read_text()+'</style>',html)
    html=re.sub(r'<script type="importmap">.*?</script>','',html,flags=re.S)
    html=re.sub(r'<script type="module" src="/static/app.js"></script>','',html)
    def bundle(files):
        parts=[]
        for name in files:
            s=(ROOT/'web'/name).read_text()
            s=re.sub(r'^import .+?;\s*','',s,flags=re.M)
            s=re.sub(r'\bexport (?=(?:async )?(?:function|class|const|let|var)\b)','',s)
            parts.append(s)
        return '\n'.join(parts)
    def bridge(payload):
        url=payload['url'];url=base+url if url.startswith('/') else url
        if not url.startswith(base+'/'):raise ValueError('Source-mode bridge only serves the local application')
        if payload.get('form'):
            files=[];data={}
            for p in payload['form']:
                if 'data' in p:files.append((p['key'],(p['name'],base64.b64decode(p['data']),p['type'])))
                else:data[p['key']]=p['value']
            response=client.request(payload.get('method','GET'),url,data=data,files=files)
        else:response=client.request(payload.get('method','GET'),url,content=payload.get('body'),headers=payload.get('headers',{}))
        return {'status':response.status_code,'headers':dict(response.headers),'data':base64.b64encode(response.content).decode()}
    page.expose_function('mokaRequest',bridge)
    worker=bundle(['core.js','motion.js','motion-worker.js'])
    prelude="""window.fetch=async(input,options={})=>{const payload={url:String(input),method:options.method||'GET',headers:options.headers||{}};
      if(options.body instanceof FormData){payload.form=[];for(const [key,v] of options.body.entries()){if(v instanceof File){const a=new Uint8Array(await v.arrayBuffer());let s='';for(let i=0;i<a.length;i+=8192)s+=String.fromCharCode(...a.subarray(i,i+8192));payload.form.push({key,name:v.name,type:v.type,data:btoa(s)});}else payload.form.push({key,value:v});}}else payload.body=options.body;
      const r=await window.mokaRequest(payload);return new Response(Uint8Array.from(atob(r.data),c=>c.charCodeAt(0)),{status:r.status,headers:r.headers});};
      const NativeWorker=window.Worker;window.Worker=class extends NativeWorker{constructor(url,options){super(url==='/static/motion-worker.js'?URL.createObjectURL(new Blob([WORKER_SOURCE],{type:'text/javascript'})):url,url==='/static/motion-worker.js'?{type:'classic'}:options);}};""".replace('WORKER_SOURCE',json.dumps(worker))
    page.set_content(html);page.add_script_tag(content=prelude)
    page.add_script_tag(content=bundle(['core.js','motion.js','stage.js','capture.js','app.js']))


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with tempfile.TemporaryDirectory(prefix='moka-e2e-') as data,(output/'server.log').open('w') as log:
        env={**os.environ,'MOKA_DATA_DIR':data}
        process=subprocess.Popen([sys.executable,'-m','moka','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=base,timeout=60) as client:
                for _ in range(100):
                    try:
                        if client.get('/api/health').status_code==200:break
                    except httpx.RequestError:pass
                    time.sleep(.1)
                else:raise RuntimeError('Local server did not start')
                with sync_playwright() as pw:
                    options={'headless':True,'args':['--no-sandbox','--enable-unsafe-swiftshader']}
                    if args.browser:options['executable_path']=args.browser
                    browser=pw.chromium.launch(**options)
                    page=browser.new_page(viewport={'width':1600,'height':1060},device_scale_factor=1,accept_downloads=True)
                    errors=[];page.on('pageerror',lambda error:(errors.append(str(error)),print('PAGEERROR:',error,flush=True)))
                    page.on('console',lambda msg:(errors.append(msg.text),print('CONSOLE:',msg.text,flush=True)) if msg.type=='error' and 'favicon.ico' not in msg.text else None)
                    if args.source_mode:source_page(page,client,base)
                    else:page.goto(base,wait_until='networkidle')
                    page.locator('#welcome-demo').click();page.wait_for_selector('#welcome',state='hidden')
                    page.wait_for_function("!document.getElementById('cut-button').disabled")
                    pid=client.get('/api/projects').json()[0]['id'];project=client.get('/api/projects/'+pid).json()
                    # Move an actual setup joint and check persistence, not just a canvas redraw.
                    box=page.locator('#source-stage').bounding_box();scale=min(box['width']/768,box['height']/768)*.86
                    ox,oy=(box['width']-768*scale)/2,(box['height']-768*scale)/2
                    joint=next(j for j in project['rig']['joints'] if j['id']=='elbow_l')
                    x,y=box['x']+ox+joint['x']*scale,box['y']+oy+joint['y']*scale
                    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+5,y+3,steps=4);page.mouse.up()
                    page.wait_for_function("document.getElementById('save-state').textContent==='Saved locally'")
                    page.wait_for_timeout(200)
                    updated=client.get('/api/projects/'+pid).json()
                    assert next(j for j in updated['rig']['joints'] if j['id']=='elbow_l')['x']!=joint['x']
                    page.locator('#cut-button').click()
                    page.wait_for_function("document.getElementById('part-count').textContent==='15 PARTS'",timeout=30000)
                    page.wait_for_function("!document.getElementById('cut-button').disabled")
                    print('browser: cut complete',flush=True);page.locator('[data-motion=wave]').click()
                    page.wait_for_function("document.getElementById('clip-select').value!==''")
                    page.locator('#play-button').click()
                    page.locator('#scrubber').evaluate("e=>{e.value=1.4;e.dispatchEvent(new Event('input',{bubbles:true}));}")
                    page.locator('[data-tab=rig]').click();page.locator('#bone-select').select_option('forearm_l')
                    page.locator('#bone-angle').evaluate("e=>{e.value=-35;e.dispatchEvent(new Event('input',{bubbles:true}));}")
                    page.locator('#bone-select').select_option('upper_arm_l')
                    page.locator('#bone-angle').evaluate("e=>{e.value=-20;e.dispatchEvent(new Event('input',{bubbles:true}));}")
                    print('browser: editing keys',flush=True);page.locator('#add-key').click();page.wait_for_timeout(300)
                    p=client.get('/api/projects/'+pid).json();frame=next(f for f in p['clips'][0]['frames'] if abs(f['time']-1.4)<.001)
                    assert frame['angles']['forearm_l']==-35 and frame['angles']['upper_arm_l']==-20
                    # Import and retarget a real BVH generated from the original fixture.
                    bvh=subprocess.check_output(['node','--input-type=module','-e',"import fs from 'node:fs';import {demoClip} from './web/core.js';import {clipToBVH} from './web/motion.js';const r=JSON.parse(fs.readFileSync('tests/fixtures/rig.json','utf8'));process.stdout.write(clipToBVH(r,demoClip(r,'wave',2,15)));"],cwd=ROOT)
                    print('browser: importing BVH',flush=True);page.locator('#motion-file').set_input_files({'name':'fixture.bvh','mimeType':'text/plain','buffer':bvh})
                    page.wait_for_timeout(500);print('browser: motion status',page.locator('#status-message').inner_text(),flush=True);page.wait_for_function("!document.getElementById('retarget-button').disabled",timeout=15000)
                    print('browser: applying retarget',flush=True);page.locator('#retarget-button').click()
                    page.wait_for_function("document.getElementById('motion-diagnostics').textContent.includes('retargeted')",timeout=15000)
                    page.locator('#save-button').click();page.wait_for_timeout(200)
                    p=client.get('/api/projects/'+pid).json();assert len(p['clips'])==2 and p['source_motion']['type']=='bvh'
                    page.locator('#clip-select').select_option(p['clips'][0]['id'])
                    page.locator('#scrubber').evaluate("e=>{e.value=1.4;e.dispatchEvent(new Event('input',{bubbles:true}));}")
                    page.locator('[data-tab=layers]').click();page.locator('#show-mesh').uncheck()
                    page.locator('.workflow-panel').evaluate('e=>e.scrollTop=0')
                    page.wait_for_timeout(5200)
                    page.screenshot(path=str(output/'preview.png'),full_page=True)
                    with page.expect_download() as download:
                        page.locator('#export-button').click();page.locator('[data-export=spine]').click()
                    download.value.save_as(output/'spine.zip')
                    with zipfile.ZipFile(output/'spine.zip') as z:
                        doc=json.loads(z.read('skeleton.json'));assert doc['skeleton']['spine'].startswith('4.2') and len(doc['animations'])==2
                    if not args.source_mode:
                        page.reload(wait_until='networkidle');page.wait_for_selector('#welcome',state='hidden')
                        assert page.locator('#part-count').inner_text()=='15 PARTS'
                    result={'mode':'source-bundle-real-api' if args.source_mode else 'native-es-modules',
                            'renderer':page.locator('#render-status').inner_text(),'layers':15,'clips':2,
                            'tests':['setup joint persistence','separation','multi-bone key editing','BVH import and retarget','Spine download'],
                            'errors':errors}
                    if args.assets:
                        if args.source_mode:raise ValueError('--assets requires native module loading, not source mode')
                        for binary in (False,True):
                            payload={'data':base64.b64encode(embedded_gltf(binary)).decode(),'name':'fixture.glb' if binary else 'fixture.gltf'}
                            sample=page.evaluate("""async p=>{const {load3D}=await import('/static/import3d.js');const bytes=Uint8Array.from(atob(p.data),c=>c.charCodeAt(0));const scene=await load3D(new File([bytes],p.name));try{return await scene.sample(0,{fps:10});}finally{scene.dispose();}}""",payload)
                            assert len(sample['frames'])==11
                            head=sample['frames'][5]['joints']['Head'];assert abs(head[0]+math.sqrt(.5))<.002 and abs(head[1]-math.sqrt(.5))<.002
                        posefile=Path(args.pose_fixture) if args.pose_fixture else None
                        if posefile and posefile.is_file():
                            image=base64.b64encode(posefile.read_bytes()).decode()
                            detected=page.evaluate("""async base64=>{const {PoseCapture}=await import('/static/capture.js');const settings=await(await fetch('/api/capabilities')).json();const capture=new PoseCapture(settings);const image=await createImageBitmap(new Blob([Uint8Array.from(atob(base64),c=>c.charCodeAt(0))],{type:'image/jpeg'}));try{return await capture.image(image);}finally{capture.stop();image.close();}}""",image)
                            assert len(detected['landmarks'][0])==33
                            result['real_pose_landmarks']=33
                        result['tests']+=['GLTF world-space animation sampling','GLB world-space animation sampling']
                    assert not errors, errors
                    (output/'browser-report.json').write_text(json.dumps(result,indent=2))
                    print(json.dumps(result,indent=2));browser.close()
        finally:
            process.terminate()
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser');parser.add_argument('--source-mode',action='store_true')
    parser.add_argument('--assets',action='store_true');parser.add_argument('--pose-fixture')
    parser.add_argument('--output',default=str(ROOT/'test-results'))
    run(parser.parse_args())
