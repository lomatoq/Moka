import copy
import io
import json
import time
import zipfile
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from moka.server import create_app


@pytest.fixture
def client(tmp_path):
    app=create_app(tmp_path/"projects")
    with TestClient(app) as c:yield c
    app.state.store.pool.shutdown(wait=True,cancel_futures=True)


def make(client):
    r=client.post('/api/demo');assert r.status_code==200;return r.json()


def run_cut(client,p):
    r=client.post(f"/api/projects/{p['id']}/cut",json={"revision":p["revision"],"engine":"cpu"});assert r.status_code==200
    jid=r.json()["id"]
    for _ in range(300):
        j=client.get('/api/jobs/'+jid).json()
        if j["status"] in ("done","failed","cancelled"):break
        time.sleep(.02)
    assert j["status"]=="done",j
    return client.get('/api/projects/'+p['id']).json()


def test_api_health_and_frontend(client):
    assert client.get('/api/health').json()["app"]=="moka"
    assert 'CHARACTER LAB' in client.get('/').text
    assert '__THREE_BASE__' not in client.get('/').text
    assert client.get('/static/app.js').status_code==200


def test_cut_and_all_exports_roundtrip(client):
    p=run_cut(client,make(client));assert len(p['layers'])==15
    for kind in ['spine','psd','layers','project']:
        r=client.get(f"/api/projects/{p['id']}/export/{kind}");assert r.status_code==200
        assert len(r.content)>100
        if kind=='project':
            imported=client.post('/api/projects',files={'file':('roundtrip.moka',r.content,'application/zip')})
            assert imported.status_code==200,imported.text
            q=imported.json();assert q['id']!=p['id'];assert q['rig']==p['rig'];assert len(q['layers'])==15


def test_stale_revision_rejected(client):
    p=make(client);url='/api/projects/'+p['id']
    assert client.put(url,json={'revision':p['revision'],'name':'Renamed'}).status_code==200
    assert client.put(url,json={'revision':p['revision'],'name':'Overwrite'}).status_code==409
    assert client.get(url).json()['name']=='Renamed'


def test_cross_origin_write_rejected(client):
    assert client.post('/api/demo',headers={'Origin':'https://untrusted.example'}).status_code==403
    assert client.post('/api/demo',headers={'Origin':'http://testserver'}).status_code==200


def test_bad_image_rejected_without_project(client):
    r=client.post('/api/projects',files={'file':('bad.png',b'not an image','image/png')})
    assert r.status_code==400
    assert client.get('/api/projects').json()==[]


def test_clip_validation(client):
    p=make(client)
    r=client.put('/api/projects/'+p['id'],json={'revision':0,'clips':[{'name':'bad','frames':[{'time':1},{'time':0}]}]})
    assert r.status_code==400
    assert client.get('/api/projects/'+p['id']).json()['clips']==[]


def test_unavailable_model_preserves_existing_layers(client,monkeypatch):
    from moka import engines
    p=run_cut(client,make(client));before=copy.deepcopy(p['layers'])
    original=engines.capabilities
    def unavailable():
        caps=original();caps['qwen']['available']=False;return caps
    monkeypatch.setattr(engines,'capabilities',unavailable)
    r=client.post(f"/api/projects/{p['id']}/cut",json={'revision':p['revision'],'engine':'qwen'})
    assert r.status_code==400
    assert client.get('/api/projects/'+p['id']).json()['layers']==before


def test_cancel_preserves_previous_project(client,monkeypatch):
    import moka.server as server
    from moka.vision import Cancelled
    p=make(client)
    def slow(*args,**kwargs):
        for _ in range(300):
            if kwargs['cancel']():raise Cancelled('cancelled')
            time.sleep(.01)
        raise RuntimeError('Cancel did not reach the worker')
    monkeypatch.setattr(server,'decompose',slow)
    j=client.post(f"/api/projects/{p['id']}/cut",json={'revision':0,'engine':'cpu'}).json()
    assert client.post('/api/jobs/'+j['id']+'/cancel').status_code==200
    for _ in range(100):
        status=client.get('/api/jobs/'+j['id']).json()['status']
        if status=='cancelled':break
        time.sleep(.02)
    assert status=='cancelled'
    q=client.get('/api/projects/'+p['id']).json();assert q['layers']==[] and q['revision']==0


def test_mask_brush_and_old_assets_preserved(client):
    p=run_cut(client,make(client));l=p['layers'][0]
    old=client.get(f"/api/projects/{p['id']}/assets/{l['image']}").content
    x,y,w,h=l['bbox']
    r=client.post(f"/api/projects/{p['id']}/mask",json={'revision':p['revision'],'layer':l['id'],'strokes':[{'erase':True,'radius':3,'points':[[x+w/2,y+h/2]]}]})
    assert r.status_code==200,r.text
    q=r.json();assert q['layers'][0]['image']!=l['image']
    assert client.get(f"/api/projects/{p['id']}/assets/{l['image']}").content==old


def test_preset_not_falsely_reported_as_detection(client):
    p=make(client)
    r=client.post(f"/api/projects/{p['id']}/preset",json={'revision':0,'preset':'quadruped'})
    assert r.status_code==200
    assert r.json()['rig']['provenance']=='template'


def test_imported_psd_recuts_anatomically(client):
    from moka.formats import write_psd
    from moka.demo import make_demo
    image,rig,_=make_demo();data=write_psd([('body',image,(0,0))],image.size)
    r=client.post('/api/projects',files={'file':('body.psd',data,'image/vnd.adobe.photoshop')})
    assert r.status_code==200,r.text
    p=r.json();assert p['suggested_engine']=='imported'
    assert p['semantic_source']=='original.psd'


@pytest.mark.parametrize('motion', [
    {'schema':'moka.motion/1','joints':['hips'],'frames':[{'time':0,'joints':{'hips':[0,0,0,2]}}]},
    {'schema':'moka.motion/1','joints':['hips'],'frames':[{'time':0,'joints':{'absent':[0,0,0]}}]},
    {'schema':'moka.motion/1','joints':['hips'],'frames':[{'time':0,'joints':{'hips':[0,0,0]}},{'time':0,'joints':{}}]},
    {'schema':'moka.motion/1','joints':[],'frames':[]},
])
def test_invalid_source_motion_rejected(client,motion):
    p=client.post('/api/demo').json()
    response=client.put('/api/projects/'+p['id'],json={'revision':p['revision'],'source_motion':motion})
    assert response.status_code==400


def test_portable_project_rejects_invalid_layer_before_creating_project(client):
    from moka.formats import write_psd
    p=client.post('/api/demo').json()
    p['layers']=[{'id':'bad','bone':'missing','name':'bad','bbox':[0,0,2,2]}]
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('project.json',json.dumps(p))
    before=len(client.get('/api/projects').json())
    response=client.post('/api/projects',files={'file':('bad.moka',stream.getvalue(),'application/zip')})
    assert response.status_code==400
    assert len(client.get('/api/projects').json())==before
