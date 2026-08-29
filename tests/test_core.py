import copy
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from moka.demo import make_demo
from moka.formats import (checked_zip, export_psd, export_spine, pack_atlas, read_psd,
                          read_psd_basic, safe_asset, spine_document, write_psd)
from moka.rig import deform_mesh, pose_transforms, setup_transforms, template_rig, validate_rig
from moka.vision import decompose, foreground, semantic_candidates, split_semantic_layers


@pytest.fixture(scope="module")
def cut(tmp_path_factory):
    root = tmp_path_factory.mktemp("character")
    image, rig, _ = make_demo()
    image.save(root/"source.png")
    layers, quality, warnings = decompose(image, rig, root, work_size=512)
    project = {"schema":"moka.project/1","width":image.width,"height":image.height,"rig":rig,
               "layers":layers,"clips":[],"source":"source.png","quality":quality,"warnings":warnings}
    return image, project, root


def composite(layers, size, root):
    result = Image.new("RGBA", size)
    for layer in sorted(layers, key=lambda l:l["order"]):
        full = Image.new("RGBA", size); full.paste(Image.open(root/layer["image"]), tuple(layer["bbox"][:2]))
        result = Image.alpha_composite(result, full)
    return np.array(result)


@pytest.mark.parametrize("preset", ["biped","creature","quadruped"])
def test_templates_valid(preset):
    rig = template_rig(768,768,preset)
    validate_rig(rig,768,768)
    assert rig["provenance"] == "template" and rig["confidence"] == 0


def test_rig_rejects_cycles():
    rig = template_rig(500,500); rig["bones"][1]["parent"] = "head"
    with pytest.raises(ValueError): validate_rig(rig,500,500)


def test_rig_rejects_nonfinite():
    rig = template_rig(500,500); rig["joints"][0]["x"] = float("nan")
    with pytest.raises(ValueError): validate_rig(rig,500,500)


def test_exact_visible_reconstruction(cut):
    image,p,root=cut
    assert p["quality"]["visible_coverage"] == 1
    assert np.array_equal(np.array(image), composite(p["layers"],image.size,root))


def test_all_expected_anatomical_parts(cut):
    _,p,_=cut
    expected={b["id"] for b in p["rig"]["bones"] if b["layer"]}
    assert {l["bone"] for l in p["layers"]}==expected
    assert len(expected)==15
    assert p["quality"]["automatic_semantic_accuracy"] is None


def test_visible_masks_disjoint(cut):
    image,p,root=cut;count=np.zeros((image.height,image.width),np.uint16)
    for l in p["layers"]:
        x,y,w,h=l["bbox"];m=np.array(Image.open(root/l["visible_mask"]))
        count[y:y+h,x:x+w]+=(m>0)
    assert count.max()==1
    assert np.array_equal(count>0,np.array(image)[...,3]>0)


def test_original_texels_unchanged(cut):
    image,p,root=cut;original=np.array(image)
    for l in p["layers"]:
        x,y,w,h=l["bbox"];m=np.array(Image.open(root/l["visible_mask"]))>0
        art=np.array(Image.open(root/l["image"]))
        assert np.array_equal(art[m],original[y:y+h,x:x+w][m])


def test_weights_normalized_and_local(cut):
    _,p,_=cut;by_id={b["id"]:b for b in p["rig"]["bones"]}
    for l in p["layers"]:
        allowed={l["bone"],by_id[l["bone"]]["parent"]}|{b["id"] for b in by_id.values() if b["parent"]==l["bone"]}
        for weights in l["mesh"]["weights"]:
            assert sum(w["weight"] for w in weights)==pytest.approx(1)
            assert all(w["bone"] in allowed and 0<=w["weight"]<=1 for w in weights)


def test_mesh_rest_identity(cut):
    _,p,_=cut
    for l in p["layers"]:
        assert np.allclose(deform_mesh(l["mesh"],p["rig"]),l["mesh"]["vertices"],atol=1e-8)


def test_fk_preserves_lengths(cut):
    _,p,_=cut;bind=setup_transforms(p["rig"])
    pose=pose_transforms(p["rig"],{"angles":{"torso":20,"upper_arm_l":-90,"forearm_l":30},"root":[30,-20]})
    for name,t in bind.items(): assert pose[name]["length"] == t["length"]
    assert pose["root"]["x"]==pytest.approx(bind["root"]["x"]+30)


def test_foreground_does_not_delete_white_interior():
    art=np.full((100,100,4),255,np.uint8);art[20:80,20:80,:3]=[30,50,80];art[40:60,40:60,:3]=255
    out,method,warnings=foreground(Image.fromarray(art))
    assert np.array(out)[0,0,3]==0
    assert np.array(out)[50,50,3]==255


def test_alpha_foreground_passthrough(cut):
    image,_,_=cut;out,_,_=foreground(image)
    assert np.array_equal(np.array(out),np.array(image))


@pytest.mark.parametrize("reader", [read_psd_basic,read_psd])
def test_psd_roundtrip_layers_and_unicode(reader):
    layers=[("задняя лапа",Image.new("RGBA",(20,30),(20,80,150,180)),(3,5)),
            ("Front",Image.new("RGBA",(15,10),(220,100,30,220)),(10,15))]
    data=write_psd(layers,(50,60));read,size,warnings=reader(data)
    assert size==(50,60);assert [l[0] for l in read]==[l[0] for l in layers]
    for actual,expected in zip(read,layers):
        assert actual[2]==expected[2]
        assert np.array_equal(np.array(actual[1]),np.array(expected[1]))


def test_psd_export_opens(cut):
    _,p,root=cut;layers,size,_=read_psd_basic(export_psd(p,root))
    assert len(layers)==15 and size==(768,768)


def spine_fk(doc, frame=None):
    result={};frame=frame or {};root=frame.get("root",[0,0]);angles=frame.get("angles",{})
    for b in doc["bones"]:
        own_angle=math.radians(b.get("rotation",0)-angles.get(b["name"],0))
        if "parent" not in b: result[b["name"]]=(root[0],-root[1],own_angle)
        else:
            x,y,a=result[b["parent"]];xx,yy=b.get("x",0),b.get("y",0)
            result[b["name"]]=(x+math.cos(a)*xx-math.sin(a)*yy,y+math.sin(a)*xx+math.cos(a)*yy,a+own_angle)
    return result


def test_spine_weighted_vertices_match_deformed_world(cut):
    _,p,_=cut;doc=spine_document(p);root=setup_transforms(p["rig"])["root"]
    frame={"angles":{"torso":13,"upper_arm_l":-72,"forearm_l":28,"root":10},"root":[11,-5]}
    transforms=spine_fk(doc,frame);names=[b["name"] for b in doc["bones"]]
    for l in p["layers"]:
        attachment=doc["skins"][0]["attachments"][l["id"]][l["id"]]
        data=attachment["vertices"];pos=0;actual=[]
        while pos<len(data):
            count=data[pos];pos+=1;x=y=0
            for _ in range(count):
                bid,lx,ly,weight=data[pos:pos+4];pos+=4;px,py,angle=transforms[names[bid]]
                x+=(px+math.cos(angle)*lx-math.sin(angle)*ly)*weight
                y+=(py+math.sin(angle)*lx+math.cos(angle)*ly)*weight
            actual.append((x,y))
        expected=deform_mesh(l["mesh"],p["rig"],frame)
        old_uv={tuple(uv):i for i,uv in enumerate(l["mesh"]["uvs"])}
        for i,(x,y) in enumerate(actual):
            uv=tuple(attachment["uvs"][i*2:i*2+2]);ex,ey=expected[old_uv[uv]]
            assert x==pytest.approx(ex-root["x"],abs=.0002)
            assert y==pytest.approx(root["y"]-ey,abs=.0002)


def test_spine_42_rotation_field_and_duration(cut):
    _,p,_=cut;p=copy.deepcopy(p)
    p["clips"]=[{"name":"Turn","frames":[{"time":0,"angles":{"torso":0}}, {"time":1.5,"angles":{"torso":-170}}]},
                 {"name":"Still","frames":[{"time":0},{"time":2}]}]
    doc=spine_document(p)
    key=doc["animations"]["Turn"]["bones"]["torso"]["rotate"][-1]
    assert key["value"]==170 and "angle" not in key
    assert doc["animations"]["Still"]["bones"]["root"]["translate"][-1]["time"]==2


def test_spine_archive_has_editable_images(cut):
    _,p,root=cut
    with zipfile.ZipFile(io.BytesIO(export_spine(p,root))) as z:
        assert {"skeleton.json","skeleton.atlas","atlas.png","IMPORT.txt"}<=set(z.namelist())
        assert len([n for n in z.namelist() if n.startswith("images/")])==15


@pytest.mark.parametrize("name",["../private.png","/tmp/secret","a/../../x","C:/secret.png","a\\b.png"])
def test_asset_traversal_rejected(tmp_path,name):
    with pytest.raises(ValueError):safe_asset(tmp_path,name)


@pytest.mark.parametrize("name",["../escape.txt","/absolute.txt","C:/escape.txt","a\\..\\x"])
def test_zip_traversal_rejected(name):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z:z.writestr(name,"bad")
    with pytest.raises(ValueError):checked_zip(b.getvalue())


def test_semantic_layers_can_split_upper_lower_and_paws(cut,tmp_path):
    image,p,_=cut
    candidates=semantic_candidates("left leg",p["rig"])
    assert {"thigh_l","shin_l","foot_l"}<=candidates
    assert not any(n.endswith("_r") for n in candidates)
    # An already completed coarse layer is not accepted as a single rigid leg.
    layers,q,warnings=split_semantic_layers([("body",image,(0,0))],image.size,p["rig"],tmp_path,provider="imported")
    assert {"thigh_l","shin_l","foot_l","thigh_r","shin_r","foot_r"}<={l["bone"] for l in layers}
    assert np.array_equal(composite(layers,image.size,tmp_path),np.array(image))


@pytest.mark.parametrize('name', [
    'front hair','back hair','headwear','face','irides','eyebrow','eyewhite',
    'eyelash','eyewear','ears','earwear','nose','mouth'])
def test_seethrough_v3_head_labels(name):
    assert semantic_candidates(name,template_rig(512,512)) == {'head'}


@pytest.mark.parametrize('name,prefixes', [
    ('neck',('neck',)),('neckwear',('neck',)),('topwear',('torso','upper_arm','forearm')),
    ('handwear',('hand',)),('bottomwear',('thigh','shin','foot')),
    ('legwear',('thigh','shin','foot')),('footwear',('foot',)),
    ('tail',('tail',)),('wings',('wing',))])
def test_seethrough_v3_clothing_not_mistaken_for_ears(name,prefixes):
    rig=template_rig(512,512,'creature')
    # A custom wing is created through the same extensible skeleton schema.
    rig['bones'].append({'id':'wing_l','parent':'torso','start':'chest','end':'elbow_l','layer':True})
    candidates=semantic_candidates(name,rig)
    assert candidates
    assert all(b.startswith(prefixes) for b in candidates)
    assert 'head' not in candidates


def test_objects_and_unrecognized_labels_keep_all_anatomical_candidates():
    rig=template_rig(512,512)
    all_ids={b['id'] for b in rig['bones'] if b['layer']}
    assert semantic_candidates('objects',rig)==all_ids
    assert semantic_candidates('qwen_layer_1',rig)==all_ids


@pytest.mark.parametrize('name,expected', [
    ('left lower arm',{'forearm_l'}),('upper arm r',{'upper_arm_r'}),
    ('thigh_l',{'thigh_l'}),('right legwear',{'thigh_r','shin_r','foot_r'}),
    ('footwear',{'foot_l','foot_r'})])
def test_anatomical_and_side_labels(name,expected):
    assert semantic_candidates(name,template_rig(512,512))==expected


def test_mask_edit_preserves_unpainted_amodal_pixels(cut,tmp_path):
    import shutil
    from moka.vision import paint_layer
    source,p,root=cut
    layer=copy.deepcopy(next(l for l in p['layers'] if l['fill_pixels']>0))
    for key in ('image','visible_mask','fill_mask'): shutil.copy(root/layer[key],tmp_path/layer[key])
    before=np.array(Image.open(tmp_path/layer['image']))
    before_fill=np.array(Image.open(tmp_path/layer['fill_mask']))
    x,y,w,h=layer['bbox']
    # No brush samples means an identity edit, including the hidden artwork.
    edited=paint_layer(source,layer,[{'points':[],'radius':2}],tmp_path,p['rig'])
    assert np.array_equal(before,np.array(Image.open(tmp_path/edited['image'])))
    assert np.array_equal(before_fill,np.array(Image.open(tmp_path/edited['fill_mask'])))
    assert edited['fill_pixels']>0


def test_psd_export_applies_layer_opacity(cut):
    _,p,root=cut
    p=copy.deepcopy(p);p['layers']=p['layers'][:1];p['layers'][0]['opacity']=.25
    layers,_,_=read_psd_basic(export_psd(p,root))
    assert np.array(layers[0][1])[...,3].max() == 64
