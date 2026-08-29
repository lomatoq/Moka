"""Original synthetic fixture with known joints; not an AI-detection showcase."""
from PIL import Image, ImageDraw
from .rig import template_rig


def make_demo():
    size = 768
    rig = template_rig(size, size, "biped", (156, 65, 456, 648))
    rig.update(provenance="synthetic-ground-truth", confidence=1.0)
    for j in rig["joints"]:
        j.update(confidence=1.0, source="synthetic-ground-truth")
    joints = {j["id"]: (j["x"], j["y"]) for j in rig["joints"]}
    layers = []
    colors = {"mint": "#99e4c7", "dark": "#223643", "suit": "#344b5b", "orange": "#ffaf65", "sole": "#172d37"}
    order = ["upper_arm_r", "forearm_r", "hand_r", "thigh_r", "shin_r", "foot_r",
             "torso", "thigh_l", "shin_l", "foot_l", "upper_arm_l", "forearm_l", "hand_l", "neck", "head"]
    bones = {b["id"]: b for b in rig["bones"]}
    for name in order:
        im = Image.new("RGBA", (size, size))
        d = ImageDraw.Draw(im)
        b = bones[name]
        a, z = joints[b["start"]], joints[b["end"]]
        if name == "head":
            d.rounded_rectangle((294, 65, 474, 226), radius=58, fill=colors["mint"], outline=colors["dark"], width=8)
            d.rounded_rectangle((310, 105, 458, 172), radius=28, fill=colors["dark"])
            for xx in (346, 422):
                d.rounded_rectangle((xx-7, 123, xx+7, 151), radius=6, fill="#d4f5df")
            d.arc((359, 169, 409, 201), 12, 168, fill=colors["dark"], width=5)
            d.rounded_rectangle((364, 52, 404, 77), radius=10, fill=colors["orange"], outline=colors["dark"], width=5)
        elif name == "torso":
            d.rounded_rectangle((304, 244, 464, 449), radius=44, fill=colors["suit"], outline=colors["dark"], width=7)
            d.rounded_rectangle((350, 274, 418, 327), radius=17, fill=colors["orange"])
            d.line((364, 312, 364, 286, 384, 304, 404, 286, 404, 312), fill=colors["dark"], width=6)
            d.rounded_rectangle((312, 408, 456, 432), radius=9, fill=colors["dark"])
        elif name == "neck":
            d.rounded_rectangle((357, 211, 411, 270), radius=17, fill=colors["orange"], outline=colors["dark"], width=5)
        else:
            radius = 26 if name.startswith(("upper", "forearm")) else 30
            if name.startswith("hand"):
                radius, color = 28, colors["mint"]
            elif name.startswith("foot"):
                radius, color = 25, colors["sole"]
            elif name.startswith("shin"):
                color = colors["suit"]
            else:
                color = colors["suit"]
            d.line((*a, *z), fill=colors["dark"], width=radius * 2 + 6)
            for xx, yy in (a, z):
                d.ellipse((xx-radius-3, yy-radius-3, xx+radius+3, yy+radius+3), fill=colors["dark"])
            d.line((*a, *z), fill=color, width=radius * 2 - 6)
            for xx, yy in (a, z):
                d.ellipse((xx-radius+3, yy-radius+3, xx+radius-3, yy+radius-3), fill=color)
            if name.startswith("forearm"):
                xx, yy = a
                d.ellipse((xx-15, yy-15, xx+15, yy+15), fill=colors["orange"])
        layers.append((name, im))
    source = Image.new("RGBA", (size, size))
    for _, layer in layers:
        source = Image.alpha_composite(source, layer)
    return source, rig, layers
