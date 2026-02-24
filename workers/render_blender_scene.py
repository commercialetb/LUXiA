# LuxIA v16: Procedural Blender scene generator + Cycles render (4:3)
#
# Run:
#   blender -b --python render_blender_scene.py -- scene.json out.png
#
import bpy
import sys, json, math
from mathutils import Vector

def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing -- args: scene.json out.png")
    idx = argv.index("--")
    args = argv[idx+1:]
    if len(args) < 2:
        raise SystemExit("Usage: -- scene.json out.png")
    return args[0], args[1]

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def set_cycles(quality):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.use_denoising = True
    if quality == "draft":
        bpy.context.scene.cycles.samples = 64
    elif quality == "high":
        bpy.context.scene.cycles.samples = 256
    else:
        bpy.context.scene.cycles.samples = 128

def set_render_size(w, h, contrast_level='medium'):
    s = bpy.context.scene
    s.render.resolution_x = int(w)
    s.render.resolution_y = int(h)
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "PNG"
    s.view_settings.view_transform = "Filmic"
    look = "Medium High Contrast"
    if contrast_level == 'low':
        look = "Medium Low Contrast"
    elif contrast_level == 'high':
        look = "Very High Contrast"
    s.view_settings.look = look

def make_material_pack(mood):
    def mat(name, base=(0.8,0.8,0.8), rough=0.6):
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*base, 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        return m

    packs = {
        "office_clean": {
            "wall": mat("wall_office", (0.92,0.92,0.92), 0.75),
            "floor": mat("floor_office", (0.55,0.55,0.57), 0.65),
            "ceiling": mat("ceiling_office", (0.96,0.96,0.96), 0.85),
        },
        "hospitality_warm": {
            "wall": mat("wall_hosp", (0.88,0.84,0.78), 0.65),
            "floor": mat("floor_hosp", (0.25,0.18,0.12), 0.55),
            "ceiling": mat("ceiling_hosp", (0.95,0.93,0.90), 0.85),
        },
        "retail_bright": {
            "wall": mat("wall_retail", (0.95,0.95,0.95), 0.70),
            "floor": mat("floor_retail", (0.45,0.45,0.48), 0.60),
            "ceiling": mat("ceiling_retail", (0.98,0.98,0.98), 0.85),
        },
        "industrial_raw": {
            "wall": mat("wall_ind", (0.65,0.65,0.68), 0.85),
            "floor": mat("floor_ind", (0.22,0.22,0.24), 0.90),
            "ceiling": mat("ceiling_ind", (0.75,0.75,0.78), 0.90),
        },
        "minimal_white": {
            "wall": mat("wall_min", (0.97,0.97,0.97), 0.80),
            "floor": mat("floor_min", (0.70,0.70,0.72), 0.75),
            "ceiling": mat("ceiling_min", (0.99,0.99,0.99), 0.90),
        }
    }
    return packs.get(mood, packs["office_clean"])

def add_room(w, d, h, mats):
    # floor
    bpy.ops.mesh.primitive_plane_add(size=1, location=(w/2, d/2, 0))
    floor = bpy.context.active_object
    floor.scale = (w/2, d/2, 1)
    floor.data.materials.append(mats["floor"])

    # ceiling
    bpy.ops.mesh.primitive_plane_add(size=1, location=(w/2, d/2, h))
    ceil = bpy.context.active_object
    ceil.scale = (w/2, d/2, 1)
    ceil.data.materials.append(mats["ceiling"])

    # walls (4)
    def wall(p1, p2):
        x1,y1 = p1; x2,y2 = p2
        length = math.sqrt((x2-x1)**2+(y2-y1)**2)
        mid = ((x1+x2)/2, (y1+y2)/2, h/2)
        bpy.ops.mesh.primitive_cube_add(size=1, location=mid)
        ob = bpy.context.active_object
        ob.scale = (length/2, 0.03, h/2)
        ang = math.atan2((y2-y1),(x2-x1))
        ob.rotation_euler = (0,0, -ang)
        ob.data.materials.append(mats["wall"])

    wall((0,0),(w,0))
    wall((w,0),(w,d))
    wall((w,d),(0,d))
    wall((0,d),(0,0))

def add_luminaire(pos, power=1.0, cct=3500):
    # simple emissive disk
    bpy.ops.mesh.primitive_circle_add(vertices=32, radius=0.12, fill_type='NGON', location=pos)
    lamp = bpy.context.active_object
    lamp.rotation_euler = (math.pi, 0, 0)

    mat = bpy.data.materials.new(name="emit")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for n in nodes: nodes.remove(n)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    em = nodes.new(type="ShaderNodeEmission")

    # crude warm/cool tint by CCT bucket
    if cct <= 3000:
        color = (1.0, 0.85, 0.72, 1.0)
    elif cct >= 4500:
        color = (0.75, 0.85, 1.0, 1.0)
    else:
        color = (0.95, 0.92, 0.85, 1.0)

    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = 35.0 * float(power)
    mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    lamp.data.materials.append(mat)

def add_fill_lights(w, d, h, mood):
    # a soft area light to help Cycles converge and feel nicer
    bpy.ops.object.light_add(type='AREA', location=(w*0.5, d*0.5, h*0.95))
    l = bpy.context.active_object
    l.data.shape = 'RECTANGLE'
    l.data.size = max(w,d)*0.9
    l.data.size_y = max(w,d)*0.9
    if mood in ("hospitality_warm",):
        l.data.color = (1.0, 0.9, 0.8)
        l.data.energy = 40
    else:
        l.data.color = (1.0, 1.0, 1.0)
        l.data.energy = 55

def add_camera_technical(w, d, h):
    bpy.ops.object.camera_add(location=(w*1.15, -d*0.55, h*1.05))
    cam = bpy.context.active_object
    cam.data.lens = 35
    cam.rotation_euler = (math.radians(60), 0, math.radians(55))
    return cam

def add_camera_client(w, d, h):
    bpy.ops.object.camera_add(location=(w*0.15, d*0.15, 1.6))
    cam = bpy.context.active_object
    cam.data.lens = 50
    # look at center
    target = Vector((w*0.55, d*0.55, 1.2))
    direction = target - Vector(cam.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    return cam

def main():
    scene_path, out_path = parse_args()
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    render = data.get("render", {})
    camera_mode = render.get("camera", "technical")
    mood = render.get("mood", "office_clean")
    quality = render.get("quality", "medium")
    w = int(render.get("width", 1600))
    h = int(render.get("height", 1200))

    room = data.get("room", {})
    rw = float(room.get("width_m", 6.0))
    rd = float(room.get("depth_m", 6.0))
    rh = float(room.get("height_m", 2.7))

    luminaires = data.get("luminaires", [])
    # expected each: {"x":..,"y":..,"z":..,"power":..,"cct":..}

    clear_scene()
    set_cycles(quality)
    set_render_size(w, h, contrast_level=str(style_pack.get('contrast_level','medium')))

    mats = make_material_pack(mood)
    add_room(rw, rd, rh, mats)
    add_fill_lights(rw, rd, rh, mood)

    # add luminaires
    for l in luminaires:
        x = float(l.get("x", rw*0.5))
        y = float(l.get("y", rd*0.5))
        z = float(l.get("z", rh-0.05))
        power = float(l.get("power", 1.0))
        cct = int(l.get("cct", 3500))
        add_luminaire((x,y,z), power=power, cct=cct)

    # camera selection
    cam = add_camera_technical(rw, rd, rh) if camera_mode == "technical" else add_camera_client(rw, rd, rh)
    bpy.context.scene.camera = cam

    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
